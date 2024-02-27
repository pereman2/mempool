#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>
#include <cstring>
#include <mutex>
#include <vector>
#include <x86intrin.h>
#include <cassert>
#include <sys/mman.h>
#include <gperftools/tcmalloc.h>

static const uint64_t NTHREADS =  1;
static const uint64_t TOTAL_MEMORY = ((uint64_t)1024 * 1024 * 1024 * 4) / NTHREADS;
static size_t page_size = sysconf(_SC_PAGE_SIZE);

enum pool_index_t {
  START
};
template<pool_index_t pool_ix, typename T>
struct CephMemoryPoolAllocator {
  // static const char reserved_page_memory[4096 * 10] __attribute__((aligned(4096)));
  // TODO(pere): add arena + page support, CEPH_PAGE_SIZE, alloc_aligned etc.
  struct FreeNode {
    FreeNode *next;
  };

  struct ThreadPage {
    char *memory;
    size_t capacity;
  };

  struct ThreadArena {
    std::mutex lock;
    std::atomic<size_t> m;
    std::vector<ThreadPage> pages;
    FreeNode *head;
  };

  // ThreadPageHeader header is a utility struct to define the structure of the first 8 bytes of a page that must map to the pointer
  // of the arena itself to allow for easy deallocation in different threads.
  struct ThreadPageHeader {
    ThreadArena* arena_pointer; // pointer to *ThreadArena
  };

  static ThreadArena* get_arena() {
    static thread_local ThreadArena arena;
    return &arena;
  }

  // default constructors
  CephMemoryPoolAllocator(const CephMemoryPoolAllocator& other) {
  };
  CephMemoryPoolAllocator()
  {
  }

  ~CephMemoryPoolAllocator() {
  }

  void* allocate() {
    ThreadArena* arena = get_arena();
    // std::lock_guard<std::mutex> l(arena->lock);
    arena->lock.lock();

    if (arena->pages.empty() || arena->head == nullptr) {
      // get page
      size_t number_of_pages_per_alloc = 1;
      char* memory = (char*)mmap(nullptr, page_size * number_of_pages_per_alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      for (int page_number = 0; page_number < number_of_pages_per_alloc; page_number++) {
        ThreadPage page;
        page.capacity = page_size;
        page.memory = memory + (page_number * page_size);

        // update header of page
        ThreadPageHeader* header = (ThreadPageHeader*)page.memory;
        header->arena_pointer = arena;

        // Build free list of newly allocated memory
        FreeNode* node = (FreeNode*)page.memory + sizeof(T);
        node->next = nullptr;
        arena->head = node;
        for (size_t i = sizeof(T); i < page_size; i += sizeof(T)) {
          node->next = (FreeNode*)(page.memory + i);
          node = node->next;
          node->next = nullptr;
        }
        arena->pages.push_back(page);
      }
    }
    FreeNode* node = arena->head;
    arena->head = node->next;
    arena->lock.unlock();
    return (void*)node;
  }

  ThreadArena* get_page_arena(void* pointer) {
    ThreadPageHeader* header = (ThreadPageHeader*)((size_t)pointer & ~(page_size - 1));
    return header->arena_pointer;
  }

  void deallocate(void* pointer) {
    ThreadArena* page_arena = get_page_arena(pointer);
    page_arena->lock.lock();
    // std::l(char*)aligned_alloc(page_size, page_size*10);ock_guard<std::mutex> l(page_arena->lock);
    FreeNode* node = (FreeNode*)pointer;
    node->next = page_arena->head;
    page_arena->head = node;
    page_arena->lock.unlock();
  }
};


void no_optimize_away(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

struct Blob {
  int a;
  int b;
  bool d;
  uint64_t c;
};


void test(char* label) {
  CephMemoryPoolAllocator<START, Blob> pool;
  auto start = __rdtsc();
  uint64_t objs = TOTAL_MEMORY / sizeof(Blob);
  std::vector<std::thread> threads;
  int nthreads = NTHREADS;
  for (int i = 0; i < nthreads; i++) {
    threads.push_back(std::thread([&] (){
      std::vector<void*> pointers;
      for (int i = 0; i < objs; i++) {
        Blob* p = (Blob*)pool.allocate();
        if (p == nullptr) {
          printf("Out of memory\n");
          break;
        }
        pointers.push_back((void*)p);
        p->a = rand() % 1000;
        no_optimize_away(p);
      }
      for (void* p : pointers) {
        pool.deallocate((void*)p);
      }
    }));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto end = __rdtsc();
  printf("Time %10s: %20lu\n", label, end - start);
}



void test_malloc(char* label) {
  auto start = __rdtsc();
  uint64_t objs = TOTAL_MEMORY / sizeof(Blob);
  std::vector<std::thread> threads;
  int nthreads = NTHREADS;
  for (int i = 0; i < nthreads; i++) {
    threads.push_back(std::thread([&] () {
      std::vector<void*> pointers;
      for (int i = 0; i < objs; i++) {
        Blob* p = (Blob*)malloc(sizeof(Blob));
        if (p == nullptr) {
          printf("Out of memory\n");
          break;
        }
        pointers.push_back((void*)p);
        p->a = rand() % 1000;
        no_optimize_away(p);
      }
      for (void* p : pointers) {
        free((Blob*)p);
      }
    }));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto end = __rdtsc();
  printf("Time %10s: %20lu\n", label, end - start);
}

void test_tcmalloc(char* label) {
  auto start = __rdtsc();
  uint64_t objs = TOTAL_MEMORY / (uint64_t)sizeof(Blob);
  std::vector<std::thread> threads;
  int nthreads = NTHREADS;
  for (int i = 0; i < nthreads; i++) {
    threads.push_back(std::thread([&] () {
      std::vector<void*> pointers;
      for (int i = 0; i < objs; i++) {
        Blob* p = (Blob*)tc_malloc(sizeof(Blob));
        if (p == nullptr) {
          printf("Out of memory\n");
          break;
        }
        pointers.push_back((void*)p);
        p->a = rand() % 1000;
        no_optimize_away(p);
      }
      for (void* p : pointers) {
        tc_free((Blob*)p);
      }
    }));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto end = __rdtsc();
  printf("Time %10s: %20lu\n", label, end - start);
}

struct Op {
  enum Type {
    ALLOC,
    DEALLOC
  };
  Type type;
  void* pointer;
};

int main() {
  uint64_t objs = TOTAL_MEMORY / sizeof(Blob);
  uint64_t remaining = objs;
  std::vector<Op> ops;
  std::vector<void*> to_dealloc;
  std::mutex to_dealloc_lock;
  uint64_t seed = 0;
  srand(seed);
  while(remaining > 0) {
    ops.push_back({Op::ALLOC, nullptr});
  }
  printf("Page size: %8lu, threads: %8lu, memory: %10lu GB\n", page_size, NTHREADS, TOTAL_MEMORY/1024/1024/1024);
  test("CephMemoryPoolAllocator");
  test_malloc("tcmalloc");
  // test_tcmalloc("tcmalloc");
  return 0;
}
