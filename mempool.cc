#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <pthread.h>
#include <queue>
#include <thread>
#include <unistd.h>
#include <cstring>
#include <mutex>
#include <vector>
#include <x86intrin.h>
#include <cassert>
#include <sys/mman.h>
#include <gperftools/tcmalloc.h>


// #define DEBUG
#ifdef DEBUG
#define ASSERT(cond) assert(cond)
#else
#define ASSERT(cond)
#endif

static uint64_t NTHREADS =  1;
static uint64_t N_DEALLOCATE_THREADS =  1;
// static const uint64_t TOTAL_MEMORY = ((uint64_t)1024 * 1024) / NTHREADS;
static uint64_t TOTAL_MEMORY = ((uint64_t)1024 * 1024 * 1024*10 ) / NTHREADS;
static size_t page_size = sysconf(_SC_PAGE_SIZE);

enum OpType {
  ALLOC,
  DEALLOC,
  NOOP
};
struct Op {
  OpType type;
  void* pointer;
};

struct QueueNode {
  OpType type;
  void *pointer;
};

struct Queue {
  std::queue<QueueNode> queue;
  std::mutex mutex;

  void push(void* pointer) {
    std::lock_guard<std::mutex> l(mutex);
    QueueNode node;
    node.pointer = pointer;
    node.type = DEALLOC;
    queue.push(node);
  }

  QueueNode pop() {
    std::lock_guard<std::mutex> l(mutex);
    if (queue.empty()) {
      QueueNode node;
      node.type = NOOP;
      return node;
    }
    QueueNode node = queue.front();
    queue.pop();
    return node;
  }
};


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
    std::vector<ThreadPage> pages;
    std::atomic<FreeNode*> head;
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

    if (arena->pages.empty() || arena->head == nullptr) {
      // get page
      size_t number_of_pages_per_alloc = 1;
      // char* memory = (char*)mmap(nullptr, page_size * number_of_pages_per_alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      char* memory = (char*)tc_malloc(page_size * number_of_pages_per_alloc);
      ASSERT((size_t)memory % page_size == 0);
      for (int page_number = 0; page_number < number_of_pages_per_alloc; page_number++) {
        ThreadPage page;
        page.capacity = page_size;
        page.memory = memory + (page_number * page_size);

        // update header of page
        ThreadPageHeader* header = (ThreadPageHeader*)page.memory;
        header->arena_pointer = arena;

        // Build free list of newly allocated memory
        FreeNode* node = (FreeNode*)page.memory + sizeof(T);
        FreeNode* new_head = node;
        node->next = nullptr;
        for (size_t i = sizeof(T); i < page_size; i += sizeof(T)) {
          node->next = (FreeNode*)(page.memory + i);
          node = node->next;
          node->next = nullptr;
        }

        FreeNode* head = arena->head.load();
        do {
          head = arena->head.load();
          node->next = head;
          // printf("allocate: new head %p %lu\n", new_head, pthread_self());
          // printf("check %p %p\n", new_head, new_head->next);
        } while(!arena->head.compare_exchange_weak(head, new_head));
        arena->pages.push_back(page);
      }
    }
    FreeNode* node = arena->head.load();
    while(arena->head.compare_exchange_weak(node, node->next) == false) {
      node = arena->head.load();
    }
    return (void*)node;
  }

  ThreadArena* get_page_arena(void* pointer) {
    ThreadPageHeader* header = (ThreadPageHeader*)((size_t)pointer & ~(page_size - 1));
    return header->arena_pointer;
  }

  void deallocate(void* pointer) {
    ThreadArena* page_arena = get_page_arena(pointer);
    // page_arena->lock.lock();
    // std::l(char*)aligned_alloc(page_size, page_size*10);ock_guard<std::mutex> l(page_arena->lock);
    FreeNode* node = (FreeNode*)pointer;
    FreeNode* head_node = page_arena->head.load();
    node->next = head_node;
    while(page_arena->head.compare_exchange_weak(head_node, node) == false) {
      head_node = page_arena->head.load();
      node->next = head_node;
    }
    // printf("free: new head %p %lu\n", node, pthread_self());
    // printf("check %p %p\n", node, node->next);
    // page_arena->lock.unlock();
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


uint64_t test(char* label, Queue& queue, std::vector<Op>& ops) {
  CephMemoryPoolAllocator<START, Blob> pool;
  std::atomic_uint64_t atomic_cycles(0);
  uint64_t objs = TOTAL_MEMORY / (uint64_t)sizeof(Blob);
  std::vector<std::thread> threads;
  int nthreads = NTHREADS;
  std::mutex mutex;
  for (int t = 0; t < nthreads; t++) {
    int thread_id = t;
    threads.push_back(std::thread([&, thread_id] () {
      std::vector<void*> pointers;
      while (true) {
        {
          std::lock_guard<std::mutex> l(mutex);
          if (ops.size() == 0) {
            break;
          }
          ops.pop_back();
        }
        auto start = __rdtsc();
        Blob* p = (Blob*)pool.allocate();
        auto end = __rdtsc();
        ASSERT(p != nullptr);
        p->a = rand() % 1000;
        atomic_cycles += end - start;
        no_optimize_away(p);
        queue.push(p);
      }
    }));
  }
  for (auto& t : threads) {
    t.join();
  }
  return atomic_cycles.load();
}



uint64_t test_malloc(char* label, Queue& queue, std::vector<Op>& ops) {
  std::atomic_uint64_t atomic_cycles(0);
  uint64_t objs = TOTAL_MEMORY / (uint64_t)sizeof(Blob);
  std::vector<std::thread> threads;
  int nthreads = NTHREADS;
  std::mutex mutex;
  for (int t = 0; t < nthreads; t++) {
    int thread_id = t;
    threads.push_back(std::thread([&, thread_id] () {
      std::vector<void*> pointers;
      while (true) {
        {
          std::lock_guard<std::mutex> l(mutex);
          if (ops.size() == 0) {
            break;
          }
          ops.pop_back();
        }
        auto start = __rdtsc();
        Blob* p = (Blob*)malloc(sizeof(Blob));
        auto end = __rdtsc();
        ASSERT(p != nullptr);
        p->a = rand() % 1000;
        atomic_cycles += end - start;
        no_optimize_away(p);
        queue.push(p);
      }
    }));
  }
  for (auto& t : threads) {
    t.join();
  }
  return atomic_cycles.load();
}

uint64_t test_tcmalloc(char* label, Queue& queue, std::vector<Op>& ops) {
  std::atomic_uint64_t atomic_cycles(0);
  uint64_t objs = TOTAL_MEMORY / (uint64_t)sizeof(Blob);
  std::vector<std::thread> threads;
  int nthreads = NTHREADS;
  std::mutex mutex;
  for (int t = 0; t < nthreads; t++) {
    int thread_id = t;
    threads.push_back(std::thread([&, thread_id] () {
      std::vector<void*> pointers;
      auto offset = thread_id*(ops.size() / NTHREADS);
      auto fin = ops.size() / NTHREADS;
      if (thread_id == NTHREADS - 1) {
        fin = ops.size();
      }
      // printf("%d %llu %llu\n", thread_id, offset, fin);
      while (true) {
        {
          std::lock_guard<std::mutex> l(mutex);
          if (ops.size() == 0) {
            break;
          }
          ops.pop_back();
        }
        auto start = __rdtsc();
        Blob* p = (Blob*)tc_malloc(sizeof(Blob));
        auto end = __rdtsc();
        ASSERT(p != nullptr);
        p->a = rand() % 1000;
        atomic_cycles += end - start;
        no_optimize_away(p);
        queue.push(p);

        // pointers.push_back((void*)p);
      }
      // for (void* p : pointers) {
      //   tc_free((Blob*)p);
      // }
    }));
  }
  for (auto& t : threads) {
    t.join();
  }
  return atomic_cycles.load();
}


int main(int argc, char** argv) {
  std::vector<Op> ops;
  std::atomic_int counter = 0;
  Queue queue;


  char *label = argv[1];
  NTHREADS = atoi(argv[2]);
  N_DEALLOCATE_THREADS = atoi(argv[3]);
  TOTAL_MEMORY = ((uint64_t)1024 * 1024 * 1024*atoi(argv[4]) ) / NTHREADS;

  Op op;
  op.type = ALLOC;
  op.pointer = nullptr;
  for (uint64_t i = 0; i < TOTAL_MEMORY / sizeof(Blob); i++) {
    ops.push_back(op);
  }

  std::vector<std::thread> de_threads;
  std::atomic_uint64_t *atomic_cycles = new std::atomic_uint64_t(0);
  for (int i = 0; i < N_DEALLOCATE_THREADS; i++) {
    de_threads.push_back(std::thread([&] () {
      CephMemoryPoolAllocator<START, Blob> pool;
      while (counter.load() < ops.size()) {
        QueueNode node = queue.pop();
        if (node.type == NOOP) {
          continue;
        }
        uint64_t start = __rdtsc();
        if (strcmp(label, "tcmalloc") == 0) {
          tc_free(node.pointer);
        } else if (strcmp(label, "malloc") == 0) {
          free(node.pointer);
        } else {
          pool.deallocate(node.pointer);
        }
        uint64_t end = __rdtsc();
        atomic_cycles->fetch_add(end - start);
        counter++;
      }
    }));
  }

  printf("Page size: %8lu, allocate threads: %8lu, deallocate threads: %8lu, memory: %10lu GB\n", page_size, NTHREADS, N_DEALLOCATE_THREADS, TOTAL_MEMORY/1024/1024/1024);
  if (strcmp(label, "tcmalloc") == 0) {
    atomic_cycles->fetch_add(test_tcmalloc("tcmalloc", queue, ops));
  }
  if (strcmp(label, "malloc") == 0) {
    atomic_cycles->fetch_add(test_malloc("malloc", queue, ops));
  }
  if (strcmp(label, "CephMemoryPoolAllocator") == 0) {
    atomic_cycles->fetch_add(test("CephMemoryPoolAllocator", queue, ops));
  }

  for (auto &thread : de_threads) {
    thread.join();
  }
  uint64_t cycles = atomic_cycles->load();
  printf("%30s cycles %lu\n", label, cycles);
  return 0;
}
