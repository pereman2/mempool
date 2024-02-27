
#!/bin/bash
#

for t in tcmalloc CephMemoryPoolAllocator;
do
  if [ $t == "tcmalloc" ]; then
    clang++ mempool.cc -o out -ggdb -O2 -ltcmalloc
    echo "Running with tcmalloc"
  elif [ $t == "malloc" ]; then
    echo "Running with malloc"
    clang++ mempool.cc -o out -ggdb -O2
  else
    echo "Running with CephMemoryPoolAllocator"
    clang++ mempool.cc -o out -ggdb -O2 -ltcmalloc
  fi
  for threads in 1 2 4 8 16;
  do
    for de_threads in 1 2 4;
    do 
      ./out $t $threads $de_threads 2
    done
  done
done
