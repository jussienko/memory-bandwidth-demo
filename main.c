// Copyright 2013 Alex Reece.
//
// A simple memory bandwidth profiler.
//
// Each of the write_memory_* functions read from a 1GB array. Each of the
// write_memory_* writes to the 1GB array. The goal is to get the max memory
// bandwidth as advertised by the intel specs: 23.8 GiB/s (http://goo.gl/r8Aab)

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "./functions.h"

#define SAMPLES 5
#define TIMES 5
#define BYTES_PER_GiB (1024*1024*1024LL)
#define BYTES_PER_GB (1000*1000*1000LL)
#define SIZE (1*BYTES_PER_GiB)
#define PAGE_SIZE (1<<12)


// Compute the bandwidth in GB/s.
static inline double to_bw(size_t bytes, double secs) {
  double size_bytes = (double) bytes;
  double size_gb = size_bytes / ((double) BYTES_PER_GB);
  return size_gb / secs;
}

// Time a function, printing out time to perform the memory operation and
// the computed memory bandwidth. Use openmp to do threading (set environment
// variable OMP_NUM_THREADS to control threads use.
#define timefun(f, array) timeitp(f, array, #f)
void timeitp(void (*function)(void*, size_t), char* array, char* name) {
  double min = INFINITY;
  size_t i;
  for (i = 0; i < SAMPLES; i++) {
    double before, after, total;

    size_t chunk_size = SIZE;
#pragma omp parallel
    {
#pragma omp barrier
#pragma omp master
      before = omp_get_wtime();
      int j;
      for (j = 0; j < TIMES; j++) {
	function(&array[chunk_size * omp_get_thread_num()], chunk_size);
      }
#pragma omp barrier
#pragma omp master
      after = omp_get_wtime();
    }

    total = after - before;
    if (total < min) {
      min = total;
    }
  }

  printf("%28s: %5.2f GB/s\n", name, to_bw(omp_get_max_threads() * SIZE * TIMES, min));
}

#define timefun2(f, array, array2) timeitp2(f, array, array2, #f)
void timeitp2(void (*function)(void*, void*, size_t), char* array,
              char* array2, char* name) {
  double min = INFINITY;
  size_t i;
  for (i = 0; i < SAMPLES; i++) {
    double before, after, total;

    size_t chunk_size = SIZE;
#pragma omp parallel
    {
#pragma omp barrier
#pragma omp master
      before = omp_get_wtime();
      int j;
      for (j = 0; j < TIMES; j++) {
	function(&array[chunk_size * omp_get_thread_num()], 
                 &array2[chunk_size * omp_get_thread_num()], chunk_size);
      }
#pragma omp barrier
#pragma omp master
      after = omp_get_wtime();
    }

    total = after - before;
    if (total < min) {
      min = total;
    }
  }

  printf("%28s: %5.2f GB/s\n", name, to_bw(omp_get_max_threads() * SIZE * TIMES, min));
}

int main() {

  char *array, *array2;
  
  int nthreads = omp_get_max_threads();
  printf("# Running with %d threads\n", nthreads);
  double size_bytes = (double) SIZE;
  double size_gb = size_bytes / ((double) BYTES_PER_GB);

  printf("# Array size per thread %5.3f GB\n", size_gb);
  
  // Arrays must be at least 64 byte aligned for AVX512.
  // Have PAGE_SIZE buffering so we don't have to do math for prefetching.
  posix_memalign((void *) &array, 64, omp_get_max_threads() * SIZE + PAGE_SIZE);
  posix_memalign((void *) &array2, 64, omp_get_max_threads() * SIZE + PAGE_SIZE);

  size_t chunk_size = SIZE;
// First touch
#pragma omp parallel
    {
       memset(&array[chunk_size * omp_get_thread_num()], 0xFF, chunk_size); 
       memset(&array2[chunk_size * omp_get_thread_num()], 0xFE, chunk_size); 
    }

  timefun(read_memory_rep_lodsq, array);
  timefun(read_memory_loop, array);
#ifdef __SSE4_1__
  timefun(read_memory_sse, array);
#endif
#ifdef __AVX__
  timefun(read_memory_avx, array);
  timefun(read_memory_prefetch_avx, array);
#endif

  timefun(write_memory_rep_stosq, array);  
  timefun(write_memory_loop, array);
#ifdef __SSE4_1__
  timefun(write_memory_sse, array);
  timefun(write_memory_nontemporal_sse, array);
#endif
#ifdef __AVX__
  timefun(write_memory_avx, array);
  timefun(write_memory_nontemporal_avx, array);
  timefun2(memcpy_avx, array, array2);
  timefun2(memcpy_nontemporal_avx, array, array2);
#endif
  timefun(write_memory_memset, array);
  return 0;
}
