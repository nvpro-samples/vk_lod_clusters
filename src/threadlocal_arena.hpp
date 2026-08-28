/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lodclusters {

// Stack allocator meant to be held per thread: allocation is a pointer increment,
// deallocation rewinds. Each allocation carries a header with its chunk and the previous
// offset, which is what allows freeing individually instead of only resetting the whole
// arena. Only valid for callers that free in strict stack order, last allocated first.
// Chunks are kept for reuse, so the steady state is one high-water buffer per thread
// rather than millions of heap round trips contending on the global allocator lock.
//
// For a description of the pattern, including the previous-offset header, see
// gingerBill's "Memory Allocation Strategies", part 3:
// https://www.gingerbill.org/article/2019/02/15/memory-allocation-strategies-003/
class ThreadLocalArena
{
public:
  static constexpr size_t DEFAULT_CHUNK_SIZE    = 1024 * 1024;
  static constexpr size_t DEFAULT_RETAIN_BUDGET = 8 * 1024 * 1024;

  ThreadLocalArena();
  ~ThreadLocalArena();

  void* allocate(size_t size);
  void  deallocate(void* ptr);

  // Releases retained chunks down to `keepBytes`. Only acts while the arena is empty.
  void trim(size_t keepBytes = DEFAULT_CHUNK_SIZE);

  size_t reservedBytes() const { return m_reserved; }
  size_t peakLiveBytes() const { return m_peakLive; }
  size_t liveBytes() const { return m_liveBytes; }

private:
  struct Chunk
  {
    size_t capacity;
    size_t used;
    // payload follows
  };

  // precedes every allocation and records where deallocate rewinds to
  struct Header
  {
    uint64_t chunkIndex;
    uint64_t prevUsed;
  };

  static uint8_t* chunkData(Chunk* chunk) { return reinterpret_cast<uint8_t*>(chunk) + sizeof(Chunk); }

  Chunk* createChunk(size_t minCapacity);
  void   destroyChunk(Chunk* chunk);

  std::vector<Chunk*> m_chunks;
  size_t              m_topChunk  = 0;  // chunk we currently bump from
  size_t              m_reserved  = 0;
  size_t              m_liveBytes = 0;
  size_t              m_peakLive  = 0;
};

// Installs the arena as meshoptimizer's allocator, process wide.
//
// `meshopt_setAllocator` is documented to only ever be used for temporary allocations,
// freed in stack order, last allocated first. That is precisely the contract the stack
// allocator above needs, so no reset hook or lifetime tracking is required on our side:
// the arena drains itself as each meshopt_ call unwinds. Cluster lod building leans on
// this heavily, it is otherwise millions of small allocations across all worker threads.
//
// Must run before any meshopt_ call, the setter is explicitly not thread safe. Arenas
// are created lazily per thread, so threads that never touch meshoptimizer pay nothing.
void threadLocalArenaInstall(size_t retainBudget = ThreadLocalArena::DEFAULT_RETAIN_BUDGET);

// Releases this thread's retained chunks. Call at a coarse boundary, e.g. once a
// geometry's lod build is done.
void threadLocalArenaTrimThread();

// Trims every arena, not just the caller's; inner-parallel builds leave worker arenas
// retained. Only safe once all workers finished their meshoptimizer work.
void threadLocalArenaTrimAll();

// Aggregated across every thread that created an arena, for reporting.
size_t threadLocalArenaReservedBytes();
size_t threadLocalArenaPeakReservedBytes();

// Non-zero here means this thread's arena still has an outstanding allocation,
// i.e. some caller didn't free in strict stack order. trim() is a no-op until
// this returns to 0, so the thread's reserved chunks won't shrink until it does.
size_t threadLocalArenaThreadLiveBytes();

}  // namespace lodclusters
