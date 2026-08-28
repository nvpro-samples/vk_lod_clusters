/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <new>

#include <meshoptimizer.h>

#include "threadlocal_arena.hpp"

namespace lodclusters {

static constexpr size_t ARENA_ALIGNMENT = 16;

static inline size_t alignUp(size_t value, size_t alignment)
{
  return (value + alignment - 1) & ~(alignment - 1);
}

static inline void* alignedAlloc(size_t size)
{
#if defined(_MSC_VER)
  return _aligned_malloc(size, ARENA_ALIGNMENT);
#else
  return std::aligned_alloc(ARENA_ALIGNMENT, alignUp(size, ARENA_ALIGNMENT));
#endif
}

static inline void alignedFree(void* ptr)
{
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

// per thread retention cap, see ThreadLocalArena::deallocate
static size_t s_retainBudget = ThreadLocalArena::DEFAULT_RETAIN_BUDGET;

// reporting only
static std::atomic_uint64_t s_reservedBytes     = 0;
static std::atomic_uint64_t s_peakReservedBytes = 0;

static void addReserved(int64_t delta)
{
  uint64_t reserved = s_reservedBytes.fetch_add(uint64_t(delta)) + uint64_t(delta);
  if(delta > 0)
  {
    uint64_t peak = s_peakReservedBytes.load();
    while(reserved > peak && !s_peakReservedBytes.compare_exchange_weak(peak, reserved))
    {
    }
  }
}

// every live arena, so a coarse trim can reach threads other than the caller
static std::mutex                     s_arenaRegistryMutex;
static std::vector<ThreadLocalArena*> s_arenaRegistry;

ThreadLocalArena::ThreadLocalArena()
{
  std::lock_guard<std::mutex> lock(s_arenaRegistryMutex);
  s_arenaRegistry.push_back(this);
}

ThreadLocalArena::~ThreadLocalArena()
{
  {
    std::lock_guard<std::mutex> lock(s_arenaRegistryMutex);
    s_arenaRegistry.erase(std::find(s_arenaRegistry.begin(), s_arenaRegistry.end(), this));
  }

  for(Chunk* chunk : m_chunks)
  {
    destroyChunk(chunk);
  }
}

ThreadLocalArena::Chunk* ThreadLocalArena::createChunk(size_t minCapacity)
{
  size_t capacity = std::max(minCapacity, DEFAULT_CHUNK_SIZE);
  Chunk* chunk    = reinterpret_cast<Chunk*>(alignedAlloc(sizeof(Chunk) + capacity));
  if(!chunk)
    throw std::bad_alloc();

  chunk->capacity = capacity;
  chunk->used     = 0;

  m_reserved += sizeof(Chunk) + capacity;
  addReserved(int64_t(sizeof(Chunk) + capacity));
  return chunk;
}

void ThreadLocalArena::destroyChunk(Chunk* chunk)
{
  size_t total = sizeof(Chunk) + chunk->capacity;
  m_reserved -= total;
  addReserved(-int64_t(total));
  alignedFree(chunk);
}

void* ThreadLocalArena::allocate(size_t size)
{
  size_t need = sizeof(Header) + alignUp(size, ARENA_ALIGNMENT);

  if(m_chunks.empty())
  {
    m_chunks.push_back(createChunk(need));
    m_topChunk = 0;
  }

  Chunk* chunk = m_chunks[m_topChunk];
  if(chunk->used + need > chunk->capacity)
  {
    // current chunk is full, move to the next retained one or grow the list
    size_t next = m_topChunk + 1;
    if(next < m_chunks.size() && m_chunks[next]->capacity < need)
    {
      // retained chunk is too small for this request, replace it
      destroyChunk(m_chunks[next]);
      m_chunks[next] = createChunk(need);
    }
    else if(next >= m_chunks.size())
    {
      m_chunks.push_back(createChunk(need));
    }

    m_topChunk  = next;
    chunk       = m_chunks[next];
    chunk->used = 0;
  }

  Header* header     = reinterpret_cast<Header*>(chunkData(chunk) + chunk->used);
  header->chunkIndex = m_topChunk;
  header->prevUsed   = chunk->used;
  chunk->used += need;

  m_liveBytes += need;
  m_peakLive = std::max(m_peakLive, m_liveBytes);

  return reinterpret_cast<uint8_t*>(header) + sizeof(Header);
}

void ThreadLocalArena::deallocate(void* ptr)
{
  if(!ptr)
    return;

  Header* header = reinterpret_cast<Header*>(ptr) - 1;

  m_topChunk   = size_t(header->chunkIndex);
  Chunk* chunk = m_chunks[m_topChunk];

  m_liveBytes -= chunk->used - size_t(header->prevUsed);
  chunk->used = size_t(header->prevUsed);

  // The whole-mesh calls at the start of a lod build reserve far more than the
  // per-group work that follows. Hand those back rather than pinning them on this
  // thread for the rest of the geometry, they are rare enough that re-reserving
  // them costs little.
  if(m_liveBytes == 0 && m_reserved > s_retainBudget)
  {
    trim(s_retainBudget);
  }
}

void ThreadLocalArena::trim(size_t keepBytes)
{
  if(m_liveBytes != 0)
    return;

  while(!m_chunks.empty() && m_reserved > keepBytes)
  {
    destroyChunk(m_chunks.back());
    m_chunks.pop_back();
  }

  m_topChunk = 0;
  if(!m_chunks.empty())
    m_chunks[0]->used = 0;
}

// Created on first use so threads that never call meshoptimizer allocate nothing,
// and destroyed on thread exit.
static ThreadLocalArena& threadArena()
{
  static thread_local ThreadLocalArena arena;
  return arena;
}

static void* MESHOPTIMIZER_ALLOC_CALLCONV arenaAllocate(size_t size)
{
  return threadArena().allocate(size);
}

static void MESHOPTIMIZER_ALLOC_CALLCONV arenaDeallocate(void* ptr)
{
  threadArena().deallocate(ptr);
}

void threadLocalArenaInstall(size_t retainBudget)
{
  s_retainBudget = retainBudget;
  meshopt_setAllocator(arenaAllocate, arenaDeallocate);
}

void threadLocalArenaTrimAll()
{
  std::lock_guard<std::mutex> lock(s_arenaRegistryMutex);
  for(ThreadLocalArena* arena : s_arenaRegistry)
  {
    // release everything, processing is over and the next run re-reserves
    arena->trim(0);
  }
}

void threadLocalArenaTrimThread()
{
  threadArena().trim();
}

size_t threadLocalArenaReservedBytes()
{
  return size_t(s_reservedBytes.load());
}

size_t threadLocalArenaPeakReservedBytes()
{
  return size_t(s_peakReservedBytes.load());
}

size_t threadLocalArenaThreadLiveBytes()
{
  return threadArena().liveBytes();
}

}  // namespace lodclusters
