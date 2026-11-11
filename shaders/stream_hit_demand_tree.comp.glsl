/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Faster stream-in demand propagation for hit-based streaming. Instead of
  touching every group, it only visits the groups the hit surface reaches.

  A multi-pass BFS over the (now free) traversalNodeInfos queue, seeded by
  stream_hit_propagation from the resident hit frontier. Each pass pushes a
  queued group's demand one lod level finer into its finer overlap groups and
  enqueues any child it advanced. A child still within
  STREAMING_HITLOD_DEMAND_REFRESH and not demanding finer detail is pruned, so a
  steady view does not re-flood every frame.

  A thread loads one queued group, but a group's finer-child count
  (`countFine`) is highly divergent across a subgroup. Rather than looping the
  children per-thread (few fat tasks stalling the whole subgroup while thin ones
  idle), the subgroup pools all its children and processes them in densely
  packed warp-wide runs - the same technique as `processAllSubTasks` in
  `traversal_run.comp.glsl`.
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_memory_scope_semantics : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;
};

////////////////////////////////////////////

layout(local_size_x=STREAM_HIT_DEMAND_TREE_WORKGROUP) in;

////////////////////////////////////////////

#ifndef USE_HIT_STREAMING
#define USE_HIT_STREAMING 1
#endif
#ifndef USE_HIT_STREAMING_LOD
#define USE_HIT_STREAMING_LOD 1
#endif

#include "streaming.glsl"

////////////////////////////////////////////

// Wrap-safe relaxation of a finer child's demand, stamped at the current frame
// (invWidth inverted: larger = finer). Returns true when the demand was advanced
// (re-stamped after going stale, or pulled finer) so the child should be enqueued;
// false when the child is still fresh and no finer width is offered, leaving its
// subtree valid and un-reflooded.
bool stream_hit_demand_relax(uint32s_inout groupDemand, uint childIndex, uint invWidth, uint cyclicFrame)
{
  uint desiredWidth = invWidth & STREAMING_HITFRAME_WIDTH_MASK;
  uint old          = groupDemand.d[childIndex];
  [[loop]] while (true)
  {
    // cyclic delta stays correct across the 15-bit frame wrap
    uint threshold   = STREAMING_HITLOD_DEMAND_REFRESH;
    bool fresh       = streamingHitFrameDelta(old) <= threshold;
    uint storedWidth = old & STREAMING_HITFRAME_WIDTH_MASK;
    bool finer       = desiredWidth > storedWidth;

    if (fresh && !finer)
      return false;  // fresh and no finer demand: prune, subtree is up to date

    // stale -> take our width; fresh but finer -> keep the finest seen this frame
    uint newWidth = fresh ? max(storedWidth, desiredWidth) : desiredWidth;
    uint desired  = (cyclicFrame << STREAMING_HITFRAME_FRAME_SHIFT) | newWidth;
    if (desired == old)
      return false;

    uint prev = atomicCompSwap(groupDemand.d[childIndex], old, desired);
    if (prev == old)
      return true;  // we advanced this child: enqueue it for further descent
    old = prev;      // lost the race, retry against the newer value
  }
}

////////////////////////////////////////////

// A queued group ("task") and the per-task state needed to relax one of its finer
// children. Held in registers, spread one-per-lane across the subgroup, and pulled
// to a processing lane via shuffle (like `TraversalInfo` in traversal_run).
struct DemandTask
{
  uint geometryID;
  uint fineBegin;   // first finer-overlap entry of this task's group
  uint invWidth;    // demand width to push one lod level finer
#if STREAMING_HITLOD_DEMAND_PRUNE_COARSE
  uint coarseChildren;  // low-detail seed: relax finer children but don't enqueue them
#endif
};

struct TaskInfo
{
  uint taskID;  // subgroup lane that owns the compacted task
};

// 2 subgroups per workgroup, each owns its own [subgroupOffset, +gl_SubgroupSize) slice
shared TaskInfo s_tasks[STREAM_HIT_DEMAND_TREE_WORKGROUP];

// Relax (and conditionally enqueue) a single finer child `taskSubID` of the task
// owned by subgroup lane `taskID`, mirroring `processSubTask`: the descriptor is
// shuffled in from the owning lane.
void processDemandChild(DemandTask subgroupTasks, uint taskID, uint taskSubID, bool isValid, uint cyclicFrame)
{
  uint geometryID     = subgroupShuffle(subgroupTasks.geometryID, taskID);
  uint fineBegin      = subgroupShuffle(subgroupTasks.fineBegin, taskID);
  uint invWidth       = subgroupShuffle(subgroupTasks.invWidth, taskID);
#if STREAMING_HITLOD_DEMAND_PRUNE_COARSE
  uint coarseChildren = subgroupShuffle(subgroupTasks.coarseChildren, taskID);
#endif

  // padding lanes of the last (partial) run took part in the shuffles above, but own
  // no real child; they must not touch the demand or the queue
  if (!isValid)
    return;

  Geometry geometry   = geometries[geometryID];
  uint     childIndex = decode20Bit(geometry.groupOverlapEntries, fineBegin + taskSubID);

  if (stream_hit_demand_relax(geometry.groupDemand, childIndex, invWidth, cyclicFrame))
  {
#if STREAMING_HITLOD_DEMAND_PRUNE_COARSE
    if (coarseChildren != 0)
      return;  // low-detail seed: stop the descent, don't enqueue
#endif
    uint64_t childEntry = uint64_t(childIndex) | (uint64_t(geometryID) << 32);
#if USE_PERSISTENT_HIT_DEMAND_KERNEL
    // producer side of the shared traversal producer/consumer queue: reserve a slot,
    // count the new in-flight task, then release-store it so a consumer racing ahead
    // sees the entry (and, by release/acquire, the task-counter bump) before consuming.
    // Only in-bounds stores are counted, so an overflow drops children instead of
    // leaving the task counter stuck above zero (which would hang termination).
    uint slot = atomicAdd(buildRW.traversalNodeWriteCounter, 1);
    if (slot < build.maxTraversalInfos)
    {
      atomicAdd(buildRW.traversalTaskCounter, 1);
      uint64s_coh demandQueue = uint64s_coh(uint64_t(build.traversalNodeInfos));
      atomicStore(demandQueue.d[slot], childEntry, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    }
#else
    uint64s_inout demandQueue = uint64s_inout(uint64_t(build.traversalNodeInfos));
    uint          slot        = atomicAdd(buildRW.hitDemandWriteCounter, 1);
    if (slot < build.maxTraversalInfos)
      demandQueue.d[slot] = childEntry;
#endif
  }
}

// Pool every task's finer children across the subgroup and process them in densely
// packed warp-wide runs. `threadSubCount == 0` (out-of-range thread, or a finest-lod
// group with no finer overlaps) contributes nothing. See `processAllSubTasks` in
// `traversal_run.comp.glsl` for the full derivation of the packing math.
void processAllDemandChildren(DemandTask threadTask, int threadSubCount, uint cyclicFrame)
{
  int endOffset    = subgroupInclusiveAdd(threadSubCount);
  int startOffset  = endOffset - threadSubCount;
  int totalThreads = subgroupShuffle(endOffset, SUBGROUP_SIZE - 1);
  int totalRuns    = (totalThreads + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  const uint subgroupOffset = gl_SubgroupID * gl_SubgroupSize;

  // compact the tasks that actually carry children. A runnable-but-childless task is
  // deliberately excluded here (and from the start bits below) so that the per-run
  // start-bit counting stays aligned with this compaction.
  bool  hasTask    = threadSubCount > 0;
  uvec4 taskVote   = subgroupBallot(hasTask);
  uint  taskOffset = subgroupBallotExclusiveBitCount(taskVote);

  if (hasTask) {
    s_tasks[subgroupOffset + taskOffset].taskID = gl_SubgroupInvocationID;
  }

  memoryBarrier(gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);

  // iterate the flat set of virtual worker threads: each locates which task it belongs
  // to (via the per-run start-bit ballot) and its child index within that task.
  int taskBase = -1;
  for (int r = 0; r < totalRuns; r++)
  {
    int tFirst   = r * SUBGROUP_SIZE;
    int t        = tFirst + int(gl_SubgroupInvocationID);
    int relStart = startOffset - tFirst;

#if SUBGROUP_SIZE > 32
    uvec2 startBits = subgroupOr(unpack32(hasTask && relStart >= 0 && relStart < SUBGROUP_SIZE ? (uint64_t(1) << relStart) : uint64_t(0)));
    int   task      = bitCount(startBits.x & gl_SubgroupLeMask.x) + bitCount(startBits.y & gl_SubgroupLeMask.y) + taskBase;
#else
    // set a bit where each task starts within this run
    uint  startBits = subgroupOr(hasTask && relStart >= 0 && relStart < SUBGROUP_SIZE ? (1u << relStart) : 0u);
    int   task      = bitCount(startBits & gl_SubgroupLeMask.x) + taskBase;
#endif

    uint taskID       = s_tasks[subgroupOffset + task].taskID;
    uint taskSubID    = t - subgroupShuffle(startOffset, taskID);
    uint taskSubCount = subgroupShuffle(threadSubCount, taskID);
    taskBase          = subgroupShuffle(task, SUBGROUP_SIZE - 1);  // carry into the next run

    bool taskValid    = taskSubID < taskSubCount;

    processDemandChild(threadTask, taskID, taskSubID, taskValid, cyclicFrame);
  }
}

////////////////////////////////////////////

// zero-initialised task for lanes that carry no queue entry (out of range, or the
// subgroup drained): threadSubCount stays 0 so they contribute nothing to the packing.
DemandTask emptyDemandTask()
{
  DemandTask task;
  task.geometryID     = 0;
  task.fineBegin      = 0;
  task.invWidth       = 0;
#if STREAMING_HITLOD_DEMAND_PRUNE_COARSE
  task.coarseChildren = 0;
#endif
  return task;
}

// Parse a queue entry into its task descriptor; returns the finer-child count.
int setupDemandTask(uint64_t entry, out DemandTask task)
{
  uint groupIndex = uint(entry);
#if STREAMING_HITLOD_DEMAND_PRUNE_COARSE
  // top bit tags a low-detail seed: relax its finer children's demand below, but do
  // not enqueue them, so the descent stops here instead of flooding deeper
  task.coarseChildren = (uint(entry >> 32) & STREAMING_HITLOD_DEMAND_COARSE_CHILDREN_BIT) != 0 ? 1u : 0u;
  task.geometryID     = uint(entry >> 32) & STREAMING_HITLOD_DEMAND_GEOMETRY_MASK;
#else
  task.geometryID     = uint(entry >> 32);
#endif

  Geometry geometry = geometries[task.geometryID];

  // this queued group was stamped current-frame; its stored width is the demand
  // to push one lod level finer
  task.invWidth = geometry.groupDemand.d[groupIndex] & STREAMING_HITFRAME_WIDTH_MASK;

  // the finer overlaps follow the countCoarse coarser entries in the packed run
  GroupOverlapInfo overlap = geometry.groupOverlapInfos.d[groupIndex];
  task.fineBegin = overlap.offset + uint(overlap.countCoarse);
  return int(overlap.countFine);
}

////////////////////////////////////////////

#if USE_PERSISTENT_HIT_DEMAND_KERNEL

// Persistent producer/consumer variant of the demand descent. A fixed pool of threads
// consumes the demand queue and produces the children it advances back into it, until
// the queue drains - no host round-trips between BFS levels. It reuses the traversal
// producer/consumer setup verbatim: the traversalNodeReadCounter/WriteCounter and the
// traversalTaskCounter over the ~0-cleared traversalNodeInfos queue. See `run_persistent`
// in `traversal_run.comp.glsl` for the core mechanism.
void run_persistent()
{
  uint cyclicFrame = streamingHitCyclicFrame();

  uint threadReadIndex = ~0u;
  while (true)
  {
    // once the whole subgroup ran dry, grab a fresh batch of read slots
    if (subgroupAll(threadReadIndex == ~0u))
    {
      if (subgroupElect())
        threadReadIndex = atomicAdd(buildRW.traversalNodeReadCounter, SUBGROUP_SIZE);
      threadReadIndex = subgroupBroadcastFirst(threadReadIndex) + gl_SubgroupInvocationID;
      threadReadIndex = threadReadIndex >= build.maxTraversalInfos ? ~0u : threadReadIndex;

      // all read offsets are past the queue end: nothing left for this subgroup
      if (subgroupAll(threadReadIndex == ~0u))
        break;
    }

    // reads race ahead of writes; spin until our slot is written, or the queue drains
    bool     threadRunnable = false;
    uint64_t entry          = ~uint64_t(0);
    while (true)
    {
      if (threadReadIndex != ~0u)
      {
        memoryBarrierBuffer();
        entry = atomicLoad(build.traversalNodeInfos.d[threadReadIndex],
                           gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
        // the slot still holds the ~0 clear until a producer release-stores into it
        threadRunnable = entry != ~uint64_t(0);
      }

      if (subgroupAny(threadRunnable))
        break;

      // the whole subgroup saw no valid work; if nothing is in flight we are done
      memoryBarrierBuffer();
      bool isEmpty = atomicLoad(buildRW.traversalTaskCounter,
                                gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) == 0;
      if (subgroupAny(isEmpty))
        return;
    }

    if (subgroupAny(threadRunnable))
    {
      DemandTask threadTask     = emptyDemandTask();
      int        threadSubCount = 0;
      if (threadRunnable)
        threadSubCount = setupDemandTask(entry, threadTask);

      // divergent per-task child counts: distribute them evenly across the subgroup.
      // Any children advanced here are produced back into the queue (see processDemandChild).
      processAllDemandChildren(threadTask, threadSubCount, cyclicFrame);

      // every consumed task leaves the in-flight set
      uint numRunnable = subgroupBallotBitCount(subgroupBallot(threadRunnable));
      if (subgroupElect())
        atomicAdd(buildRW.traversalTaskCounter, -int(numRunnable));

      if (threadRunnable)
        threadReadIndex = ~0u;  // consumed; grab a new slot next round
    }
  }
}

#else

// Multi-pass variant: one dispatch per BFS level, each processing the frontier range
// [hitDemandStart, hitDemandEnd) that the previous pass appended.
void main_multipass()
{
  uint threadIndex    = getGlobalInvocationIndex(gl_GlobalInvocationID) + build.hitDemandStart;
  bool threadRunnable = threadIndex < build.hitDemandEnd;

  uint cyclicFrame = streamingHitCyclicFrame();

  // out-of-range threads stay empty (threadSubCount == 0) but must still reach the
  // work distribution below, so the subgroup stays converged for its collectives.
  DemandTask threadTask     = emptyDemandTask();
  int        threadSubCount = 0;
  if (threadRunnable)
  {
    uint64s_inout demandQueue = uint64s_inout(uint64_t(build.traversalNodeInfos));
    threadSubCount = setupDemandTask(demandQueue.d[threadIndex], threadTask);
  }

  // divergent per-task child counts: distribute them evenly across the subgroup
  processAllDemandChildren(threadTask, threadSubCount, cyclicFrame);
}

#endif

void main()
{
#if USE_PERSISTENT_HIT_DEMAND_KERNEL
  run_persistent();
#else
  main_multipass();
#endif
}
