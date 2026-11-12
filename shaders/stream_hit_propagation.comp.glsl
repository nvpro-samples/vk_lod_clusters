/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  This compute shader runs once per frame over all resident groups (including
  the persistent low-detail groups) for hit-based streaming:

  - coarser groups inherit the newest hit frame (and on ties the smallest hit
    cone width) from their resident finer (generating) groups, as rays only hit
    the currently rendered lod cut.
  - it records per group whether any finer group is resident, so the age filter
    only hit-unloads fine-to-coarse and the inheritance chain cannot break.
  - it re-normalizes stamps past the current cyclic frame, so hit registration's
    atomicMax ordering survives the 15-bit frame wrap.
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
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

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

layout(local_size_x=STREAM_AGEFILTER_GROUPS_WORKGROUP) in;

////////////////////////////////////////////

#ifndef USE_HIT_STREAMING
#define USE_HIT_STREAMING 1
#endif

#include "streaming.glsl"

////////////////////////////////////////////

void main()
{
  uint threadID    = getGlobalInvocationIndex(gl_GlobalInvocationID);
  uint activeCount = streaming.resident.activeGroupsCount;
  uint loCount     = streaming.resident.loGroupsCount;

  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_AGEFILTER_CLUSTERS_WORKGROUP
  uint residentID = threadID < loCount ? threadID : streaming.resident.activeGroups.d[threadID - loCount];
  if (threadID < loCount + activeCount)
  {
    uint lodLevel     = streaming.resident.groups.d[residentID].lodLevel;
    Group_in groupRef = streaming.resident.groups.d[residentID].group;
    uint geometryID   = streaming.resident.groups.d[residentID].geometryID;
    uint geoLastLevel = geometries[geometryID].lodLevelsCount - 1;
#if STREAMING_DEBUG_ADDRESSES
    if (uint64_t(groupRef) >= STREAMING_INVALID_ADDRESS_START)
    {
      streamingRW.request.errorHitPropagation   = 1;
      streamingRW.request.errorHitGeometryID = geometryID;
      streamingRW.request.errorHitGroupID    = streaming.resident.groupIDs.d[residentID];
      streamingRW.request.errorHitIndex      = residentID;
      return;
    }
#endif

    // Rays only hit the currently rendered lod cut, so walk the finer (generating)
    // groups: inherit their newest hit frame (on ties the smallest cone width) and
    // record whether any is resident in the hit frame's top bit (inverted, see
    // STREAMING_HITFRAME_NO_FINER_BIT). Finest-level groups have none.
    // Comparisons use a delta-based key (fresher frame, then smaller width) robust
    // against the frame wrap; the rewrite below re-normalizes the stamp.

    uint cyclicFrame = streamingHitCyclicFrame();

    // temporally smoothed demand width (see `groupHitWidths`): pulled finer
    // instantly by recent hits and finer groups, relaxed toward coarse by
    // STREAMING_HITLOD_WIDTH_RELAX per frame otherwise. Without it a single frame
    // of only-wide or missing hits would flip the lod demand coarse and flicker.
    uint invSmoothed = streaming.resident.groupHitWidths.d[residentID];
    invSmoothed      = invSmoothed > STREAMING_HITLOD_WIDTH_RELAX ? invSmoothed - STREAMING_HITLOD_WIDTH_RELAX : 0;

    // age of this group's own (or inherited) hit, for the demand level below
    uint hitDelta = STREAMING_HITFRAME_FRAME_MASK;

    if (lodLevel != 0)
    {
      uint noFiner  = STREAMING_HITFRAME_NO_FINER_BIT;
      uint hitValue = streaming.resident.groupHitFrames.d[residentID];
      uint bestKey  = ((STREAMING_HITFRAME_FRAME_MASK - streamingHitFrameDelta(hitValue)) << STREAMING_HITFRAME_FRAME_SHIFT)
                     | (hitValue & STREAMING_HITFRAME_WIDTH_MASK);

      if (streamingHitFrameDelta(hitValue) <= 1)
      {
        invSmoothed = max(invSmoothed, hitValue & STREAMING_HITFRAME_WIDTH_MASK);
      }

      // Walk the immediate finer (generating) groups from the precomputed fine
      // overlap list (built offline) instead of re-deriving them per cluster via
      // Group_getGeneratingGroup. The fine entries follow the countCoarse coarse
      // entries in the packed table.
      uint groupIndex          = streaming.resident.groupIDs.d[residentID];
      GroupOverlapInfo overlap = geometries[geometryID].groupOverlapInfos.d[groupIndex];
      uint fineBegin           = overlap.offset + uint(overlap.countCoarse);

      for (uint i = 0; i < uint(overlap.countFine); i++)
      {
        uint genGroupIndex = decode20Bit(geometries[geometryID].groupOverlapEntries, fineBegin + i);

#if STREAMING_DEBUG_ADDRESSES
        // the last lod level always has exactly one group, giving the total count
        //uint geoLastLevel   = geometries[geometryID].lodLevelsCount - 1;
        uint geoGroupsCount = geometries[geometryID].lodLevels.d[geoLastLevel].groupOffset + 1;
        if (genGroupIndex >= geoGroupsCount)
        {
          streamingRW.request.errorHitPropagation   = 2;
          streamingRW.request.errorHitGeometryID = geometryID;
          streamingRW.request.errorHitGroupID    = streaming.resident.groupIDs.d[residentID];
          streamingRW.request.errorHitIndex      = genGroupIndex;
          streamingRW.request.errorHitLimit      = geoGroupsCount;
          continue;
        }
#endif

        uint64_t groupAddress = geometries[geometryID].streamingGroupAddresses.d[genGroupIndex];
        if (groupAddress < STREAMING_INVALID_ADDRESS_START && groupAddress > 0)
        {
          noFiner = 0;

          Group_in groupHiRef = Group_in(groupAddress);
          uint hiResidentID   = groupHiRef.d.residentID;

#if STREAMING_DEBUG_ADDRESSES
          if (hiResidentID >= streaming.maxResidentGroups)
          {
            streamingRW.request.errorHitPropagation   = 3;
            streamingRW.request.errorHitGeometryID = geometryID;
            streamingRW.request.errorHitGroupID    = streaming.resident.groupIDs.d[residentID];
            streamingRW.request.errorHitIndex      = hiResidentID;
            streamingRW.request.errorHitLimit      = streaming.maxResidentGroups;
            continue;
          }
#endif

          uint hiValue = streaming.resident.groupHitFrames.d[hiResidentID];
          uint hiKey   = ((STREAMING_HITFRAME_FRAME_MASK - streamingHitFrameDelta(hiValue)) << STREAMING_HITFRAME_FRAME_SHIFT)
                       | (hiValue & STREAMING_HITFRAME_WIDTH_MASK);
          bestKey = max(bestKey, hiKey);

          // inherit the finer group's smoothed demand width (one frame old, benign)
          invSmoothed = max(invSmoothed, streaming.resident.groupHitWidths.d[hiResidentID]);
        }
      }

      uint bestDelta = STREAMING_HITFRAME_FRAME_MASK - (bestKey >> STREAMING_HITFRAME_FRAME_SHIFT);
      hitDelta       = bestDelta;
      // normalized stamp, carries the age over the wrap
      uint stamp = streamingHitStamp(bestDelta);

      streaming.resident.groupHitFrames.d[residentID] =
          noFiner | (stamp << STREAMING_HITFRAME_FRAME_SHIFT) | (bestKey & STREAMING_HITFRAME_WIDTH_MASK);

#if STREAM_PROPAGATION_MODE != STREAM_PROPAGATION_ALL_LEVELS
      // faster stream-in: seed the demand BFS from this resident hit frontier. If
      // the group was hit (or inherited a hit) within the demand window and has
      // finer overlaps, stamp its demand at the current frame and enqueue it so
      // stream_hit_demand_tree.comp.glsl descends the finer links from here. Seeding all
      // recently-hit resident groups is simplest and correct: interior seeds relax
      // their already-fresh neighbours and stop after one hop. The queue reuses the
      // now-free traversal frontier storage. (ALL_LEVELS self-seeds in its sweep.)
      if (bestDelta <= STREAMING_HITLOD_DEMAND_WINDOW && uint(overlap.countFine) != 0)
      {
        geometries[geometryID].groupDemand.d[groupIndex] =
            (cyclicFrame << STREAMING_HITFRAME_FRAME_SHIFT) | (invSmoothed & STREAMING_HITFRAME_WIDTH_MASK);

        uint slot = atomicAdd(buildRW.hitDemandWriteCounter, 1);
        if (slot < build.maxTraversalInfos)
        {
          uint64s_inout demandQueue = uint64s_inout(uint64_t(buildRW.traversalNodeInfos));
          uint          geometryTag = geometryID;
#if STREAMING_HITLOD_DEMAND_PRUNE_COARSE
          // low-detail seed (within PRUNE_COARSE levels of the coarsest): tag it so the
          // descent relaxes its immediate finer children but does not descend deeper
          if (lodLevel >= (geoLastLevel - STREAMING_HITLOD_DEMAND_PRUNE_COARSE + 1))
            geometryTag |= STREAMING_HITLOD_DEMAND_COARSE_CHILDREN_BIT;
#endif
          demandQueue.d[slot] = uint64_t(groupIndex) | (uint64_t(geometryTag) << 32);
        }
      }
#endif
    }
    else
    {
      // finest level groups receive no inheritance, only fold their own hits
      // into the smoothed demand width, and re-normalize the stamp after a
      // frame wrap so hit registration's atomicMax keeps working
      uint hitValue = streaming.resident.groupHitFrames.d[residentID];
      hitDelta      = streamingHitFrameDelta(hitValue);
      if (hitDelta <= 1)
      {
        invSmoothed = max(invSmoothed, hitValue & STREAMING_HITFRAME_WIDTH_MASK);
      }

      uint rawStamp = (hitValue >> STREAMING_HITFRAME_FRAME_SHIFT) & STREAMING_HITFRAME_FRAME_MASK;
      uint stamp    = streamingHitStamp(hitDelta);
      if (stamp != rawStamp)
      {
        streaming.resident.groupHitFrames.d[residentID] =
            (hitValue & (STREAMING_HITFRAME_NO_FINER_BIT | STREAMING_HITFRAME_WIDTH_MASK))
            | (stamp << STREAMING_HITFRAME_FRAME_SHIFT);
      }
    }

    streaming.resident.groupHitWidths.d[residentID] = invSmoothed;

    if (streaming.useHitStreamingLod != 0)
    {
      // The finest level a traversal of this geometry can open next frame, which classify
      // raises `lodLevelMin` to: an instance on a per-frame blas gets nothing finer than
      // the rays demand, see `streamingDemandsFiner`. The demand passes after this one
      // refresh resident groups from their own hits, so those count as well. The demand
      // floods finer while the width stays below a group's error, bounded by the error of
      // each level's root node.
      uint groupIndex = streaming.resident.groupIDs.d[residentID];
      uint demand     = geometries[geometryID].groupDemand.d[groupIndex];
      uint invWidth   = streamingHitFrameDelta(demand) <= STREAMING_HITLOD_DEMAND_WINDOW ? demand & STREAMING_HITFRAME_WIDTH_MASK : 0;
      if (hitDelta <= STREAMING_HITLOD_DEMAND_WINDOW)
        invWidth = max(invWidth, invSmoothed & STREAMING_HITFRAME_WIDTH_MASK);

      float width    = streamingDecodeHitWidth(invWidth) / streaming.hitLodWidthFactor;
      bool  coarsest = geometries[geometryID].groupOverlapInfos.d[groupIndex].countCoarse == 0;
      if (coarsest || width < groupRef.d.traversalMetric.maxQuadricError)
      {
        uint childOffset = LOD_LEVEL_NODE_OFFSET;
        uint level       = lodLevel;
        while (level > 0 && width < geometries[geometryID].nodes.d[childOffset + level - 1].traversalMetric.maxQuadricError)
        {
          level--;
        }
        if (level < build.geometryHitDemandLevel.d[geometryID])
        {
          atomicMin(build.geometryHitDemandLevel.d[geometryID], level);
        }
      }
    }

#if STREAM_DEBUG_HIT_STATS
    // per lod-level statistics (post-inheritance hit frame; readback debug arrays
    // are zero-filled at frame start). bins are relative to each geometry's last
    // (coarsest) lod level, so bin 0 = coarsest.
    uint geoLodLevelsCount = geometries[geometryID].lodLevelsCount;
    uint statBin           = geoLodLevelsCount - 1 - lodLevel;
    atomicAdd(readback.debugB[statBin], 1);
    if (streamingHitFrameDelta(streaming.resident.groupHitFrames.d[residentID]) < 32)
    {
      atomicAdd(readback.debugA[statBin], 1);
    }
    // there is exactly one persistent lo group per geometry, so these threads
    // also accumulate their geometry's total group counts, independent of residency
    if (threadID < loCount)
    {
      for (uint l = 0; l < geoLodLevelsCount; l++)
      {
        atomicAdd(readback.debugC[geoLodLevelsCount - 1 - l], geometries[geometryID].lodLevels.d[l].groupCount);
      }
    }
#endif
  }
}

