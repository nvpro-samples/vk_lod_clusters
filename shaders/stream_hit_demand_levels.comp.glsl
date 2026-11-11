/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Full-sweep stream-in demand propagation for hit-based streaming (the ALL_LEVELS
  mode). Dispatched once per reverse lod level (coarsest first), so every group is
  visited exactly once per frame.

  A pass for reverse level k covers a flat per-level index space; each thread
  binary-searches its geometry, maps to (geometry, group), and writes that group's
  persistent demand `Geometry::groupDemand` from:
   - its own resident hit, if hit (or inherited) within STREAMING_HITLOD_DEMAND_WINDOW.
   - one hop from its coarser overlaps' demand. Those are one reverse level up,
     written in the preceding pass (barrier between), so the demand still reaches
     every level in one frame despite each group being touched once.

  A thread represents one group at this reverse level.
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

layout(push_constant) uniform pushData
{
  uint reverseLevel;                // which reverse lod level this pass processes (0 == coarsest)
  uint levelTotalGroups;            // total groups at this reverse level across active geometries
  uint32s_in geometryLevelOffsets;  // compact level-major flat prefix table [reverseLevel*numGeometries + g]
} push;

////////////////////////////////////////////

layout(local_size_x=STREAM_AGEFILTER_GROUPS_WORKGROUP) in;

////////////////////////////////////////////

#ifndef USE_HIT_STREAMING
#define USE_HIT_STREAMING 1
#endif
#ifndef USE_HIT_STREAMING_LOD
#define USE_HIT_STREAMING_LOD 1
#endif

#include "streaming.glsl"

////////////////////////////////////////////

void main()
{
  // this pass processes reverse lod level `push.reverseLevel` (0 == coarsest)
  // across all geometries
  uint levelIndex = getGlobalInvocationIndex(gl_GlobalInvocationID);
  if (levelIndex >= push.levelTotalGroups)
    return;

  // map the flat per-level index back to (geometry, local group): binary search for
  // the largest geometry whose flat offset <= levelIndex (geometries lacking this
  // level contribute a zero-width range, so the search lands on the owner)
  uint base = push.reverseLevel * build.numGeometries;
  uint lo = 0;
  uint hi = build.numGeometries;
  while (lo + 1 < hi)
  {
    uint mid = (lo + hi) >> 1;
    if (push.geometryLevelOffsets.d[base + mid] <= levelIndex)
      lo = mid;
    else
      hi = mid;
  }

  Geometry geometry = geometries[lo];

  // defensive: a valid levelIndex always lands on a geometry that has this level.
  if (push.reverseLevel >= uint(geometry.lodLevelsCount))
    return;

  // forward lod level for this reverse level, then the group's index within it
  uint lodLevel   = uint(geometry.lodLevelsCount) - 1 - push.reverseLevel;
  uint groupIndex = geometry.lodLevels.d[lodLevel].groupOffset + (levelIndex - push.geometryLevelOffsets.d[base + lo]);

  uint cyclicFrame = streamingHitCyclicFrame();

  uint bestWidth = 0;  // inverted width, 0 == no demand (largest width)
  bool anyRecent = false;

  // seed from the group's own resident hit. The rendered cut and the coarser
  // levels carry the finest width their rays (or inherited finer hits) demand.
  uint64_t groupAddress = geometry.streamingGroupAddresses.d[groupIndex];
  if (groupAddress < STREAMING_INVALID_ADDRESS_START)
  {
    uint residentID = Group_in(groupAddress).d.residentID;
    if (streamingHitFrameDelta(streaming.resident.groupHitFrames.d[residentID]) <= STREAMING_HITLOD_DEMAND_WINDOW)
    {
      bestWidth = max(bestWidth, streaming.resident.groupHitWidths.d[residentID]);
      anyRecent = true;
    }
  }

  // gather one hop from the coarser overlaps' demand (the first countCoarse
  // entries of this group's overlap run), written in the preceding pass. A recent
  // coarser demand advances one level finer here and is re-stamped to this frame.
  GroupOverlapInfo overlap = geometry.groupOverlapInfos.d[groupIndex];
  for (uint i = 0; i < uint(overlap.countCoarse); i++)
  {
    uint overlapIndex = decode20Bit(geometry.groupOverlapEntries, overlap.offset + i);

    uint demand = geometry.groupDemand.d[overlapIndex];
    if (streamingHitFrameDelta(demand) <= 1)
    {
      bestWidth = max(bestWidth, demand & STREAMING_HITFRAME_WIDTH_MASK);
      anyRecent = true;
    }
  }

  geometry.groupDemand.d[groupIndex] =
      anyRecent ? ((cyclicFrame << STREAMING_HITFRAME_FRAME_SHIFT) | (bestWidth & STREAMING_HITFRAME_WIDTH_MASK)) : 0;
}
