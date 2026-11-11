/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  This compute shader writes the streaming request for
  groups to be unloaded. We determine this based on an
  age since the group has been used last.
  
  A thread represents one resident group.
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

#include "streaming.glsl"

////////////////////////////////////////////

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);

  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_AGEFILTER_CLUSTERS_WORKGROUP
  uint residentID = streaming.resident.activeGroups.d[threadID];
  if (threadID < streaming.resident.activeGroupsCount)
  {
    Group_in groupRef = streaming.resident.groups.d[residentID].group;
    uint geometryID   = streaming.resident.groups.d[residentID].geometryID;
    
    streamingAgeFilter(residentID, geometryID, groupRef, streaming.useBlasCaching != 0, streaming.useHitStreaming != 0);

#if STREAM_DEBUG_RESIDENCY
    uint age      = streaming.resident.groups.d[residentID].age;
    uint hitDelta = streamingHitFrameDelta(streaming.resident.groupHitFrames.d[residentID]);
    bool cached   = streaming.useBlasCaching != 0
                  && streaming.resident.groups.d[residentID].lodLevel >= build.geometryCachedRetain.d[geometryID];
    uint bin = cached ? 4 : age > 1 ? 3 : hitDelta <= 2 ? 0 : hitDelta <= 32 ? 1 : 2;
    atomicAdd(readback.debugA[bin], uint(groupRef.d.clusterCount));
    atomicAdd(readback.debugB[bin], 1);
#endif
  }
}

