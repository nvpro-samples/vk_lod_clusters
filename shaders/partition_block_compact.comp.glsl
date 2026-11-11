/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Only used for USE_PARTITION_EARLY_OUT

  A thread represents one block of `PARTITION_INSTANCE_BLOCK_SIZE` partition-sorted
  instances. `partition_setup.comp.glsl` marked the blocks covered by the
  partitions that must be processed this frame; this compacts those block ids so the
  per-instance kernels can dispatch over them.

  A block id is a start offset: block b covers `partitionInstanceOrder` entries
  [b * PARTITION_INSTANCE_BLOCK_SIZE, ...), so no search is needed to consume it.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;
};

layout(local_size_x = PARTITION_BLOCK_COMPACT_WORKGROUP) in;

void main()
{
  uint block   = gl_GlobalInvocationID.x;
  bool isValid = block < build.numInstanceBlocks;
  bool isSet   = isValid && (build.partitionBlockBits.d[block >> 5] & (1u << (block & 31))) != 0;

  uvec4 vote   = subgroupBallot(isSet);
  uint  count  = subgroupBallotBitCount(vote);
  uint  offset = 0;

  if (subgroupElect() && count != 0)
  {
    offset = atomicAdd(buildRW.activeBlockCounter, count);
  }
  offset = subgroupBroadcastFirst(offset) + subgroupBallotExclusiveBitCount(vote);

  if (isSet)
  {
    build.activeBlocks.d[offset] = block;
  }
}
