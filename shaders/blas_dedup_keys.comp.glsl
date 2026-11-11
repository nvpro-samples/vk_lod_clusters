/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Only used for TARGETS_RAY_TRACING and USE_BLAS_DEDUP

  This compute shader accumulates the order-invariant key of every BLAS candidate from
  its completed CLAS reference list, which `blas_dedup_scan.comp.glsl` then compares.
  Folding the hashes in during `blas_clusters_insert.comp.glsl` would cost two more
  atomics per CLAS. Here the lists are packed, so a subgroup mostly covers one candidate
  and issues a single atomic per key component.

  A single thread represents one CLAS reference, looping over all of them.
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
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

////////////////////////////////////////////

layout(local_size_x=BLAS_DEDUP_SCAN_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint threadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

  // the per-blas lists are packed, `blasClasCounter` covers all of them
  for(uint slot = gl_GlobalInvocationID.x; slot < build.blasClasCounter; slot += threadCount)
  {
    uint     candidateIndex = build.blasCandidateSlots.d[slot];
    // a CLAS address is unique per resident cluster within the frame
    uint64_t address = build.blasClusterAddresses.d[slot];
    uint     hash    = murmurHash3Finalize(murmurHash3Combine(murmurHash3Combine(0, uint(address)), uint(address >> 32)), 8);

    if(subgroupAllEqual(candidateIndex))
    {
      uint keyXor = subgroupXor(hash);
      uint keyAdd = subgroupAdd(hash);
      if(subgroupElect())
      {
        atomicXor(build.blasCandidateInfos.d[candidateIndex].keyXor, keyXor);
        atomicAdd(build.blasCandidateInfos.d[candidateIndex].keyAdd, keyAdd);
      }
    }
    else
    {
      atomicXor(build.blasCandidateInfos.d[candidateIndex].keyXor, hash);
      atomicAdd(build.blasCandidateInfos.d[candidateIndex].keyAdd, hash);
    }
  }
}
