/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Only used for TARGETS_RAY_TRACING and USE_BLAS_DEDUP

  This compute shader finds BLAS candidates that reference the same set of CLAS and
  builds only one of them. Equivalence is tested via the order-invariant key that
  `blas_dedup_keys.comp.glsl` accumulated.

  Candidates insert themselves into a hash table, those that find an equivalent entry
  record its representative, the others allocate a build and copy their descriptor into
  the compacted `blasBuildInfos` that the build operation reads.
  Not finding an equivalent is safe, we then just build a BLAS we could have re-used.

  A single thread represents one BLAS candidate, looping over all candidates.
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

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_RENDERMATERIALS_SSBO, set = 0) buffer renderMaterialsBuffer
{
  RenderMaterial materials[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;
};

////////////////////////////////////////////

layout(local_size_x=BLAS_DEDUP_SCAN_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  uint numDuplicates = 0;
  uint numMissed     = 0;
  uint numProbes     = 0;

  // the dispatch is sized for the typical candidate count, not for every possible blas
  uint threadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  for (uint candidateIndex = getGlobalInvocationIndex(gl_GlobalInvocationID); candidateIndex < build.blasCandidateCounter;
       candidateIndex += threadCount)
  {
    bool isDuplicate = false;
    bool isInserted  = false;

    BlasCandidateInfo candidate = build.blasCandidateInfos.d[candidateIndex];

    uint tableMask = build.blasDedupTableMask;
    uint bucket    = candidate.keyXor & tableMask;

    [[dont_unroll]]
    for (uint probe = 0; probe < BLAS_DEDUP_PROBE_COUNT; probe++)
    {
      numProbes++;

      // zero means empty, hence we store the index biased by one
      uint prev = atomicCompSwap(build.blasDedupTable.d[bucket], 0, candidateIndex + 1);

      if (prev == 0)
      {
        // we own this bucket and therefore are unique
        isInserted = true;
        break;
      }

      // compare the full key against the bucket's owner, which is always a unique
      // candidate whose key was written in the previous kernel
      uint              otherIndex = prev - 1;
      BlasCandidateInfo other      = build.blasCandidateInfos.d[otherIndex];

      if (other.keyXor == candidate.keyXor && other.keyAdd == candidate.keyAdd
          && other.clusterReferencesCount == candidate.clusterReferencesCount)
      {
        build.blasCandidateInfos.d[candidateIndex].representative = otherIndex;
        // only used by the "blas reuse" visualization
        atomicOr(build.blasCandidateInfos.d[otherIndex].flags, BLAS_CANDIDATE_HAS_DUPLICATES);
        isDuplicate = true;
        break;
      }

      // different content in this bucket, try the next
      bucket = (bucket + 1) & tableMask;
    }

    if (!isDuplicate)
    {
      // Unique, or we ran out of probes and conservatively build our own blas.
      uint buildIndex = atomicAdd(buildRW.blasBuildCounter, 1);

      build.blasBuildInfos.d[buildIndex].clusterReferencesCount  = candidate.clusterReferencesCount;
      build.blasBuildInfos.d[buildIndex].clusterReferencesStride = 8;
      build.blasBuildInfos.d[buildIndex].clusterReferences       = candidate.clusterReferences;

      build.blasCandidateInfos.d[candidateIndex].buildIndex = buildIndex;
    }

    numDuplicates += isDuplicate ? 1 : 0;
    numMissed += !isDuplicate && !isInserted ? 1 : 0;
  }

  // stats, aggregated per subgroup to keep the atomic traffic low.
  // `numBlasDedupProbes` over `numBlasCandidates` is the average probe length, and
  // `numBlasDedupMissed` counts candidates that neither found a free bucket nor an
  // equivalent. Both rise when the table is too small or the key distributes poorly.
  {
    uint duplicateSum = subgroupAdd(numDuplicates);
    uint missedSum    = subgroupAdd(numMissed);
    uint probeSum     = subgroupAdd(numProbes);

    if (subgroupElect())
    {
      if (duplicateSum != 0)
      {
        atomicAdd(readback.numBlasDeduped, duplicateSum);
      }
      if (missedSum != 0)
      {
        atomicAdd(readback.numBlasDedupMissed, missedSum);
      }
      if (probeSum != 0)
      {
        atomicAdd(readback.numBlasDedupProbes, probeSum);
      }
    }

    if (getGlobalInvocationIndex(gl_GlobalInvocationID) == 0)
    {
      readback.numBlasCandidates = build.blasCandidateCounter;
    }
  }
}
