/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Only used for TARGETS_RAY_TRACING
  
  This compute shader assigns the blasReference address to each
  tlas instance description prior updating the tlas and after
  the per-frame blas were built.

  A single thread represents one instance
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

#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};
#endif

////////////////////////////////////////////

layout(local_size_x=INSTANCES_ASSIGN_BLAS_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
#if USE_PARTITION_EARLY_OUT
  // dispatch is over the compacted blocks of partition-sorted instances, a block id is a
  // start offset so the mapping is arithmetic
  uint workGroupID  = getWorkGroupIndex(gl_WorkGroupID);
  uint blockSlot    = workGroupID / (PARTITION_INSTANCE_BLOCK_SIZE / PARTITION_ACTIVE_INSTANCES_WORKGROUP);
  uint blockLocal   = (workGroupID % (PARTITION_INSTANCE_BLOCK_SIZE / PARTITION_ACTIVE_INSTANCES_WORKGROUP))
                          * PARTITION_ACTIVE_INSTANCES_WORKGROUP
                      + gl_LocalInvocationID.x;
  // only a 16-bit grid overshoots the active blocks, past them the slots are stale
  bool blockValid   = USE_16BIT_DISPATCH == 0 || blockSlot < build.activeBlockCounter;
  uint orderedIndex = build.activeBlocks.d[blockValid ? blockSlot : 0] * PARTITION_INSTANCE_BLOCK_SIZE + blockLocal;
  // clamp the index, carry validity separately: the guarded loads below are predicated
  // rather than branched, and a ~0 id would land at base + 4 GiB (32-bit byte offsets)
  blockValid        = blockValid && orderedIndex < build.numRenderInstances;
  uint instanceID   = build.partitionInstanceOrder.d[blockValid ? orderedIndex : 0];
#else
  uint instanceID   = getGlobalInvocationIndex(gl_GlobalInvocationID);
  bool blockValid   = true;
#endif
  
  if (blockValid && instanceID < build.numRenderInstances)
  {
    InstanceBuildInfo buildInfo = build.instanceBuildInfos.d[instanceID];
    uint buildIndex             = buildInfo.blasBuildIndex;
    
    bool doStats         = true;
    uint64_t blasAddress = 0;
    // a blas that is built every frame changes its content even when it happens to
    // land on the same address again
    bool isPerFrameBlas  = false;
    bool cachedRebuilt   = false;

  #if USE_BLAS_SHARING
    // we might reference another instance's blas, the two bits are mutually exclusive.
    // Resolved first: with sticky levels the shared instance can be on the cached blas.
    if (buildIndex != BLAS_BUILD_INDEX_LOWDETAIL && (buildIndex & BLAS_BUILD_INDEX_SHARE_BIT) != 0)
    {
      buildIndex = build.instanceBuildInfos.d[buildIndex & ~BLAS_BUILD_INDEX_SHARE_BIT].blasBuildIndex;
      // don't add to build stats if we are referencing another instance
      doStats    = false;
    }
  #endif
  #if USE_BLAS_CACHING
    // we might reference the geometry's cached blas
    if (buildIndex != BLAS_BUILD_INDEX_LOWDETAIL && (buildIndex & BLAS_BUILD_INDEX_CACHE_BIT) != 0)
    {
      blasAddress = streaming.cachedBlasAddresses.d[
          (buildIndex & BLAS_BUILD_INDEX_GEOMETRY_MASK) * GEOMETRY_CACHED_BLAS_STRIDE
          + ((buildIndex & BLAS_BUILD_INDEX_CACHE_SLOT_MASK) >> BLAS_BUILD_INDEX_CACHE_SLOT_SHIFT)];
      // don't add to build stats, the cached blas has its own persistent storage
      doStats     = false;
    #if USE_PTLAS
      // rebuilt this frame, possibly where the blas it replaced was
      cachedRebuilt = geometries[buildIndex & BLAS_BUILD_INDEX_GEOMETRY_MASK].cachedBuildFrame == streaming.frameIndex;
    #endif

      if (blasAddress == 0)
      {
        // streaming can invalidate the entry after the routing decision. Fall back, since
        // `buildIndex` still carries CACHE_BIT and would index `blasBuildAddresses`
        // with 1<<30 plus the geometry id.
        buildIndex = BLAS_BUILD_INDEX_LOWDETAIL;
      }
    }
  #endif

  #if USE_BLAS_DEDUP
    // `blasBuildIndex` references a blas candidate, a deduplicated one points at
    // its representative. Like sharing it must not add to the memory stats twice.
    if (buildIndex != BLAS_BUILD_INDEX_LOWDETAIL && blasAddress == 0)
    {
      uint representative = build.blasCandidateInfos.d[buildIndex].representative;
      doStats             = doStats && (representative == buildIndex);
      buildIndex          = build.blasCandidateInfos.d[representative].buildIndex;
    }
  #endif

    // By default tlasInstances are set to low detail blas,
    // override when applicable.

    if (buildIndex != BLAS_BUILD_INDEX_LOWDETAIL)
    {
    #if USE_MEMORY_STATS
      if (doStats)
      {
        atomicAdd(readback.blasActualSizes, uint64_t(build.blasBuildSizes.d[buildIndex]));
      }
    #endif
      
    #if USE_BLAS_CACHING
      isPerFrameBlas = blasAddress == 0;
    #else
      isPerFrameBlas = true;
    #endif
      
      blasAddress =
      #if USE_BLAS_CACHING
        blasAddress != 0 ? blasAddress : 
      #endif
        build.blasBuildAddresses.d[buildIndex];
    }
  #if !USE_PTLAS
    if (blasAddress != 0)
    {
      build.tlasInstances.d[instanceID].blasReference = blasAddress;
    }
  #else
    // updates carry the blas explicitly, resolve the fallback here
    if (blasAddress == 0)
    {
      blasAddress = geometries[instances[instanceID].geometryID].lowDetailBlasAddress;
    }
  #endif

  #if USE_PARTITION_EARLY_OUT
    if (isPerFrameBlas)
    {
      // marked on the assignment, not on the routing intent in `traversal_init_blas_reuse`:
      // `blas_setup_insertion` can still hand a routed instance a per-frame build slot
      atomicAdd(build.partitionNonPersistent.d[build.instancePartitions.d[instanceID]], 1);
    }
  #endif
    
  #if USE_PTLAS
    // later builds only touch the instances whose blas changed. Persistent blas
    // (cached or pre-built low detail) keep their content, so comparing the address
    // is enough for those, unless rebuilt.
    if (isPerFrameBlas || cachedRebuilt || build.ptlasInstanceBlas.d[instanceID] != blasAddress)
    {
      build.ptlasInstanceBlas.d[instanceID] = blasAddress;
      
      PtlasInstanceUpdate instanceUpdate;
      instanceUpdate.instanceIndex     = instanceID;
      instanceUpdate.instanceSbtOffset = 0;
      instanceUpdate.blasReference     = blasAddress;
      
      build.ptlasInstanceUpdates.d[atomicAdd(buildRW.ptlasUpdateCounter, 1)] = instanceUpdate;
      
      // statistics on how the updates spread across the partitions
      uint partitionIndex = build.instancePartitions.d[instanceID];
      if (atomicAdd(build.ptlasPartitionUpdates.d[partitionIndex], 1) == 0)
      {
        atomicAdd(readback.numPtlasDirtyPartitions, 1);
        atomicAdd(readback.numPtlasDirtyInstances, build.partitionRanges.d[partitionIndex].y);
      }
    }
  #endif
  }
  
#if 1
  // stats, from the first thread: instance 0 is not processed when the early-out skips its partition
  if (getGlobalInvocationIndex(gl_GlobalInvocationID) == 0)
  {
    readback.numBlasBuilds = build.blasBuildCounter;
  }
#endif
}