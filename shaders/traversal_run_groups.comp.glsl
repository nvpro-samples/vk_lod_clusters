
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  This compute shader implements the traversal of cluster groups
  in the scene. Cluster groups iterate over their children
  and test the traversal metric of their generating groups
  in the opposite direction. Depending on the result
  it will then enqueue the clusters for rendering.
  
  `traversal_run.comp.glsl` is run before and outputs
    - `build.traversalGroupInfos` all traversed cluster groups that fulfill the metric.
    - `build.traversalGroupWriteCounter` number of the groups (may exceed recorded maximum).
    - `build.indirectDispatchGroups.gridX` the dimensions of this kernel's dispatch based on above
  
  The cluster groups fill the list of to be rendered
  clusters.
    - `build.renderClusterInfos` stores all clusters that are to be rendered as linear array
    - `build.renderClusterCounter` is used to append the clusters

  one thread represents one cluster group.
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
#extension GL_KHR_memory_scope_semantics : require

#include "shaderio.h"

////////////////////////////////////////////

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

#if TARGETS_RASTERIZATION && USE_TWO_PASS_CULLING
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];
#else
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
#endif

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

layout(local_size_x=TRAVERSAL_GROUPS_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////


void main()
{
#if USE_PERSISTENT_TRAVERSAL_KERNEL
  uint threadReadIndex = getGlobalInvocationIndex(gl_GlobalInvocationID);
  if (threadReadIndex >= min(build.traversalGroupWriteCounter, build.maxTraversalInfos)) return;
#else
  uint threadReadIndex = getGlobalInvocationIndex(gl_GlobalInvocationID) + build.traversalGroupStart;
  if (threadReadIndex >= build.traversalGroupEnd) return;
#endif
  
  // load group and test its clusters
  
  // pull required inputs
  TraversalInfo traversalInfo = unpackTraversalInfo(build.traversalGroupInfos.d[threadReadIndex]);
  uint instanceID             = traversalInfo.instanceID;
  // seeded at a discrete lod level, all clusters of this group are rendered
  bool forceTraverse          = unpackTraversalDiscrete(instanceID);
  uint groupIndex             = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
  uint groupClusterCount      = PACKED_GET(traversalInfo.packedNode, Node_packed_groupClusterCountMinusOne) + 1;

  uint geometryID   = instances[instanceID].geometryID;
  Geometry geometry = geometries[geometryID];

  // retrieve traversal & culling related information from the child node or cluster
  TraversalMetric traversalMetric;
#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
  BBox bbox;
#endif

  mat4x3 worldMatrix = transpose(instances[instanceID].worldMatrix);
  float uniformScale = computeUniformScale(worldMatrix);
  float errorScale   = 1.0;
#if USE_CULLING && TARGETS_RAY_TRACING
  uint visibilityState = build.instanceVisibility.d[instanceID];
  #if USE_CULLING && !USE_FORCED_INVISIBLE_CULLING
  // instance is not primary visible, apply different error scale
  if ((visibilityState & INSTANCE_VISIBLE_BIT) == 0) errorScale = build.culledErrorScale;
  #endif
#endif
  mat4x3 traversalMatrix = mat4x3(build.traversalViewMatrix * toMat4(worldMatrix));

#if USE_STREAMING
  // traversal_run ensured we never get here without ensuring residency
  // and we never traverse to a group that isn't resident.
  uint64_t groupAddress = geometry.streamingGroupAddresses.d[groupIndex];

#if TARGETS_RASTERIZATION && USE_TWO_PASS_REJECT_LISTS
  // Exception: the second pass seeds this kernel with group leaves that the first
  // pass rejected before reaching traversal_run's residency check, so do it here.
  if (groupAddress >= STREAMING_INVALID_ADDRESS_START)
  {
    uint64_t lastRequestFrameIndex = atomicMax(geometry.streamingGroupAddresses.d[groupIndex], streaming.request.frameIndex);
    bool triggerRequest = lastRequestFrameIndex != streaming.request.frameIndex;

    uvec4 voteRequested  = subgroupBallot(triggerRequest);
    uint  offsetRequested = 0;
    if (subgroupElect()) {
      offsetRequested = atomicAdd(streamingRW.request.loadCounter, subgroupBallotBitCount(voteRequested));
    }
    offsetRequested = subgroupBroadcastFirst(offsetRequested) + subgroupBallotExclusiveBitCount(voteRequested);

    if (triggerRequest && offsetRequested <= streaming.request.maxLoads) {
      streaming.request.loadGeometryGroups.d[offsetRequested] = uvec2(geometryID, groupIndex);
    }

    // cannot happen in the first pass, traversal_run checks residency before
    // enqueuing, but leaving a stale mask behind would make the reject kernel
    // dereference a group that is not resident
    if (build.cullPass == 0) {
      [[unroll]] for (uint w = 0; w < REJECT_CLUSTER_MASK_WORDS; w++) {
        build.rejectClusterMasks.d[threadReadIndex * REJECT_CLUSTER_MASK_WORDS + w] = 0;
      }
    }
    return;
  }
#endif

  Group_in groupRef = Group_in(groupAddress);
  Group group = groupRef.d;
  #if USE_BLAS_MERGING && TARGETS_RAY_TRACING
    // handled in traversal_run
  #else
    streaming.resident.groups.d[group.residentID].age = uint16_t(0);
  #endif
#else
  // can directly access the group
  Group_in groupRef = Group_in(geometry.preloadedGroups.d[groupIndex]);
  Group group = groupRef.d;
#endif

#if TARGETS_RASTERIZATION && USE_TWO_PASS_REJECT_LISTS
  // one reject bit per cluster, plus the object space union of all rejected
  // cluster bboxes for a single frustum test at the end
  uint rejectMask[REJECT_CLUSTER_MASK_WORDS];
  [[unroll]] for (uint w = 0; w < REJECT_CLUSTER_MASK_WORDS; w++) { rejectMask[w] = 0; }
  vec3 rejectLo = vec3(FLT_MAX);
  vec3 rejectHi = vec3(-FLT_MAX);
#endif

  for (uint clusterIndex = 0; clusterIndex < groupClusterCount; clusterIndex++)
  {
    bool forceCluster = false;
    bool isValid = true;

    {
    #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
      bbox        = Group_getClusterBBox(groupRef, clusterIndex);
    #endif
      
      // The continuous lod algorithm optimizes to get the lowest detail we can get away with.
      
      // We render a cluster if its own group was traversed because it had an error
      // greater than the threshold (it is "coarse enough"). This is fulfilled when reach
      // the code here.
      //
      // However, multiple cluster groups of previous lod levels (higher detail) may cover this 
      // same region. Therefore we must ensure that it's really this cluster to be drawn (it is "fine enough").
      //
      // This is achieved by looking at the cluster's generating group. The generating group
      // contained the geometry that this cluster was simplified from and is from the previous,
      // lower, lod level with a lower error.
      //
      // If that group wasn't traversed then we know we must be drawn, because we have the highest
      // detail required. You will see a bit later down that we use the negated results 
      // of `testForTraversal` for clusters.
      //
      // If this cluster is from the highest detail level, then there is no generating group
      // as encoded by `SHADERIO_ORIGINAL_MESH_GROUP`.
      // In streaming, it may also occur that the generating group isn't loaded, that also
      // means this cluster is the highest detail available.
      
      uint32_t clusterGeneratingGroup = forceTraverse ? SHADERIO_ORIGINAL_MESH_GROUP : Group_getGeneratingGroup(groupRef, clusterIndex);
    #if USE_STREAMING
      if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP
          && geometry.streamingGroupAddresses.d[clusterGeneratingGroup] < STREAMING_INVALID_ADDRESS_START)
      {
        // streaming must check if the other group actually is resident, if not then we always draw this group
        // as we know no other lod variant was loaded.
        traversalMetric = Group_in(geometry.streamingGroupAddresses.d[clusterGeneratingGroup]).d.traversalMetric;
      }
    #else
      if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP)
      {
        traversalMetric = Group_in(geometry.preloadedGroups.d[clusterGeneratingGroup]).d.traversalMetric;
      }
    #endif
      else {
        // the generating group doesn't exist, draw this group
        
        // this should always evaluate true
        traversalMetric = group.traversalMetric;
        forceCluster    = true;
      }
      // prepare to append this cluster for rendering, if metric evaluates properly
      
      // TraversalInfo aliases with ClusterInfo, packeNode == clusterID
      traversalInfo.packedNode = group.clusterResidentID + clusterIndex;
    }


    bool useAlpha = false;
    bool useSW = false;

  #if TARGETS_RASTERIZATION
    useAlpha = queryClusterUsesAlpha(instanceID, groupRef, group, clusterIndex);
  #endif

    // perform traversal & culling logic
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    bool isVisible     = queryClusterWasVisible(worldMatrix, bbox, useSW);
    isValid            = isValid && isVisible;
  #else
    const bool isVisible = true;
  #endif
    bool traverse      = testForTraversal(traversalMatrix, uniformScale, traversalMetric, errorScale);
    bool lodAccept     = !traverse || forceCluster;                  // clusters use negated test or are forced
    bool renderClusterAny = isValid && lodAccept;

  #if TARGETS_RASTERIZATION && USE_TWO_PASS_REJECT_LISTS
    // The lod decision is identical in both passes, only visibility differs, so a
    // cluster the metric wants but that failed here is all the second pass needs.
    if (build.cullPass == 0 && lodAccept && !isVisible)
    {
      rejectMask[clusterIndex >> 5] |= 1u << (clusterIndex & 31);
      rejectLo = min(rejectLo, bbox.lo);
      rejectHi = max(rejectHi, bbox.hi);
    }
  #endif

    // nodes will enqueue their children again (producer)
    // groups will write out the clusters for rendering
    
    // we use subgroup intrinsics to avoid doing per-thread
    // atomics to get the storage offsets

  #if TARGETS_RASTERIZATION
    rasterBinning(traversalInfo.packedNode, instanceID, useAlpha, useSW, renderClusterAny);
  #else
    // Ray tracing: single renderClusterInfos queue (no alpha / SW raster lists).
    bool renderCluster = renderClusterAny;

    uvec4 voteClusters = subgroupBallot(renderCluster);
    uint countClusters = subgroupBallotBitCount(voteClusters);

    uint offsetClusters = 0;
    if (subgroupElect())
    {
      offsetClusters = atomicAdd(buildRW.renderClusterCounter, countClusters);
    }

    offsetClusters = subgroupBroadcastFirst(offsetClusters);
    offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);

    renderCluster = renderCluster && offsetClusters < build.maxRenderClusters;

    if (renderCluster)
    {
      // For ray tracing count how many clusters we later add to each instance/blas.
      // this will help us determine the list length for each blas.
      // The `blas_setup_insertion.comp.glsl` kernel then sub-allocates space for the lists
      // based on this counter.
      atomicAdd(build.instanceBuildInfos.d[instanceID].clusterReferencesCount, 1);
      // the render list we write below is filled in an unsorted manner with clusters
      // from different instances. We later use the `blas_insert_clusters.comp.glsl` kernel to build
      // the list for each blas.

      // given TraversalInfo and ClusterInfo were chosen to alias in memory and be a single u64
      // we do just have to adjust the output addresses.
      uint writeIndex       = offsetClusters;
      uint64s_inout(build.renderClusterInfos).d[writeIndex] = packTraversalInfo(traversalInfo);
    }
  #endif
  }

#if TARGETS_RASTERIZATION && USE_TWO_PASS_REJECT_LISTS
  if (build.cullPass == 0)
  {
    bool anyReject = false;
    [[unroll]] for (uint w = 0; w < REJECT_CLUSTER_MASK_WORDS; w++) { anyReject = anyReject || rejectMask[w] != 0; }

    // if the union of all rejected clusters is outside the current frustum,
    // none of them can reappear in the second pass
    if (anyReject && !intersectFrustumOnly(build.cullViewProjMatrix, rejectLo, rejectHi, worldMatrix))
    {
      [[unroll]] for (uint w = 0; w < REJECT_CLUSTER_MASK_WORDS; w++) { rejectMask[w] = 0; }
    }

    // indexed by the group's slot in the first pass' `traversalGroupInfos`
    [[unroll]] for (uint w = 0; w < REJECT_CLUSTER_MASK_WORDS; w++) {
      build.rejectClusterMasks.d[threadReadIndex * REJECT_CLUSTER_MASK_WORDS + w] = rejectMask[w];
    }
  }
#endif
}
