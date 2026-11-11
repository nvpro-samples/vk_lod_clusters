/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Only used for TARGETS_RAY_TRACING && USE_HYBRIDPATH

  Second raster pass of the hybrid path. The traversal appended the clusters that
  last frame's hiz occluded to `hybridRejectClusterInfos`. Those passing the hiz of
  the first pass are appended to `hybridClusterInfos`, which `BUILD_SETUP_HYBRID_DRAW`
  turns into the second pass' draw. They are part of the lod cut, so the blas matches.

  Compiled with USE_TWO_PASS_CULLING, `texHizFar[1]` is the first pass' hiz.

  A thread represents one cluster.
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

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];

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
#endif

////////////////////////////////////////////

layout(local_size_x=HYBRIDPATH_CULL_WORKGROUP) in;

#include "culling.glsl"

////////////////////////////////////////////

void main()
{
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);
  uint loadID   = min(threadID, build.numHybridRejectClusters - 1);
  bool isValid  = threadID < build.numHybridRejectClusters;

  ClusterInfo    cinfo    = build.hybridRejectClusterInfos.d[loadID];
  RenderInstance instance = instances[cinfo.instanceID];
  Geometry       geometry = geometries[instance.geometryID];

#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[cinfo.clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[cinfo.clusterID]);
#endif
  BBox bbox = Cluster_getBBox(clusterRef);

  // the traversal already tested this frame's frustum, only occlusion is left.
  // No size test, a small cluster can still cover a pixel.
  vec4 clipMin;
  vec4 clipMax;
  vec2 clipExtent;
  bool clipValid;
  intersectFrustum(build.cullViewProjMatrix, bbox.lo, bbox.hi, transpose(instance.worldMatrix), clipMin, clipMax, clipValid, clipExtent);
  bool isVisible = isValid && (!clipValid || intersectHiz(clipMin, clipMax, 1));

#if USE_HYBRID_SW_RASTER
  bool useSW = hybridClusterUsesSW(clipMin, clipMax, clipExtent, clipValid, bbox, transpose(instance.worldMatrix), cinfo.instanceID, uint(clusterRef.d.stateBits));
#else
  const bool useSW = false;
#endif
  bool appendHW = isVisible && !useSW;

  uvec4 voteClusters  = subgroupBallot(appendHW);
  uint  offsetCluster = 0;
#if USE_HYBRID_SW_RASTER
  bool  appendSW = isVisible && useSW;
  uvec4 voteSW   = subgroupBallot(appendSW);
  uint  offsetSW = 0;
#endif
  if (subgroupElect())
  {
    offsetCluster = atomicAdd(buildRW.hybridClusterCounter, subgroupBallotBitCount(voteClusters));
#if USE_HYBRID_SW_RASTER
    offsetSW = atomicAdd(buildRW.renderClusterCounterSW, subgroupBallotBitCount(voteSW));
#endif
  }
  offsetCluster = subgroupBroadcastFirst(offsetCluster) + subgroupBallotExclusiveBitCount(voteClusters);

  if (appendHW && offsetCluster < build.maxRenderClusters)
  {
    build.hybridClusterInfos.d[offsetCluster] = cinfo;
  }
#if USE_HYBRID_SW_RASTER
  offsetSW = subgroupBroadcastFirst(offsetSW) + subgroupBallotExclusiveBitCount(voteSW);
  if (appendSW && offsetSW < build.maxRenderClusters)
  {
    build.renderClusterInfosSW.d[offsetSW] = cinfo;
  }
#endif
}
