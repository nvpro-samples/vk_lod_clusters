/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Only used for USE_PARTITION_EARLY_OUT

  A thread represents one partition.

  A partition is `stable` when every instance in it resolved to a persistent blas
  last frame (cached or pre-built low detail). Such a partition has no traversal,
  no blas build and no ptlas update to do, so the per-instance kernels can skip
  its whole range.

  Skipping defers quality, not correctness: an instance that moved closer keeps
  its coarser persistent blas until the partition is processed again, which
  `partitionRevalidateFrames` bounds. Partitions crossing or entering the frustum within
  `partitionFrustumDepth` are processed every frame, their culled instances were
  classified at `culledErrorScale` and must not wait. In hybrid every partition in this
  frame's frustum is processed, its instances are rasterized. Streaming moving a cached blas is a
  different matter and must set `partitionInvalidateAll`.

  Partitions that are not skipped mark their instance blocks in `partitionBlockBits`,
  `partition_block_compact` turns them into the active blocks the per-instance kernels
  dispatch over.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : require
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

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;
};

#define CULLING_NO_HIZ
#include "culling.glsl"

layout(local_size_x = PARTITION_SETUP_WORKGROUP) in;

void main()
{
  uint partitionIndex = gl_GlobalInvocationID.x;
  if (partitionIndex >= build.numPartitions)
    return;

  uvec2 range = build.partitionRanges.d[partitionIndex];
  if (range.y == 0)
    return;

  uint nonPersistent = build.partitionNonPersistent.d[partitionIndex];
  uint age           = build.frameIndex - build.partitionLastFrame.d[partitionIndex];

  // same frustum matrix as `instance_classify_lod`
  BBox bbox     = build.partitionBboxes.d[partitionIndex];
  uint  andBits = ~0u;
  uint  orBits  = 0;
  float minW    = 3.402823466e+38;
  uint andBitsRaster = ~0u;
  for (int n = 0; n < 8; n++)
  {
    vec4 corner = getBoxCorner(bbox.lo, bbox.hi, n);
    vec4 hPos   = build.cullViewProjMatrixLast * corner;
    uint bits   = getCullBits(hPos);
    andBits &= bits;
    orBits  |= bits;
    minW     = min(minW, hPos.w);
  #if USE_HYBRIDPATH
    andBitsRaster &= getCullBits(build.cullViewProjMatrix * corner);
  #endif
  }
  bool visible  = andBits == 0;
  // hybrid rasterizes this frame's traversal of every instance in this frame's frustum,
  // see `getHybridVisibility`. The host turns the early-out off for hybrid without culling.
  bool rasterVisible = USE_HYBRIDPATH != 0 && andBitsRaster == 0;
  // distant partitions stay coarse, so they need not be revalidated. w is the view depth,
  // negative behind the camera.
  bool nearField = minW < build.partitionFrustumDepth;
  // straddles a plane; the flag also catches a camera cut that moves a partition
  // from outside to fully inside in one frame
  bool crossing = build.partitionFrustumRevalidate != 0 && visible && nearField && orBits != 0;
  bool entered  = build.partitionFrustumRevalidate != 0 && visible && nearField && build.partitionVisible.d[partitionIndex] == 0;
  build.partitionVisible.d[partitionIndex] = visible ? 1 : 0;

  bool stable = nonPersistent == 0 && build.partitionInvalidateAll == 0 && !crossing && !entered && !rasterVisible;
  bool aged   = build.partitionRevalidateFrames != 0 && age >= build.partitionRevalidateFrames;

  if (stable && !aged)
    return;

  // only the partitions we process re-accumulate, so only they reset
  build.partitionNonPersistent.d[partitionIndex] = 0;
  build.partitionLastFrame.d[partitionIndex]     = build.frameIndex;
  atomicAdd(buildRW.partitionActiveCounter, 1);

  // Mark the blocks of the partition-sorted instance array this partition covers.
  // Compacting blocks rather than instances keeps the per-frame work proportional to blocks
  // (thousands), at the cost of a block dragging a stable partition it shares along.
  uint blockFirst = range.x / PARTITION_INSTANCE_BLOCK_SIZE;
  uint blockLast  = (range.x + range.y - 1) / PARTITION_INSTANCE_BLOCK_SIZE;

  for (uint block = blockFirst; block <= blockLast; block++)
  {
    atomicOr(build.partitionBlockBits.d[block >> 5], 1u << (block & 31));
  }
}
