/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Only used for USE_BLAS_CACHING

  A thread represents one geometry.

  `cachedLevel` is recomputed every frame from the instances that ran, so it drops a
  geometry's retention the moment its instances stop being processed - which is exactly
  what the ptlas partition early-out does. The age filter would then unload groups that
  skipped instances still reference, leaving their cached blas dangling.

  This keeps a second value that is the minimum over a whole revalidation cycle instead
  of over one frame. A minimum over a window is at least as fine as the instantaneous
  one, so levels are held too long rather than released too early, and the error is
  memory rather than correctness.

  Two generations are needed, not one. `cachedRetainReset` fires on a global frame
  phase while a partition's forced revalidation fires on its own age, and the two drift
  apart as soon as partitions start being processed on different frames. A single value
  that takes this frame outright on the reset therefore collapses to invalid for every
  geometry whose partitions all happen to be skipped on that frame, and stays there
  until one of them is revalidated - up to `partitionRevalidateFrames`, far past
  `ageThreshold`. `retainNext` accumulates the running cycle and only becomes live once
  that cycle has completed, so the live value always covers at least one full cycle back
  and therefore at least one visit of every partition.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : require

#include "shaderio.h"

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(local_size_x = GEOMETRY_CACHED_RETAIN_WORKGROUP) in;

void main()
{
  uint geometryID = gl_GlobalInvocationID.x;
  if (geometryID >= build.numGeometries)
    return;

  uint now = build.geometryCachedInfos.d[geometryID].cachedLevel;

  if (build.frameIndex == 0)
  {
    // nothing is skipped on frame 0, seed both generations
    build.geometryCachedRetain.d[geometryID]     = now;
    build.geometryCachedRetainNext.d[geometryID] = now;
    return;
  }

  uint next = min(build.geometryCachedRetainNext.d[geometryID], now);

  if (build.cachedRetainReset != 0)
  {
    // the cycle that just ended goes live, the new one starts from this frame
    build.geometryCachedRetain.d[geometryID]     = next;
    build.geometryCachedRetainNext.d[geometryID] = now;
  }
  else
  {
    // within a cycle the live value may only get finer
    build.geometryCachedRetainNext.d[geometryID] = next;
    build.geometryCachedRetain.d[geometryID]     = min(build.geometryCachedRetain.d[geometryID], next);
  }
}
