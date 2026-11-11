/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Only used for USE_HIT_STREAMING_LOD

  A thread represents one instance.

  Filters last frame's `instanceHitDemand` into the persistent `instanceHitScale`
  that classify and traversal read, and resets the raw value for this frame's rays.

  The raw value is a per-frame minimum over ray hits and sits just below 1.0 for
  plain camera rays, so it is quantized to a log2 grid, roughly one lod level per
  step, and only a demand of 0.5 or finer counts. Finer applies at once, coarser one
  step at a time after STREAMING_HITLOD_INSTANCE_HOLD_FRAMES without demand.

  A changed scale wakes the instance's partition, the early-out would otherwise
  not see it before the partition's next revalidation.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(local_size_x = INSTANCE_HIT_SCALE_WORKGROUP) in;

void main()
{
  uint instanceID = getGlobalInvocationIndex(gl_GlobalInvocationID);
  if (instanceID >= build.numRenderInstances)
    return;

  // floatBitsToUint(FLT_MAX), no demand. Only the few instances rays demanded need the reset.
  uint  demandBits = build.instanceHitDemand.d[instanceID];
  float demand     = uintBitsToFloat(demandBits);
  if (demandBits != 0x7F7FFFFFu)
    build.instanceHitDemand.d[instanceID] = 0x7F7FFFFFu;

  // floor rounds toward coarser, so everything above 0.5 is level 0
  const uint levelMax = uint(-log2(STREAMING_HITLOD_INSTANCE_SCALE_MIN) + 0.5);
  uint       level    = demand <= 0.5 ? min(uint(floor(-log2(demand))), levelMax) : 0;

  uint state   = build.instanceHitScale.d[instanceID];
  uint applied = state & 0xFF;
  uint hold    = state >> 8;

  if (level >= applied)
  {
    applied = level;
    hold    = 0;
  }
  else if (++hold >= STREAMING_HITLOD_INSTANCE_HOLD_FRAMES)
  {
    applied--;
    hold = 0;
  }

  uint stateNew = applied | (hold << 8);
  if (stateNew == state)
    return;

  build.instanceHitScale.d[instanceID] = stateNew;

  if (applied != (state & 0xFF))
  {
#if USE_PARTITION_EARLY_OUT
    atomicAdd(build.partitionNonPersistent.d[build.instancePartitions.d[instanceID]], 1);
#endif
    atomicAdd(readback.numInstanceHitScaleChanges, 1);
  }
}
