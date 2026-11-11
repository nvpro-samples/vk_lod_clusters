/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Minimal closest-hit for the basic path tracer. Unlike the regular
  ray-trace hit shader it does NO shading: it only reports the hit (instance,
  cluster and triangle IDs + barycentrics + hitT). All shading happens in
  render_pathtrace.rgen.glsl.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_tracing_position_fetch : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#extension GL_EXT_spirv_intrinsics : require

// at the time of writing, no GLSL extension was available, we leverage
// GL_EXT_spirv_intrinsics to hook up the new builtin.
spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

#include "shaderio.h"

/////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
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

/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

layout(location = 0) rayPayloadInEXT PathRayPayload rayHit;

#if USE_STREAMING && USE_HIT_STREAMING
#include "streaming.glsl"
#endif

#define CLUSTER_HIT_SKIP_ALPHA_TEST 1
#include "render_cluster_hit.glsl"
#include "render_pathtrace_clusters_hit.glsl"

/////////////////////////////////

void main()
{
  ptClosestHit(clusterRayHitBuiltins());
}
