/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Any-hit for the basic path tracer: alpha-mask test for cutout geometry.
  Shared by primary/indirect and shadow rays (all carry PathRayPayload). It reads
  the propagated ray cone from the payload and reuses the same footprint helper
  (coneFootprintUV) as the ray-gen material sampling, so cutouts filter
  consistently at every bounce. Sampled alpha below the threshold ignores the hit.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_tracing_position_fetch : require

#extension GL_EXT_spirv_intrinsics : require

spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

#include "shaderio.h"

/////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
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

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
#endif

layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];

/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

// Same payload as the primary/indirect rays, so we can read the propagated ray cone. The ray-gen sets
// coneWidth/coneSpread before every trace (including shadow rays), which is why this any-hit can assume it.
layout(location = 0) rayPayloadInEXT PathRayPayload rayHit;

/////////////////////////////////

#include "attribute_encoding.h"
#include "texturing.glsl"
#include "render_cluster_hit.glsl"

/////////////////////////////////

void main()
{
  // the ray-gen sets the ray cone before every trace, width(t) = coneWidth + coneSpread*t.
  // A camera ray (--hybridmatchcheck) takes the raster's lod.
  float coneWidth = rayHit.coneWidth + rayHit.coneSpread * gl_HitTEXT;
  if(!clusterAlphaTest(clusterRayHitBuiltins(), coneWidth, rayHit.coneWidth == 0.0, vec2(gl_LaunchIDEXT.xy) + 0.5))
  {
    ignoreIntersectionEXT;
  }
}
