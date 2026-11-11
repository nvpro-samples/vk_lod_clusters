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

/////////////////////////////////

void main()
{
#if USE_STREAMING && USE_HIT_STREAMING
  // Register this hit for hit-based streaming so the path tracer drives residency (primary ray and every
  // GI bounce), mirroring render_raytrace_clusters.rchit. The ray-gen writes the propagated ray cone into
  // the payload before each trace.
  uint clusterID  = uint(gl_ClusterIDNV_);
  uint instanceID = gl_InstanceID;
#if USE_HIT_STREAMING_LOD
  // unified pixel-footprint cone -> geometry-error budget (see render_raytrace_clusters.rchit)
  float coneToError = build.errorOverDistanceThreshold / view.pixelAngle;
  streamingRegisterHitLod(clusterID, rayHit.coneWidth * coneToError, rayHit.coneSpread * coneToError, gl_WorldRayOriginEXT, gl_ObjectToWorldEXT);
  streamingRegisterInstanceDemand(instanceID, clusterID, rayHit.coneWidth * coneToError, rayHit.coneSpread * coneToError, gl_WorldRayOriginEXT,
                                  view.viewPos.xyz, gl_ObjectToWorldEXT, build.errorOverDistanceThreshold, view.nearPlane,
                                  build.instanceHitDemand);
#else
  streamingRegisterHit(clusterID);
#endif
#endif

  rayHit.hitT       = gl_HitTEXT;
  rayHit.instanceID = gl_InstanceID;
  rayHit.clusterID  = uint(gl_ClusterIDNV_);
  rayHit.triangleID = gl_PrimitiveID;
  rayHit.bary       = barycentrics;
  // CLAS positions; the ray-gen has no hit builtins to fetch them
#if PATHTRACE_HIT_POSITIONS
  rayHit.hitPos0 = gl_HitTriangleVertexPositionsEXT[0];
  rayHit.hitPos1 = gl_HitTriangleVertexPositionsEXT[1];
  rayHit.hitPos2 = gl_HitTriangleVertexPositionsEXT[2];
#else
  // world-space geometric normal; its length is the triangle world area (ray-cone texture LOD)
  vec3 we1            = gl_ObjectToWorldEXT * vec4(gl_HitTriangleVertexPositionsEXT[1] - gl_HitTriangleVertexPositionsEXT[0], 0.0);
  vec3 we2            = gl_ObjectToWorldEXT * vec4(gl_HitTriangleVertexPositionsEXT[2] - gl_HitTriangleVertexPositionsEXT[0], 0.0);
  rayHit.hitGeoNormal = cross(we1, we2);
#endif

#if DEBUG_VISUALIZATION && ALLOW_SHADING
  // Barycentric footprint for the primary-hit wireframe overlay. The ray-gen has no hit builtins, so we
  // convert the ray-cone width at this hit into a per-pixel barycentric delta here (using the triangle's
  // world-space barycentric gradients) and hand it back. Cheap; only reached in debug-viz pipelines.
  {
#if PATHTRACE_HIT_POSITIONS
    vec3 we1 = gl_ObjectToWorldEXT * vec4(gl_HitTriangleVertexPositionsEXT[1] - gl_HitTriangleVertexPositionsEXT[0], 0.0);
    vec3 we2 = gl_ObjectToWorldEXT * vec4(gl_HitTriangleVertexPositionsEXT[2] - gl_HitTriangleVertexPositionsEXT[0], 0.0);
#endif
    float d00   = dot(we1, we1);
    float d01   = dot(we1, we2);
    float d11   = dot(we2, we2);
    float denom = max(d00 * d11 - d01 * d01, 1e-20);
    // gradients of the barycentric weights for vertices 1 and 2 w.r.t. world position in the triangle plane
    vec3  gb1   = (d11 * we1 - d01 * we2) / denom;
    vec3  gb2   = (d00 * we2 - d01 * we1) / denom;
    vec3  gb0   = -(gb1 + gb2);
    // cone width (world-space footprint diameter) at this hit; width(t) = coneWidth + coneSpread * t
    float coneW = rayHit.coneWidth + rayHit.coneSpread * gl_HitTEXT;
    rayHit.wireBaryDeltas = coneW * vec3(length(gb0), length(gb1), length(gb2));
  }
#endif
}
