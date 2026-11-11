/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  The ray tracer of `render_raytrace.rgen.glsl` as a compute shader with ray
  queries (--rayquery). The closest-hit shading is the pipeline's
  `render_raytrace_clusters_hit.glsl`, the any-hit, AO hit group and miss shaders
  become the candidate loops below and in `render_rayquery.glsl`.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require
#extension GL_EXT_ray_tracing_position_fetch : require
#extension GL_NV_cluster_acceleration_structure : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : require

#define USE_RAYQUERY 1
#define RT_LAUNCH_ID gl_GlobalInvocationID.xy
#define RT_LAUNCH_SIZE uvec2(view.viewport)

#include "shaderio.h"
#if USE_DLSS
#include "dlss_util.h"
#endif

layout(local_size_x = RAYQUERY_WORKGROUP, local_size_y = RAYQUERY_WORKGROUP) in;

//////////////////////////////////////////////////////////////

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

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba16f) uniform image2D imgColor;
#if USE_DLSS
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssAlbedo, rgba8)             uniform image2D imgDlssAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecAlbedo, rgba16f)       uniform image2D imgDlssSpecAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssNormalRoughness, rgba16f)  uniform image2D imgDlssNormalRoughness;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssMotion, rg16f)             uniform image2D imgDlssMotion;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecHitDist, r16f)         uniform image2D imgDlssSpecHitDist;
#endif

#if HAS_ALPHA_TEST || HAS_TEXTURED_MATERIALS
layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];
#endif

//////////////////////////////////////////////////////////////

// the payloads of the pipeline
RayPayload rayHit;
#if USE_HIT_STREAMING_LOD
RayPayloadAO rayHitAO;
#else
float rayHitAO;
#endif

//////////////////////////////////////////////////////////////

#define SUPPORTS_RT 1
// no hit shader builtins
#define SKIP_OBJECT_SPACE_HELPERS 1

#include "nvshaders/sky_functions.h.slang"
#include "attribute_encoding.h"
#include "texturing.glsl"
#if USE_STREAMING && USE_HIT_STREAMING
#include "streaming.glsl"
#endif
#include "render_cluster_hit.glsl"
#include "render_rayquery.glsl"
#include "render_shading.glsl"
#include "render_raytrace_clusters_hit.glsl"
#if ALLOW_SHADING
#include "environment.glsl"
#endif

//////////////////////////////////////////////////////////////

// traceRayEXT of the primary and the reflected ray, the any-hit, closest-hit and miss shaders in one
void rtTrace(uint flags, vec3 origin, float tMin, vec3 dir, float tMax)
{
  rayQueryEXT rq;
  rayQueryInitializeEXT(rq, accelerationStructureEXT(build.tlasAddress), flags, 0xff, origin, tMin, dir, tMax);
  while(rayQueryProceedEXT(rq))
  {
#if HAS_ALPHA_TEST
    // as `render_raytrace_clusters.rahit.glsl`
    ClusterRayHit hit = rqCandidateHit(rq);
    if(!clusterAlphaTest(hit, rayHit.coneWidth + rayHit.coneSpread * hit.hitT, false, vec2(0)))
      continue;
#endif
    rayQueryConfirmIntersectionEXT(rq);
  }

  if(rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
  {
    rtClosestHit(rqCommittedHit(rq));
    return;
  }

  // as `render_raytrace.rmiss.glsl`
  rayHit.hitT = 0;
#if ALLOW_SHADING
  vec3 skyColor = evalEnvironment(view.skyPhysical, view.envHdri, dir);
  rayHit.color  = skyColor;
#if USE_DLSS
  rayHit.dlssAlbedo          = vec4(skyColor, 1);
  rayHit.dlssNormalRoughness = vec4(0);
  rayHit.dlssSpecular        = vec3(0);
#endif
#elif !USE_DEPTH_ONLY
  rayHit.color = vec3(0);
#endif
}

//////////////////////////////////////////////////////////////

#include "render_raytrace_raygen.glsl"
