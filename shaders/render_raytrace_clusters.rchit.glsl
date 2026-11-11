/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  This hit shader handles the shading of clusters in
  ray tracing. 
  
  Note the use of a new input: `gl_ClusterIDNV`
  
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
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_tracing_position_fetch : require

#extension GL_EXT_spirv_intrinsics : require

// at the time of writing, no GLSL extension was available, we leverage
// GL_EXT_spirv_intrinsics to hook up the new builtin.
#extension GL_EXT_spirv_intrinsics : require

// Note that `VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV::allowClusterAccelerationStructure` must
// be set to `VK_TRUE` to make this valid.
spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

// While not required in this sample, as we use dedicated hit-shader for clusters,
// `int gl_ClusterIDNoneNV = -1;` can be used to dynamically detect regular hits.


#include "shaderio.h"

/////////////////////////////////

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

#if HAS_ALPHA_TEST || HAS_TEXTURED_MATERIALS
layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];
#endif

/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

/////////////////////////////////

layout(location = 0) rayPayloadInEXT RayPayload rayHit;
#if USE_HIT_STREAMING_LOD
layout(location = 1) rayPayloadEXT RayPayloadAO rayHitAO;
#else
layout(location = 1) rayPayloadEXT float rayHitAO;
#endif

/////////////////////////////////


#define SUPPORTS_RT 1

#if USE_DLSS
#include "dlss_util.h"
#endif

#include "attribute_encoding.h"
#include "texturing.glsl"
#include "render_shading.glsl"

#if USE_STREAMING && USE_HIT_STREAMING
#include "streaming.glsl"
#endif

#include "render_cluster_hit.glsl"
#include "render_raytrace_clusters_hit.glsl"

/////////////////////////////////

void main()
{
  rtClosestHit(clusterRayHitBuiltins());
}
