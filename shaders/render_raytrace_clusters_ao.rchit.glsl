/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Closest hit of the shadow rays in ray tracing: no shading, it only registers
  the hit for hit-based streaming (USE_HIT_STREAMING) and reports the occlusion.
  
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

/////////////////////////////////

#if USE_STREAMING && USE_HIT_STREAMING
#include "streaming.glsl"
#endif

/////////////////////////////////

#if USE_HIT_STREAMING_LOD
layout(location = 1) rayPayloadInEXT RayPayloadAO rayHitAO;
#else
layout(location = 1) rayPayloadInEXT float rayHitAO;
#endif

/////////////////////////////////

void main()
{
#if USE_STREAMING && USE_HIT_STREAMING
  uint clusterID  = gl_ClusterIDNV_;
#if USE_HIT_STREAMING_LOD
  uint instanceID = gl_InstanceID;
  // unified pixel-footprint cone -> geometry-error budget (see render_raytrace_clusters.rchit)
  float coneToError = build.errorOverDistanceThreshold / view.pixelAngle;
  streamingRegisterHitLod(clusterID, rayHitAO.coneWidth * coneToError, rayHitAO.coneSpread * coneToError, gl_WorldRayOriginEXT, gl_ObjectToWorldEXT);
  streamingRegisterInstanceDemand(instanceID, clusterID, rayHitAO.coneWidth * coneToError, rayHitAO.coneSpread * coneToError, gl_WorldRayOriginEXT,
                                  view.viewPos.xyz, gl_ObjectToWorldEXT, build.errorOverDistanceThreshold, view.nearPlane,
                                  build.instanceHitDemand);
#else
  streamingRegisterHit(clusterID);
#endif
#endif

#if USE_HIT_STREAMING_LOD
  rayHitAO.hit = 1;
#else
  rayHitAO = 1;
#endif
}