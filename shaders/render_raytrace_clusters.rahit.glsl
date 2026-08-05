/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;


layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];

/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

/////////////////////////////////

layout(location = 0) rayPayloadInEXT RayPayload rayHit;
layout(location = 1) rayPayloadEXT float rayHitAO;

/////////////////////////////////


#define SUPPORTS_RT 1

#if USE_DLSS
#include "dlss_util.h"
#endif

#include "attribute_encoding.h"
#include "texturing.glsl"

/////////////////////////////////

// Primary/reflection rays run the closest hit; shadow & AO rays are traced with SkipClosestHit and carry a
// different (location 1) payload, so only the former have the RayPayload ray cone / differentials.
bool hasPrimaryRayPayload()
{
  return (gl_IncomingRayFlagsEXT & gl_RayFlagsSkipClosestHitShaderEXT) == 0;
}

void main()
{
  float pixelAngle = view.pixelAngle;

  // get IDs
  uint clusterID  = gl_ClusterIDNV_;
  uint instanceID = gl_InstanceID;
  uint triangleID = gl_PrimitiveID;

  RenderInstance instance = instances[instanceID];
  Geometry       geometry = geometries[instance.geometryID];

  // Fetch cluster header
#if USE_STREAMING
  // dereference the cluster from the resident cluster table
  uint64_t clusterAddress = streaming.resident.clusters.d[clusterID];
#else
  // access the cluster data directly from the preloaded array
  uint64_t clusterAddress = geometry.preloadedClusters.d[clusterID];
#endif
  Cluster_in clusterRef = Cluster_in(clusterAddress);
  Cluster    cluster    = clusterRef.d;

  uint visData = clusterID;

  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(Cluster_in(clusterRef)));

  uvec3 triangleIndices =
      uvec3(localIndices.d[triangleID * 3 + 0], localIndices.d[triangleID * 3 + 1], localIndices.d[triangleID * 3 + 2]);

  vec3 baryWeight = vec3((1.f - barycentrics[0] - barycentrics[1]), barycentrics[0], barycentrics[1]);

  vec2s_in oTexCoords = Cluster_getVertexTexCoords(clusterRef);

  vec2 uv0       = oTexCoords.d[triangleIndices.x];
  vec2 uv1       = oTexCoords.d[triangleIndices.y];
  vec2 uv2       = oTexCoords.d[triangleIndices.z];
  vec2 oTexCoord = baryWeight.x * uv0 + baryWeight.y * uv1 + baryWeight.z * uv2;

  // positions come from the CLAS via the ray-tracing position-fetch extension (no group-memory read)
  vec3 pos0 = gl_HitTriangleVertexPositionsEXT[0];
  vec3 pos1 = gl_HitTriangleVertexPositionsEXT[1];
  vec3 pos2 = gl_HitTriangleVertexPositionsEXT[2];

  // Isotropic ray-cone TexLOD - the same helper the closest-hit material sampling uses, so the cutout honors
  // TEXTURE_LOD_MODE too. Primary/reflection rays carry the propagated cone in the payload; shadow & AO rays
  // fall back to a first-hit footprint (ray length).
  float texelDensity = computeTexelDensity(gl_ObjectToWorldEXT, pos0, pos1, pos2, uv0, uv1, uv2);
  float coneWidth = hasPrimaryRayPayload() ? (rayHit.coneWidth + rayHit.coneSpread * gl_HitTEXT) : (pixelAngle * gl_HitTEXT);
  vec3  wGeoNormal = normalize(cross(pos1 - pos0, pos2 - pos0) * mat3(instance.worldMatrixI));
  float incidence  = abs(dot(wGeoNormal, gl_WorldRayDirectionEXT));
  TexLOD texLod    = makeConeTexLOD(coneWidth, texelDensity, incidence);

  uint texIndex = resolveAlphaMaskTextureIndex(instance, clusterRef, triangleID);
  float alpha   = sampleBindless(texIndex, oTexCoord, texLod).a;
  if(alpha < 0.333)
  {
    ignoreIntersectionEXT;
  }
}
