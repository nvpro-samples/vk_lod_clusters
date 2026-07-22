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

/////////////////////////////////

void main()
{
  // get IDs
  uint clusterID  = gl_ClusterIDNV_;
  uint instanceID = gl_InstanceID;
  uint triangleID = gl_PrimitiveID;

  RenderInstance instance = instances[instanceID];
  Geometry       geometry = geometries[instance.geometryID];

  // Fetch cluster header
#if USE_STREAMING
  uint64_t clusterAddress = streaming.resident.clusters.d[clusterID];
#else
  uint64_t clusterAddress = geometry.preloadedClusters.d[clusterID];
#endif
  Cluster_in clusterRef = Cluster_in(clusterAddress);
  Cluster    cluster    = clusterRef.d;

  vec3s_in  oVertices    = Cluster_getVertexPositions(clusterRef);
  uint8s_in localIndices = Cluster_getTriangleIndices(clusterRef);

  uvec3 triangleIndices =
      uvec3(localIndices.d[triangleID * 3 + 0], localIndices.d[triangleID * 3 + 1], localIndices.d[triangleID * 3 + 2]);

  vec3 baryWeight = vec3((1.f - barycentrics[0] - barycentrics[1]), barycentrics[0], barycentrics[1]);

  vec2s_in oTexCoords = Cluster_getVertexTexCoords(clusterRef);
  vec2     uv0        = oTexCoords.d[triangleIndices.x];
  vec2     uv1        = oTexCoords.d[triangleIndices.y];
  vec2     uv2        = oTexCoords.d[triangleIndices.z];
  vec2     oTexCoord  = baryWeight.x * uv0 + baryWeight.y * uv1 + baryWeight.z * uv2;

  vec3 pos0 = oVertices.d[triangleIndices.x];
  vec3 pos1 = oVertices.d[triangleIndices.y];
  vec3 pos2 = oVertices.d[triangleIndices.z];

  // Propagate the ray cone (from the payload) to this candidate hit and build the same TexLOD the ray-gen
  // uses for material sampling, so the cutout honors TEXTURE_LOD_MODE too. width(t) = coneWidth + coneSpread*t.
  float coneWidth    = rayHit.coneWidth + rayHit.coneSpread * gl_HitTEXT;
  vec3  wGeoNormal   = normalize(cross(pos1 - pos0, pos2 - pos0) * mat3(instance.worldMatrixI));
  float incidence    = abs(dot(wGeoNormal, gl_WorldRayDirectionEXT));
  float texelDensity = computeTexelDensity(gl_ObjectToWorldEXT, pos0, pos1, pos2, uv0, uv1, uv2);
  TexLOD texLod      = makeConeTexLOD(coneWidth, texelDensity, incidence);

  uint  texIndex = resolveAlphaMaskTextureIndex(instance, clusterRef, triangleID);
  float alpha    = sampleBindless(texIndex, oTexCoord, texLod).a;
  if(alpha < 0.333)
  {
    ignoreIntersectionEXT;
  }
}
