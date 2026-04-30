/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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

#if USE_ANISOTROPIC_GRADIENT
bool hasPrimaryRayPayload()
{
  return (gl_IncomingRayFlagsEXT & gl_RayFlagsSkipClosestHitShaderEXT) == 0;
}
#endif

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

  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(Cluster_in(clusterRef)));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(Cluster_in(clusterRef)));

  uvec3 triangleIndices =
      uvec3(localIndices.d[triangleID * 3 + 0], localIndices.d[triangleID * 3 + 1], localIndices.d[triangleID * 3 + 2]);

  vec3 baryWeight = vec3((1.f - barycentrics[0] - barycentrics[1]), barycentrics[0], barycentrics[1]);

  vec2     oTexCoord     = vec2(0);
  vec2     texGradDdx    = vec2(0);
  vec2     texGradDdy    = vec2(0);
  bool     texCoordValid = false;
  vec2s_in oTexCoords    = Cluster_getVertexTexCoords(clusterRef);

  vec2 uv0  = oTexCoords.d[triangleIndices.x];
  vec2 uv1  = oTexCoords.d[triangleIndices.y];
  vec2 uv2  = oTexCoords.d[triangleIndices.z];
  oTexCoord = baryWeight.x * uv0 + baryWeight.y * uv1 + baryWeight.z * uv2;

  vec3 pos0 = oVertices.d[triangleIndices.x];
  vec3 pos1 = oVertices.d[triangleIndices.y];
  vec3 pos2 = oVertices.d[triangleIndices.z];
#if USE_ANISOTROPIC_GRADIENT
  bool usedRayDifferentials = false;

  if(hasPrimaryRayPayload())
  {
    vec3 directionX = gl_WorldToObjectEXT * vec4(rayHit.rayDifferentialX, 0.0);
    vec3 directionY = gl_WorldToObjectEXT * vec4(rayHit.rayDifferentialY, 0.0);
    usedRayDifferentials = computeRayDifferentialTextureGradients(gl_ObjectRayOriginEXT, directionX, directionY, pos0, pos1,
                                                                  pos2, uv0, uv1, uv2, oTexCoord, texGradDdx, texGradDdy);
  }

  if(!usedRayDifferentials)
#endif
  {
    float texGrad = computeRayFootprintGrad(gl_HitTEXT, pixelAngle,
                                            computeTexelDensity(gl_ObjectToWorldEXT, pos0, pos1, pos2, uv0, uv1, uv2));
    texGradDdx    = vec2(texGrad, 0.0);
    texGradDdy    = vec2(0.0, texGrad);
  }

  uint texIndex = resolveAlphaMaskTextureIndex(instance, clusterRef, triangleID);
  float alpha   = textureGrad(bindlessTextures[nonuniformEXT(texIndex)], oTexCoord, texGradDdx, texGradDdy).a;
  if(alpha < 0.333)
  {
    ignoreIntersectionEXT;
  }
}
