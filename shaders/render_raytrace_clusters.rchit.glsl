/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "render_shading.glsl"

/////////////////////////////////

void main()
{
  // get IDs
  uint clusterID  = gl_ClusterIDNV_;
  uint instanceID = gl_InstanceID;
  uint triangleID = gl_PrimitiveID;

  RenderInstance instance = instances[instanceID];
  Geometry geometry       = geometries[instance.geometryID];

  // Fetch cluster header
#if USE_STREAMING
  // dereference the cluster from the resident cluster table
  uint64_t clusterAddress = streaming.resident.clusters.d[clusterID];
#else
  // access the cluster data directly from the preloaded array 
  uint64_t clusterAddress = geometry.preloadedClusters.d[clusterID];
#endif
  Cluster_in clusterRef = Cluster_in(clusterAddress);
  Cluster cluster = clusterRef.d;
  
  uint visData = clusterID;
  
#if ALLOW_SHADING
  vec3s_in  oVertices      = vec3s_in(Cluster_getVertexPositions(Cluster_in(clusterRef)));
  uint8s_in localIndices   = uint8s_in(Cluster_getTriangleIndices(Cluster_in(clusterRef)));
  
  uvec3 triangleIndices    = uvec3(localIndices.d[triangleID * 3 + 0],
                                   localIndices.d[triangleID * 3 + 1],
                                   localIndices.d[triangleID * 3 + 2]);

  vec3 baryWeight = vec3((1.f - barycentrics[0] - barycentrics[1]), barycentrics[0], barycentrics[1]);

  vec3 oPos = baryWeight.x * gl_HitTriangleVertexPositionsEXT[0] +
              baryWeight.y * gl_HitTriangleVertexPositionsEXT[1] + 
              baryWeight.z * gl_HitTriangleVertexPositionsEXT[2];    
  vec3 wPos = vec3(gl_ObjectToWorldEXT * vec4(oPos, 1.0));
  
  if (view.visualize == VISUALIZE_LOD || view.visualize == VISUALIZE_GROUP)
  {
    if (view.visualize == VISUALIZE_LOD)
    {
      visData = floatBitsToUint(float(cluster.lodLevel) * instances[instanceID].maxLodLevelRcp);
    }
    else {
      uvec2 baseAddress =  unpackUint2x32(clusterAddress - cluster.groupChildIndex * Cluster_size);
      visData =  baseAddress.x ^ baseAddress.y;
    }
  }
  else if (view.visualize == VISUALIZE_TRIANGLE)
  {
    visData = clusterID * 256 + uint(triangleID);
  }

  vec4 wTangent = vec4(1);
  vec2 oTexCoord = vec2(1);
  vec3 oNormal;
  bool backFacing = false;
  
  mat3 worldMatrixI = mat3(instance.worldMatrixI);

#if ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS
  uint32s_in oNormals   = Cluster_getVertexNormals(clusterRef);
  vec2s_in   oTexCoords = Cluster_getVertexTexCoords(clusterRef);

  if(view.facetShading != 0 || (cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_NORMAL) == 0)
#endif
  {
    // Otherwise compute geometric normal
    vec3 e0 = gl_HitTriangleVertexPositionsEXT[1] - gl_HitTriangleVertexPositionsEXT[0];
    vec3 e1 = gl_HitTriangleVertexPositionsEXT[2] - gl_HitTriangleVertexPositionsEXT[0];
    oNormal    = (cross(e0, e1));
    
    backFacing = dot(oNormal, gl_ObjectRayDirectionEXT) > 0;
  }
#if ALLOW_VERTEX_NORMALS
  if(view.facetShading == 0)
  {
    uvec3 triNormalsPacked = uvec3(oNormals.d[triangleIndices.x], oNormals.d[triangleIndices.y], oNormals.d[triangleIndices.z]);
    vec3  triNormals[3];
    
    triNormals[0] = normal_unpack(triNormalsPacked.x);
    triNormals[1] = normal_unpack(triNormalsPacked.y);
    triNormals[2] = normal_unpack(triNormalsPacked.z);
      
    oNormal = baryWeight.x * triNormals[0] + 
              baryWeight.y * triNormals[1] + 
              baryWeight.z * triNormals[2];

  #if ALLOW_VERTEX_TANGENTS
    if ((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TANGENT) != 0)
    {
      vec4 tangent0 = tangent_unpack(triNormals[0], triNormalsPacked.x >> ATTRENC_NORMAL_BITS);
      wTangent.w    = tangent0.w;

      vec3 oTangent = baryWeight.x * tangent0.xyz + 
                      baryWeight.y * tangent_unpack(triNormals[1],triNormalsPacked.y >> ATTRENC_NORMAL_BITS).xyz + 
                      baryWeight.z * tangent_unpack(triNormals[2],triNormalsPacked.z >> ATTRENC_NORMAL_BITS).xyz;
            
      wTangent.xyz = oTangent * worldMatrixI;
    }
  #endif
  }
#endif
#if ALLOW_VERTEX_TEXCOORD_0
  if ((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TEX_0) != 0)
  {
    oTexCoord = baryWeight.x * oTexCoords.d[triangleIndices.x] + 
                baryWeight.y * oTexCoords.d[triangleIndices.y] + 
                baryWeight.z * oTexCoords.d[triangleIndices.z];
  }
#endif

  vec3 wNormal = normalize(vec3(oNormal * worldMatrixI));
  if(view.flipWinding == 1 || (view.flipWinding == 2 && backFacing))
  {
    wNormal = -wNormal;
  }

  vec4 shaded;
  {
    float ambientOcclusion =
        ambientOcclusion(wPos, wNormal, view.ambientOcclusionSamples, view.ambientOcclusionRadius * view.sceneSize);

    float sunContribution  = 1.0;
    vec3  directionToLight = view.skyParams.sunDirection;
    if(view.doShadow == 1)
      sunContribution = traceShadowRay(wPos, wNormal, directionToLight);

    shaded = shading(instanceID, wPos, wNormal, wTangent, oTexCoord, visData, sunContribution, ambientOcclusion
    #if USE_DLSS
      , rayHit.dlssAlbedo, rayHit.dlssSpecular, rayHit.dlssNormalRoughness
    #endif
    );
  }
#else
  float relative = (float(gl_PrimitiveID) / float(clusterRef.d.triangleCountMinusOne)) * 0.25 + 0.75;
  vec4 shaded = vec4(colorizeID(visData) * relative, 1.0);
#endif

#if DEBUG_VISUALIZATION && ALLOW_SHADING
  if(view.doWireframe != 0)
  {
    vec3 derivativeTargetX = gl_WorldToObjectEXT * vec4(gl_WorldRayOriginEXT + rayHit.color.xyz, 1);
    vec3 derivativeDirX    = derivativeTargetX.xyz - gl_ObjectRayOriginEXT;
    vec3 derivativeX = intersectRayTriangle(gl_ObjectRayOriginEXT, derivativeDirX, gl_HitTriangleVertexPositionsEXT[0], gl_HitTriangleVertexPositionsEXT[1], gl_HitTriangleVertexPositionsEXT[2]);
    derivativeX = abs(derivativeX - baryWeight);


    vec3 derivativeTargetY = gl_WorldToObjectEXT * vec4(gl_WorldRayOriginEXT + rayHit.differentialY.xyz, 1);
    vec3 derivativeDirY    = derivativeTargetY.xyz - gl_ObjectRayOriginEXT;
    vec3 derivativeY = intersectRayTriangle(gl_ObjectRayOriginEXT, derivativeDirY, gl_HitTriangleVertexPositionsEXT[0], gl_HitTriangleVertexPositionsEXT[1], gl_HitTriangleVertexPositionsEXT[2]);
    derivativeY = abs(derivativeY - baryWeight);

    vec3 derivative = max(derivativeX, derivativeY);

    rayHit.color.xyz = addWireframe(shaded.xyz, baryWeight, true, derivative, view.wireColor);
  }
  else
#endif
  {
    rayHit.color.xyz = shaded.xyz;
  }
  
  if(gl_LaunchIDEXT.xy == view.mousePosition)
  {
  #if !ALLOW_SHADING
    vec3 wPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  #endif
    vec4  projected            = (view.viewProjMatrix * vec4(wPos, 1.f));
    float depth                = projected.z / projected.w;
    readback.clusterTriangleId = packPickingValue((clusterID << 8) | triangleID, depth);
    readback.instanceId        = packPickingValue(instanceID, depth);
  }

  rayHit.hitT = gl_HitTEXT;
}