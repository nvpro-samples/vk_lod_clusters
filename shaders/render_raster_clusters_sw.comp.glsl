/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

/*
  
  Shader Description
  ==================

  This compute shader rasterizes a single cluster using 64-bit atomics.
  It is a very basic implementation.

*/


#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_image_int64 : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require


#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint instanceID;
}
push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
  FrameConstants viewLast;
};

layout(scalar,binding=BINDINGS_READBACK_SSBO,set=0) buffer readbackBuffer
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

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
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

#if USE_TWO_PASS_CULLING
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];
#else
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
#endif

layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;

////////////////////////////////////////////

#if CLUSTER_TRIANGLE_COUNT > 64
#define COMPUTE_WORKGROUP_SIZE 64
#else
#define COMPUTE_WORKGROUP_SIZE CLUSTER_TRIANGLE_COUNT
#endif

layout(local_size_x = COMPUTE_WORKGROUP_SIZE) in;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE);

////////////////////////////////////////////

#include "culling.glsl"
#include "render_shading.glsl"


////////////////////////////////////////////

float edgeFunction(vec2 a, vec2 b, vec2 c, float winding) 
{
  float edge = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x));
  return edge * winding;
}

void rasterTriangle(vec2 pixel, uint packedColor, uvec3 indices, RasterVertex a, RasterVertex b, RasterVertex c, float triArea, float winding)
{
  float baryA = edgeFunction(b.xy, c.xy, pixel, winding);
  float baryB = edgeFunction(c.xy, a.xy, pixel, winding);
  float baryC = edgeFunction(a.xy, b.xy, pixel, winding);

  if (baryA >= 0 && baryB >= 0 && baryC >= 0){
    baryA /= triArea;
    baryB /= triArea;
    baryC /= triArea;
    
    float depth = a.z * baryA + 
                  b.z * baryB + 
                  c.z * baryC;

    uint64_t u64 = packUint2x32(uvec2(packedColor, floatBitsToUint(depth)));
    imageAtomicMax(imgRasterAtomic, ivec2(pixel.xy), u64);
  }
}

////////////////////////////////////////////

shared RasterVertex s_vertices[CLUSTER_VERTEX_COUNT];

void main()
{
  uint workGroupID = getWorkGroupIndex(gl_WorkGroupID);
#if USE_16BIT_DISPATCH
  bool isValid      = workGroupID < build.numRenderedClustersSW;
  ClusterInfo cinfo = build.renderClusterInfosSW.d[min(workGroupID, build.numRenderedClustersSW-1)];
  if (!isValid) return;
#else
  ClusterInfo cinfo = build.renderClusterInfosSW.d[workGroupID];
#endif
  
  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;

  RenderInstance instance = instances[instanceID];
  Geometry geometry       = geometries[instance.geometryID];

#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
#endif
  Cluster cluster = clusterRef.d;
  
  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(clusterRef));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));

  uint vertMax = cluster.vertexCountMinusOne;
  uint triMax  = cluster.triangleCountMinusOne;
  
#if USE_RENDER_STATS
  if (gl_LocalInvocationID.x == 0) {
    atomicAdd(readback.numRenderedTrianglesSW, uint(triMax + 1));
  }
#endif
  
  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert        = gl_LocalInvocationID.x + i * COMPUTE_WORKGROUP_SIZE;
    uint vertLoad    = min(vert, vertMax);
    
    vec3 oPos = oVertices.d[vertLoad]; 
    vec3 wPos = instance.worldMatrix * vec4(oPos, 1.0f);

    if(vert <= vertMax)
    {
      s_vertices[vert] = getRasterVertex(view.viewProjMatrix * vec4(wPos,1));
    }
  }
  
  memoryBarrierShared();
  barrier();
  
  uint numRasteredTriangles = 0;
  
  for(uint tri = gl_LocalInvocationID.x; tri <= triMax; tri += COMPUTE_WORKGROUP_SIZE )
  {
    uint triLoad  = tri;
    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0],
                          localIndices.d[triLoad * 3 + 1],
                          localIndices.d[triLoad * 3 + 2]);
#if !USE_FORCED_TWO_SIDED
    if (instance.flipWinding != 0
#if USE_TWO_SIDED
      || (instance.twoSided != 0 && !isFrontFacingSW(s_vertices[indices.x],
                                                     s_vertices[indices.y],
                                                     s_vertices[indices.z]))
#endif
    )
    {
      indices.xy = indices.yx;
    }
#endif

    RasterVertex a = s_vertices[indices.x];
    RasterVertex b = s_vertices[indices.y];
    RasterVertex c = s_vertices[indices.z];
    
    vec2 pixelMin;
    vec2 pixelMax;
    float triArea;
    
    bool visible  = testTriangleSW(a,b,c, pixelMin, pixelMax, triArea);
    float winding = 1.0;
  #if USE_FORCED_TWO_SIDED
    if (triArea < 0)
    {
      triArea = -triArea;
      winding = -winding;
    }
  #endif
    
    if (visible) 
    {
      //uint triangleCountMinusOne = triMax;
      uint triangleCountMinusOne = CLUSTER_TRIANGLE_COUNT-1;
      float relative   = (float(tri) / float(triangleCountMinusOne)) * 0.25 + 0.75;
      
    #if USE_DEPTH_ONLY
      uint packedColor = packUnorm4x8(vec4(0,0,0,1));
    #else
      vec4 color       = vec4(colorizeID(clusterID) * relative, 1.0);
      uint packedColor = packUnorm4x8(color);
    #endif
    
      uvec2 pixelDim  = uvec2(pixelMax - pixelMin);

      for (uint p = 0; p < pixelDim.x * pixelDim.y; p++)
      {
        uint px = p % pixelDim.x;
        uint py = p / pixelDim.x;

        vec2 pixel = pixelMin + vec2(0.5) + vec2(px,py);
          
        rasterTriangle(pixel, packedColor, indices, a, b, c, triArea, winding);
      }
#if USE_RENDER_STATS
      numRasteredTriangles += subgroupBallotBitCount(subgroupBallot(true));
#endif
    }
  }
#if USE_RENDER_STATS
  if (subgroupElect()) {
    atomicAdd(readback.numRasteredTrianglesSW, numRasteredTriangles);
  }
#endif
}