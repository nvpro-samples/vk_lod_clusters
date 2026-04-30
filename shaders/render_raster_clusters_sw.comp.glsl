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
#extension GL_EXT_nonuniform_qualifier : require
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
layout(binding = BINDINGS_HIZ_TEX) uniform sampler2D texHizFar[2];
#else
layout(binding = BINDINGS_HIZ_TEX) uniform sampler2D texHizFar;
#endif

layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;

#if HAS_ALPHA_TEST && ALLOW_VERTEX_TEXCOORDS
layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];
#endif

////////////////////////////////////////////

#if CLUSTER_TRIANGLE_COUNT > 64
#define COMPUTE_WORKGROUP_SIZE CLUSTER_TRIANGLE_COUNT
#else
#define COMPUTE_WORKGROUP_SIZE CLUSTER_TRIANGLE_COUNT
#endif

layout(local_size_x = COMPUTE_WORKGROUP_SIZE) in;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE);

////////////////////////////////////////////

#include "culling.glsl"
#include "render_shading.glsl"
#include "texturing.glsl"


////////////////////////////////////////////

float edgeFunction(vec2 a, vec2 b, vec2 c, float winding)
{
  float edge = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x));
  return edge * winding;
}

////////////////////////////////////////////

shared RasterVertex s_vertices[CLUSTER_VERTEX_COUNT];
#if HAS_ALPHA_TEST
shared vec2 s_texcoords[CLUSTER_VERTEX_COUNT];
#endif

void main()
{
  uint workGroupID = getWorkGroupIndex(gl_WorkGroupID);
#if USE_16BIT_DISPATCH
#if HAS_ALPHA_TEST
  uint numRenderedClustersSW = build.numRenderedClustersAlphaSW
#else
  uint numRenderedClustersSW = build.numRenderedClustersSW;
#endif
  bool isValid = workGroupID < numRenderedClustersSW;
  uint loadID = min(workGroupID, numRenderedClustersSW - 1) ClusterInfo cinfo = build.renderClusterInfosSW.d[];
#else
  uint loadID = workGroupID;
#endif

#if HAS_ALPHA_TEST
  ClusterInfo cinfo = build.renderClusterInfosAlphaSW.d[loadID];
#else
  ClusterInfo cinfo = build.renderClusterInfosSW.d[loadID];
#endif

#if USE_16BIT_DISPATCH
  if(!isValid)
    return;
#endif

  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;

  RenderInstance instance = instances[instanceID];
  Geometry       geometry = geometries[instance.geometryID];

#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
#endif
  Cluster cluster = clusterRef.d;

  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(clusterRef));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));

#if HAS_ALPHA_TEST
  vec2s_in oTexCoords = Cluster_getVertexTexCoords(clusterRef);
#endif

  uint vertMax = cluster.vertexCountMinusOne;
  uint triMax  = cluster.triangleCountMinusOne;

#if USE_RENDER_STATS
  if(gl_LocalInvocationID.x == 0)
  {
  #if HAS_ALPHA_TEST
    atomicAdd(readback.numRenderedTrianglesAlphaSW, uint(triMax + 1));
  #else
    atomicAdd(readback.numRenderedTrianglesSW, uint(triMax + 1));
  #endif
  }
#endif

  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert     = gl_LocalInvocationID.x + i * COMPUTE_WORKGROUP_SIZE;
    uint vertLoad = min(vert, vertMax);

    vec3 oPos = oVertices.d[vertLoad];
    vec3 wPos = instance.worldMatrix * vec4(oPos, 1.0f);
#if HAS_ALPHA_TEST
    vec2 oTex = oTexCoords.d[vertLoad];
#endif

    if(vert <= vertMax)
    {
      s_vertices[vert] = getRasterVertex(view.viewProjMatrix * vec4(wPos, 1));
#if HAS_ALPHA_TEST
      s_texcoords[vert] = oTex;
#endif
    }
  }

  barrier();

  uint numRasteredTriangles = 0;

#if COMPUTE_WORKGROUP_SIZE < CLUSTER_TRIANGLE_COUNT
  for(uint tri = gl_LocalInvocationID.x; tri <= triMax; tri += COMPUTE_WORKGROUP_SIZE)
#else
  uint tri = gl_LocalInvocationID.x;
  if(tri <= triMax)
#endif
  {
    uint triLoad = tri;
    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0], localIndices.d[triLoad * 3 + 1], localIndices.d[triLoad * 3 + 2]);
#if !USE_FORCED_TWO_SIDED
  #if USE_TWO_SIDED
    bool effectiveTwoSided = resolveTriangleTwoSided(instance, clusterRef, triLoad);
  #endif
    if(instance.flipWinding != 0
#if USE_TWO_SIDED
       || (effectiveTwoSided && !isFrontFacingSW(s_vertices[indices.x], s_vertices[indices.y], s_vertices[indices.z]))
#endif
    )
    {
      indices.xy = indices.yx;
    }
#endif

    RasterVertex a = s_vertices[indices.x];
    RasterVertex b = s_vertices[indices.y];
    RasterVertex c = s_vertices[indices.z];
#if HAS_ALPHA_TEST
    vec2 oTexCoordA = s_texcoords[indices.x];
    vec2 oTexCoordB = s_texcoords[indices.y];
    vec2 oTexCoordC = s_texcoords[indices.z];
#endif

    vec2  pixelMin;
    vec2  pixelMax;
    float triArea;

    bool  visible = testTriangleSW(a, b, c, pixelMin, pixelMax, triArea);
    float winding = 1.0;
#if USE_FORCED_TWO_SIDED
    if(triArea < 0)
    {
      triArea = -triArea;
      winding = -winding;
    }
#endif
    if(visible)
    {
#if USE_DEPTH_ONLY
      uint packedColor = packUnorm4x8(vec4(0, 0, 0, 1));
#else
      uint  triangleCountMinusOne = CLUSTER_TRIANGLE_COUNT - 1;
      float relative              = (float(tri) / float(triangleCountMinusOne)) * 0.25 + 0.75;
      vec4  color                 = vec4(colorizeID(clusterID) * relative, 1.0);
      uint  packedColor           = packUnorm4x8(color);
#endif
      float invTriArea = 1.0f / triArea;

#if HAS_ALPHA_TEST
      uint texIndex = resolveAlphaMaskTextureIndex(instance, clusterRef, triLoad);
      vec2 texGradDdx;
      vec2 texGradDdy;
#if USE_ANISOTROPIC_GRADIENT
      computeRasterTextureGradients(a.xy, b.xy, c.xy, oTexCoordA, oTexCoordB, oTexCoordC, invTriArea, winding, texGradDdx, texGradDdy);
#else
      float texGrad = computeRasterFootprintGrad(triArea, oTexCoordA, oTexCoordB, oTexCoordC);
      texGradDdx    = vec2(texGrad, 0.0);
      texGradDdy    = vec2(0.0, texGrad);
#endif
#endif

      pixelMin += vec2(0.5);
      vec2 pixel = pixelMin;

      while(true)
      {
        float baryA = edgeFunction(b.xy, c.xy, pixel, winding);
        float baryB = edgeFunction(c.xy, a.xy, pixel, winding);
        float baryC = edgeFunction(a.xy, b.xy, pixel, winding);

        if(baryA >= 0 && baryB >= 0 && baryC >= 0)
        {
          baryA *= invTriArea;
          baryB *= invTriArea;
          baryC *= invTriArea;

          float depth = a.z * baryA + b.z * baryB + c.z * baryC;

#if HAS_ALPHA_TEST
          vec2  oTexCoord = oTexCoordA * baryA + oTexCoordB * baryB + oTexCoordC * baryC;
          float alpha     = texIndex != 0xFFFF ? textureGrad(bindlessTextures[nonuniformEXT(texIndex)], oTexCoord, texGradDdx, texGradDdy).a : 1.0;
          if(alpha >= 0.333)
#endif
          {
            uint64_t u64 = packUint2x32(uvec2(packedColor, floatBitsToUint(depth)));
            imageAtomicMax(imgRasterAtomic, ivec2(pixel.xy), u64);
          }
        }

        pixel.x++;
        if(pixel.x > pixelMax.x)
        {
          pixel.x = pixelMin.x;
          pixel.y++;
          if(pixel.y > pixelMax.y)
            break;
        }
      }

#if USE_RENDER_STATS
      numRasteredTriangles += subgroupBallotBitCount(subgroupBallot(true));
#endif
    }
  }
#if USE_RENDER_STATS
  if(subgroupElect())
  {
  #if HAS_ALPHA_TEST
    atomicAdd(readback.numRasteredTrianglesAlphaSW, numRasteredTriangles);
  #else
    atomicAdd(readback.numRasteredTrianglesSW, numRasteredTriangles);
  #endif
  }
#endif
}
