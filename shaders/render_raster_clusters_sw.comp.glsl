/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================

  This compute shader rasterizes a single cluster using 64-bit atomics.
  It is a very basic implementation.

  With USE_HYBRIDPATH it stores the list index and triangle instead of a color,
  `render_hybridpath_raster_atomic.frag.glsl` turns them into visibility buffer ids.

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
#extension GL_KHR_shader_subgroup_arithmetic : require


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

#if HAS_TEXTURED_MATERIALS || (HAS_ALPHA_TEST && ALLOW_VERTEX_TEXCOORDS)
layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];
#endif

////////////////////////////////////////////

// threads per cluster, a workgroup holds SW_RASTER_CLUSTERS_PER_WORKGROUP of them
#if SW_RASTER_CLUSTERS_PER_WORKGROUP > 1
#define SW_CLUSTER_THREADS 32
#else
#define SW_CLUSTER_THREADS CLUSTER_TRIANGLE_COUNT
#endif
#define COMPUTE_WORKGROUP_SIZE (SW_CLUSTER_THREADS * SW_RASTER_CLUSTERS_PER_WORKGROUP)

layout(local_size_x = COMPUTE_WORKGROUP_SIZE) in;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + SW_CLUSTER_THREADS - 1) / SW_CLUSTER_THREADS);

////////////////////////////////////////////

#include "culling.glsl"
#include "texturing.glsl"
#include "render_shading.glsl"

// Software-raster alpha-mask gradient. Unlike ray tracing (which uses the isotropic ray cone), a
// rasterized triangle has a well-defined screen-space footprint, so default to the accurate anisotropic
// gradient (matching the hardware fragment path). Set to 0 for the cheaper isotropic footprint.
#define SW_ANISOTROPIC_GRADIENT 1


////////////////////////////////////////////

float edgeFunction(vec2 a, vec2 b, vec2 c, float winding)
{
  float edge = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x));
  return edge * winding;
}

////////////////////////////////////////////

shared RasterVertex s_vertices[CLUSTER_VERTEX_COUNT * SW_RASTER_CLUSTERS_PER_WORKGROUP];
#if HAS_ALPHA_TEST
shared vec2 s_texcoords[CLUSTER_VERTEX_COUNT * SW_RASTER_CLUSTERS_PER_WORKGROUP];
#endif

void main()
{
  uint workGroupID  = getWorkGroupIndex(gl_WorkGroupID);
  uint slot         = gl_LocalInvocationID.x / SW_CLUSTER_THREADS;
  uint lane         = gl_LocalInvocationID.x % SW_CLUSTER_THREADS;
  uint sharedBase   = slot * CLUSTER_VERTEX_COUNT;
  uint clusterIndex = workGroupID * SW_RASTER_CLUSTERS_PER_WORKGROUP + slot;
#if USE_16BIT_DISPATCH || SW_RASTER_CLUSTERS_PER_WORKGROUP > 1
  // the launch grid is rounded up, so trailing slots have no cluster to rasterize. They
  // load a valid one and skip the work, the barrier needs all threads.
#if HAS_ALPHA_TEST
  uint numRenderedClustersSW = build.numRenderedClustersAlphaSW;
#else
  uint numRenderedClustersSW = build.numRenderedClustersSW;
#endif
  bool isValid = clusterIndex < numRenderedClustersSW;
  uint loadID  = min(clusterIndex, numRenderedClustersSW - 1);
#else
  bool isValid = true;
  uint loadID  = clusterIndex;
#endif

#if HAS_ALPHA_TEST
  ClusterInfo cinfo = build.renderClusterInfosAlphaSW.d[loadID];
#else
  ClusterInfo cinfo = build.renderClusterInfosSW.d[loadID];
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

  uint vertMax  = cluster.vertexCountMinusOne;
  uint numVerts = isValid ? vertMax + 1 : 0;
  uint numTris  = isValid ? uint(cluster.triangleCountMinusOne) + 1 : 0;

#if USE_RENDER_STATS
  if(lane == 0 && isValid)
  {
  #if HAS_ALPHA_TEST
    atomicAdd(readback.numRenderedTrianglesAlphaSW, numTris);
  #else
    atomicAdd(readback.numRenderedTrianglesSW, numTris);
  #endif
  }
#endif

  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert     = lane + i * SW_CLUSTER_THREADS;
    uint vertLoad = min(vert, vertMax);

    vec3 oPos = oVertices.d[vertLoad];
#if USE_HYBRIDPATH && HYBRID_POSITION_TRUNCATE_BITS
    // as the CLAS build
    oPos = uintBitsToFloat(floatBitsToUint(oPos) & ~((1u << HYBRID_POSITION_TRUNCATE_BITS) - 1u));
#endif
    vec3 wPos = vec4(oPos, 1.0f) * instance.worldMatrix;
#if HAS_ALPHA_TEST
    vec2 oTex = oTexCoords.d[vertLoad];
#endif

    if(vert < numVerts)
    {
#if USE_HYBRIDPATH
      s_vertices[sharedBase + vert] = getRasterVertex(view.viewProjMatrixRender * vec4(wPos, 1));
#else
      s_vertices[sharedBase + vert] = getRasterVertex(view.viewProjMatrix * vec4(wPos, 1));
#endif
#if HAS_ALPHA_TEST
      s_texcoords[sharedBase + vert] = oTex;
#endif
    }
  }

  barrier();

  uint numRasteredTriangles = 0;

#if HAS_ALPHA_TEST
  uint texIndex = 0;
  bool texValid = false;
#endif

#if SW_CLUSTER_THREADS < CLUSTER_TRIANGLE_COUNT
  for(uint tri = lane; tri < numTris; tri += SW_CLUSTER_THREADS)
#else
  uint tri = lane;
  if(tri < numTris)
#endif
  {
    uint triLoad = tri;
    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0], localIndices.d[triLoad * 3 + 1], localIndices.d[triLoad * 3 + 2]) + sharedBase;
#if !USE_FORCED_TWO_SIDED
  #if USE_TWO_SIDED
    bool effectiveTwoSided = resolveTriangleTwoSided(instance, clusterRef, triLoad);
  #endif
    // a two sided triangle is turned to face the camera, the winding flip is for one sided ones
    bool swapWinding = instance.flipWinding != 0;
#if USE_TWO_SIDED
    if(effectiveTwoSided)
      swapWinding = !isFrontFacingSW(s_vertices[indices.x], s_vertices[indices.y], s_vertices[indices.z]);
#endif
    if(swapWinding)
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
    // the cluster test projects without the jitter, a triangle at the border can reach past it
    pixelMin = max(pixelMin, vec2(0));
    pixelMax = min(pixelMax, view.viewportf.xy);
    visible  = visible && all(lessThan(pixelMin, pixelMax));
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
#if USE_HYBRIDPATH
      uint packedColor = (loadID << HYBRIDPATH_VISBUFFER_TRIANGLE_BITS) | tri;
#elif USE_DEPTH_ONLY
      uint packedColor = packUnorm4x8(vec4(0, 0, 0, 1));
#else
      uint  triangleCountMinusOne = CLUSTER_TRIANGLE_COUNT - 1;
      float relative              = (float(tri) / float(triangleCountMinusOne)) * 0.25 + 0.75;
      vec4  color                 = vec4(colorizeID(clusterID) * relative, 1.0);
      uint  packedColor           = packUnorm4x8(color);
#endif
      float invTriArea = 1.0f / triArea;

#if HAS_ALPHA_TEST
      uint alphaTex = resolveAlphaMaskTextureIndex(instance, clusterRef, triLoad);
      texValid      = alphaTex != 0xFFFF;
      texIndex      = texValid ? alphaTex : 0;
      vec2 texGradDdx;
      vec2 texGradDdy;
#if SW_ANISOTROPIC_GRADIENT
      computeRasterTextureGradients(a.xy, b.xy, c.xy, oTexCoordA, oTexCoordB, oTexCoordC, invTriArea, winding, texGradDdx, texGradDdy);
#else
      float texGrad = computeRasterFootprintGrad(triArea, oTexCoordA, oTexCoordB, oTexCoordC);
      texGradDdx    = vec2(texGrad, 0.0);
      texGradDdy    = vec2(0.0, texGrad);
#endif
#endif

      // barycentrics stepped per pixel and row, as in CuRast (https://github.com/m-schuetz/CuRast)
      ivec2 pixelStart = ivec2(pixelMin);
      ivec2 pixelEnd   = ivec2(pixelMax);
      ivec2 pixel      = pixelStart;
      float baryScale  = winding * invTriArea;
      vec2  stepB      = vec2(a.xy.y - c.xy.y, c.xy.x - a.xy.x) * baryScale;
      vec2  stepC      = vec2(b.xy.y - a.xy.y, a.xy.x - b.xy.x) * baryScale;
      float rowB       = edgeFunction(c.xy, a.xy, pixelMin + 0.5, winding) * invTriArea;
      float rowC       = edgeFunction(a.xy, b.xy, pixelMin + 0.5, winding) * invTriArea;
      float baryB      = rowB;
      float baryC      = rowC;

      while(true)
      {
        float baryA = 1.0 - baryB - baryC;

        if(baryA >= 0 && baryB >= 0 && baryC >= 0)
        {
          float depth = a.z + baryB * (b.z - a.z) + baryC * (c.z - a.z);

#if HAS_ALPHA_TEST
          vec2  oTexCoord = oTexCoordA * baryA + oTexCoordB * baryB + oTexCoordC * baryC;
          float alpha     = texValid ? textureGrad(bindlessTextures[nonuniformEXT(texIndex)], oTexCoord, texGradDdx, texGradDdy).a : 1.0;
          if(alpha >= 0.333)
#endif
          {
            uint64_t u64 = packUint2x32(uvec2(packedColor, floatBitsToUint(depth)));
            imageAtomicMax(imgRasterAtomic, pixel, u64);
          }
        }

        pixel.x++;
        baryB += stepB.x;
        baryC += stepC.x;
        if(pixel.x >= pixelEnd.x)
        {
          pixel.x = pixelStart.x;
          pixel.y++;
          rowB += stepB.y;
          rowC += stepC.y;
          baryB = rowB;
          baryC = rowC;
          if(pixel.y >= pixelEnd.y)
            break;
        }
      }

#if USE_RENDER_STATS
      numRasteredTriangles++;
#endif
    }
  }
#if USE_RENDER_STATS
  numRasteredTriangles = subgroupAdd(numRasteredTriangles);
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
