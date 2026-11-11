/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  This mesh shader renders a single cluster.

  NV_mesh_shader variant with optional primitive culling,
  see `render_raster_clusters.mesh.glsl` for the EXT_mesh_shader variant.

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
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_NV_mesh_shader : require
#extension GL_EXT_control_flow_attributes : require

#include "shaderio.h"

// disable invalid config
#if USE_PRIMITIVE_CULLING && !USE_CULLING
#undef USE_PRIMITIVE_CULLING
#define USE_PRIMITIVE_CULLING 0
#endif

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

////////////////////////////////////////////

#if NEEDS_PRIMITIVE_IDS

#if USE_PERPRIMITIVE_OUT

// Used if we have less triangles than vertices per cluster.
// Stores cluster-uniform outputs as perprimitive output.

layout(location = 0) perprimitiveNV out PrimitiveAttributes
{
  uint clusterID;
  uint instanceID;
}
OUTPRIM[];

#if ALLOW_SHADING
layout(location = 2) out Interpolants
{
  vec3 wPos;
}
OUT[];
#endif

#else

// Otherwise rely on non-interpolated vertex outputs.

layout(location = 0) out Interpolants
{
  flat uint clusterID;
  flat uint instanceID;
#if ALLOW_SHADING
  vec3 wPos;
#endif
}
OUT[];

#endif
#endif

#if (!USE_DEPTH_ONLY && ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)) || (HAS_ALPHA_TEST && ALLOW_VERTEX_TEXCOORDS)
// maybe remove, could load triangle indices in fragment shader as well
layout(location = 3) out Interpolants2
{
  flat uint vertexID;
}
OUTBARY[];
#endif


////////////////////////////////////////////

#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE 32
#endif

layout(local_size_x = MESHSHADER_WORKGROUP_SIZE) in;
layout(max_vertices = CLUSTER_VERTEX_COUNT, max_primitives = CLUSTER_TRIANGLE_COUNT) out;
layout(triangles) out;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);

////////////////////////////////////////////

#if USE_PRIMITIVE_CULLING || USE_TWO_SIDED
#define CULLING_NO_HIZ
#include "culling.glsl"
#endif

#define TEXTURING_SKIP_BINDLESS 1
#include "texturing.glsl"


////////////////////////////////////////////

void main()
{
  uint workGroupID = gl_WorkGroupID.x;

#if USE_HYBRIDPATH
  // one list, opaque and alpha tested
  ClusterInfo cinfo = build.hybridClusterInfos.d[workGroupID];
#elif HAS_ALPHA_TEST
  ClusterInfo cinfo = build.renderClusterInfosAlpha.d[workGroupID];
#else
  ClusterInfo cinfo = build.renderClusterInfos.d[workGroupID];
#endif

  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;

  bool clusterFiltered = view.visFilterClusterID != ~0u && clusterID != view.visFilterClusterID;

  RenderInstance instance = instances[instanceID];
  Geometry       geometry = geometries[instance.geometryID];

#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
#endif
  Cluster cluster = clusterRef.d;

  uint vertMax = cluster.vertexCountMinusOne;
  uint triMax  = cluster.triangleCountMinusOne;

  if(clusterFiltered)
  {
    if(gl_LocalInvocationID.x == 0)
      gl_PrimitiveCountNV = 0;
    return;
  }
#if !USE_PRIMITIVE_CULLING
  if(gl_LocalInvocationID.x == 0)
  {
    gl_PrimitiveCountNV = triMax + 1;
  }
#endif

#if USE_RENDER_STATS
  if(gl_LocalInvocationID.x == 0)
  {
#if HAS_ALPHA_TEST
    atomicAdd(readback.numRenderedTrianglesAlpha, uint(triMax + 1));
#else
    atomicAdd(readback.numRenderedTriangles, uint(triMax + 1));
#endif
#if !USE_PRIMITIVE_CULLING
#if HAS_ALPHA_TEST
    atomicAdd(readback.numRasteredTrianglesAlpha, uint(triMax + 1));
#else
    atomicAdd(readback.numRasteredTriangles, uint(triMax + 1));
#endif
#endif
  }
#endif

  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(clusterRef));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));

  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert     = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint vertLoad = min(vert, vertMax);

    vec3 oPos = oVertices.d[vertLoad];
#if USE_HYBRIDPATH && HYBRID_POSITION_TRUNCATE_BITS
    // as the CLAS build
    oPos = uintBitsToFloat(floatBitsToUint(oPos) & ~((1u << HYBRID_POSITION_TRUNCATE_BITS) - 1u));
#endif
    vec3 wPos = vec4(oPos, 1.0f) * instance.worldMatrix;

    if(vert <= vertMax)
    {
      vec4 hPos                           = view.viewProjMatrixRender * vec4(wPos, 1);
      gl_MeshVerticesNV[vert].gl_Position = hPos;

#if NEEDS_PRIMITIVE_IDS
#if ALLOW_SHADING
      OUT[vert].wPos = wPos.xyz;
#endif
#if !USE_PERPRIMITIVE_OUT
      OUT[vert].clusterID  = clusterID;
      OUT[vert].instanceID = instanceID;
#endif
#endif

#if (!USE_DEPTH_ONLY && ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)) || (HAS_ALPHA_TEST && ALLOW_VERTEX_TEXCOORDS)
      OUTBARY[vert].vertexID = vert;
#endif
    }
  }

#if USE_PRIMITIVE_CULLING || USE_TWO_SIDED
  // triangle loop reads other threads' vertex positions
  barrier();
#endif

  uint triOutCount = 0;

  [[unroll]] for(uint i = 0; i < uint(MESHLET_TRIANGLE_ITERATIONS); i++)
  {
    uint tri     = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint triLoad = min(tri, triMax);

    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0], localIndices.d[triLoad * 3 + 1], localIndices.d[triLoad * 3 + 2]);
#if !USE_FORCED_TWO_SIDED
#if USE_TWO_SIDED
    bool effectiveTwoSided = resolveTriangleTwoSided(instance, clusterRef, triLoad);
#endif
    // a two sided triangle is turned to face the camera, the winding flip is for one sided ones
    bool swapWinding = instance.flipWinding != 0;
#if USE_TWO_SIDED
    if(effectiveTwoSided)
      swapWinding = !isFrontFacingHW(gl_MeshVerticesNV[indices.x].gl_Position, gl_MeshVerticesNV[indices.y].gl_Position,
                                     gl_MeshVerticesNV[indices.z].gl_Position);
#endif
    if(swapWinding)
    {
      indices.xy = indices.yx;
    }
#endif

#if USE_PRIMITIVE_CULLING
    bool isRendered = tri <= triMax
                      && testTriangleHW(gl_MeshVerticesNV[indices.x].gl_Position, gl_MeshVerticesNV[indices.y].gl_Position,
                                        gl_MeshVerticesNV[indices.z].gl_Position);

    uvec4 voteRendered = subgroupBallot(isRendered);

    uint triOut = subgroupBallotExclusiveBitCount(voteRendered) + triOutCount;
    triOutCount += subgroupBallotBitCount(voteRendered);
#else
    bool isRendered = tri <= triMax;
    uint triOut     = tri;
#endif

    if(isRendered)
    {
      gl_PrimitiveIndicesNV[triOut * 3 + 0] = indices.x;
      gl_PrimitiveIndicesNV[triOut * 3 + 1] = indices.y;
      gl_PrimitiveIndicesNV[triOut * 3 + 2] = indices.z;
#if NEEDS_PRIMITIVE_IDS
      gl_MeshPrimitivesNV[triOut].gl_PrimitiveID = int(tri);
#endif
#if USE_PERPRIMITIVE_OUT && NEEDS_PRIMITIVE_IDS
      OUTPRIM[triOut].instanceID = instanceID;
      OUTPRIM[triOut].clusterID  = clusterID;
#endif
    }
  }
#if USE_PRIMITIVE_CULLING
  if(gl_LocalInvocationID.x == 0)
  {
    gl_PrimitiveCountNV = triOutCount;
  }
#if USE_RENDER_STATS
  if(gl_LocalInvocationID.x == 0)
  {
#if HAS_ALPHA_TEST
    atomicAdd(readback.numRasteredTrianglesAlpha, triOutCount);
#else
    atomicAdd(readback.numRasteredTriangles, triOutCount);
#endif
  }
#endif
#endif
}
