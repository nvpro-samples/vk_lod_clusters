/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Hybrid path: writes the visibility buffer for the clusters of
  `build.hybridClusterInfos`, rasterized by `render_raster_clusters.mesh.glsl`.
  All shading happens in `render_hybridpath.rgen.glsl`. The alpha test uses the
  resolve's analytic texture lod (`makePixelTexLOD`).

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
#if HAS_ALPHA_TEST && ALLOW_VERTEX_TEXCOORDS
#extension GL_EXT_fragment_shader_barycentric : enable
#endif
#if USE_EXT_MESH_SHADER
#extension GL_EXT_mesh_shader : require
#else
#extension GL_NV_mesh_shader : require
#endif

#include "shaderio.h"

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

#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
#endif

#if HAS_ALPHA_TEST
layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];
#else
#define TEXTURING_SKIP_BINDLESS 1
#endif

///////////////////////////////////////////////////

#include "attribute_encoding.h"
#define TEXTURING_IMPLICIT_HAS_BIAS 1
#include "texturing.glsl"

///////////////////////////////////////////////////

#if USE_PERPRIMITIVE_OUT
layout(location = 0)
#if USE_EXT_MESH_SHADER
perprimitiveEXT
#else
perprimitiveNV
#endif
in PrimitiveAttributes
{
  uint clusterID;
  uint instanceID;
}
INPRIM;
#else
layout(location = 0) in Interpolants
{
  flat uint clusterID;
  flat uint instanceID;
}
IN;
#endif

#if HAS_ALPHA_TEST && ALLOW_VERTEX_TEXCOORDS
layout(location = 3) pervertexEXT in Interpolants2
{
  uint vertexID;
}
INBARY[];
#endif

layout(location = 0, index = 0) out uvec2 out_VisibilityBuffer;

#if !HAS_ALPHA_TEST
layout(early_fragment_tests) in;
#endif

///////////////////////////////////////////////////

void main()
{
#if USE_PERPRIMITIVE_OUT
  uint instanceID = INPRIM.instanceID;
  uint clusterID  = INPRIM.clusterID;
#else
  uint instanceID = IN.instanceID;
  uint clusterID  = IN.clusterID;
#endif

#if HAS_ALPHA_TEST && ALLOW_VERTEX_TEXCOORD_0
  RenderInstance instance = instances[instanceID];
#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometries[instance.geometryID].preloadedClusters.d[clusterID]);
#endif
  uint texIndex = resolveAlphaMaskTextureIndex(instance, clusterRef, gl_PrimitiveID);
  if(texIndex != 0xFFFF)
  {
    vec2s_in oTexCoords = Cluster_getVertexTexCoords(clusterRef);
    vec3s_in oPositions = Cluster_getVertexPositions(clusterRef);
    vec2     uv0        = oTexCoords.d[INBARY[0].vertexID];
    vec2     uv1        = oTexCoords.d[INBARY[1].vertexID];
    vec2     uv2        = oTexCoords.d[INBARY[2].vertexID];
    vec2     oTexCoord  = interpolateTexCoord(gl_BaryCoordEXT, uv0, uv1, uv2);
    vec3     oPos0      = oPositions.d[INBARY[0].vertexID];
    vec3     oPos1      = oPositions.d[INBARY[1].vertexID];
    vec3     oPos2      = oPositions.d[INBARY[2].vertexID];
#if HYBRID_POSITION_TRUNCATE_BITS
    // as the mesh shader and the resolve
    uint truncateMask = ~((1u << HYBRID_POSITION_TRUNCATE_BITS) - 1u);
    oPos0             = uintBitsToFloat(floatBitsToUint(oPos0) & truncateMask);
    oPos1             = uintBitsToFloat(floatBitsToUint(oPos1) & truncateMask);
    oPos2             = uintBitsToFloat(floatBitsToUint(oPos2) & truncateMask);
#endif
    // the resolve's lod, not the quad derivatives
    TexLOD texLod = makePixelTexLOD(instanceClipPos(instance, oPos0), instanceClipPos(instance, oPos1),
                                    instanceClipPos(instance, oPos2), gl_FragCoord.xy, uv0, uv1, uv2);
    if(sampleBindless(texIndex, oTexCoord, texLod).a < 0.333)
    {
      discard;
      return;
    }
  }
#endif

  out_VisibilityBuffer = uvec2(instanceID, (clusterID << HYBRIDPATH_VISBUFFER_TRIANGLE_BITS) | uint(gl_PrimitiveID));
}
