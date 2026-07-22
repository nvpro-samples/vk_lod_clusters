/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Shared helper for the basic path tracer. In the path tracer the
  closest-hit shader only reports the hit (instance/cluster/triangle IDs +
  barycentrics + hitT); all shading happens in the ray-generation shader.

  This file factors out the cluster attribute reconstruction that the regular
  `render_raytrace_clusters.rchit.glsl` did inline. Unlike the hit shader, the
  ray-gen has no hit builtins (gl_HitTriangleVertexPositionsEXT, gl_ObjectToWorldEXT,
  ...), so positions/normals/tangents/texcoords are fetched from the cluster
  buffers and transformed with the instance matrices instead.

  Assumes the includer already declared the descriptor bindings (instances[],
  geometries[], materials[], streaming) and included attribute_encoding.h and
  texturing.glsl (for resolveMaterialID).

*/

#ifndef RENDER_PATHTRACE_HIT_GLSL
#define RENDER_PATHTRACE_HIT_GLSL

struct PathHit
{
  vec3 wPos;         // world-space hit position
  vec3 wShadowPos;   // world-space shadow-ray origin, offset per Hanika 2021 to hide the shadow terminator
  vec3 wNormal;      // world-space shading normal (front-facing)
  vec3 wGeoNormal;   // world-space geometric normal (front-facing), used for ray offsetting
  vec4 wTangent;     // world-space tangent (xyz) + sign (w)
  vec2 oTexCoord;    // interpolated texture coordinate
  float texelDensity; // sqrt(uvArea/worldArea) for the hit triangle, for ray-cone texture LOD (0 if untextured)
  uint materialID;   // resolved render material index
  bool backFacing;   // whether the ray hit the back side of the triangle
  uint visData;      // debug-visualization payload for the current view.visualize mode
};

// pointOffset() (Hanika 2021 shadow-terminator fix) lives in render_shading.glsl, which every includer
// of this file pulls in first.

PathHit getHitAttributes(uint instanceID, uint clusterID, uint triangleID, vec2 barycentrics, vec3 wRayOrigin, vec3 wRayDir, float hitT)
{
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

  vec3 pos0 = oVertices.d[triangleIndices.x];
  vec3 pos1 = oVertices.d[triangleIndices.y];
  vec3 pos2 = oVertices.d[triangleIndices.z];

  mat3 worldMatrixI = mat3(instance.worldMatrixI);

  PathHit hit;
  // world position straight from the ray parameter (matches the depth we output)
  hit.wPos      = wRayOrigin + wRayDir * hitT;
  // shadow origin defaults to the hit point; overridden with the terminator offset when the hit uses
  // smooth per-vertex normals (facet shading / no vertex normals leave it at wPos, i.e. no offset).
  hit.wShadowPos = hit.wPos;
  hit.materialID = instance.materialID;

  // always compute geometric normal (object space) from the triangle edges
  vec3 oGeoNormal = cross(pos1 - pos0, pos2 - pos0);
  // object-space ray direction (world->object uses the inverse linear part)
  vec3 oRayDir    = worldMatrixI * wRayDir;
  hit.backFacing  = dot(oGeoNormal, oRayDir) > 0;

  vec3 oNormal   = oGeoNormal;
  vec4 wTangent  = vec4(1);
  vec2 oTexCoord = vec2(1);

#if ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS
  uint32s_in oNormals   = Cluster_getVertexNormals(clusterRef);
  vec2s_in   oTexCoords = Cluster_getVertexTexCoords(clusterRef);
#endif

#if ALLOW_VERTEX_NORMALS
  if(view.facetShading == 0 && (cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_NORMAL) != 0)
  {
    uvec3 triNormalsPacked =
        uvec3(oNormals.d[triangleIndices.x], oNormals.d[triangleIndices.y], oNormals.d[triangleIndices.z]);
    vec3 triNormals[3];

    triNormals[0] = normal_unpack(triNormalsPacked.x);
    triNormals[1] = normal_unpack(triNormalsPacked.y);
    triNormals[2] = normal_unpack(triNormalsPacked.z);

    oNormal = baryWeight.x * triNormals[0] + baryWeight.y * triNormals[1] + baryWeight.z * triNormals[2];

    // Shadow terminator offset (Hanika 2021), computed in object space and transformed to world.
    // sideFlip keeps the offset on the visible side for back-face hits on two-sided meshes.
    float sideFlip = hit.backFacing ? -1.0 : 1.0;
    vec3  oHitPos  = baryWeight.x * pos0 + baryWeight.y * pos1 + baryWeight.z * pos2;
    vec3  oShadow  = pointOffset(oHitPos, pos0, pos1, pos2, triNormals[0] * sideFlip,
                                 triNormals[1] * sideFlip, triNormals[2] * sideFlip, baryWeight);
    hit.wShadowPos = instance.worldMatrix * vec4(oShadow, 1.0);

#if ALLOW_VERTEX_TANGENTS
    if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TANGENT) != 0)
    {
      vec4 tangent0 = tangent_unpack(triNormals[0], triNormalsPacked.x >> ATTRENC_NORMAL_BITS);
      wTangent.w    = tangent0.w;

      vec3 oTangent = baryWeight.x * tangent0.xyz
                      + baryWeight.y * tangent_unpack(triNormals[1], triNormalsPacked.y >> ATTRENC_NORMAL_BITS).xyz
                      + baryWeight.z * tangent_unpack(triNormals[2], triNormalsPacked.z >> ATTRENC_NORMAL_BITS).xyz;

      wTangent.xyz = normalize(oTangent * worldMatrixI);
    }
#endif
  }
#endif
  hit.texelDensity = 0.0;
#if ALLOW_VERTEX_TEXCOORD_0
  if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TEX_0) != 0)
  {
    vec2 uv0  = oTexCoords.d[triangleIndices.x];
    vec2 uv1  = oTexCoords.d[triangleIndices.y];
    vec2 uv2  = oTexCoords.d[triangleIndices.z];
    oTexCoord = baryWeight.x * uv0 + baryWeight.y * uv1 + baryWeight.z * uv2;
    // texel density (normalized-UV per world length) drives the ray-cone texture LOD in the ray-gen loop
    hit.texelDensity = computeTexelDensity(instance.worldMatrix, pos0, pos1, pos2, uv0, uv1, uv2);
  }
#endif

  hit.wGeoNormal = normalize(oGeoNormal * worldMatrixI);
  hit.wNormal    = normalize(oNormal * worldMatrixI);
  if(hit.backFacing)
  {
    hit.wNormal     = -hit.wNormal;
    hit.wGeoNormal  = -hit.wGeoNormal;
  }
  hit.wTangent   = wTangent;
  hit.oTexCoord  = oTexCoord;

#if ALLOW_SHADING
  hit.materialID = resolveMaterialID(instance, clusterRef, triangleID);
#endif

  // debug visualization payload (mirrors render_raytrace_clusters.rchit.glsl)
  uint visData = clusterID;
  if(view.visualize == VISUALIZE_LOD)
  {
    visData = floatBitsToUint(float(cluster.lodLevel) * instance.maxLodLevelRcp);
  }
  else if(view.visualize == VISUALIZE_GROUP)
  {
    uvec2 baseAddress = unpackUint2x32(clusterAddress - cluster.groupChildIndex * Cluster_size);
    visData           = baseAddress.x ^ baseAddress.y;
  }
  else if(view.visualize == VISUALIZE_TRIANGLE)
  {
    visData = clusterID * 256 + triangleID;
  }
  else if(view.visualize == VISUALIZE_MATERIAL)
  {
    visData = hit.materialID ^ 0x14325231;
  }
  hit.visData = visData;

  return hit;
}

#endif  // RENDER_PATHTRACE_HIT_GLSL
