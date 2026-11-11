/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  A ray's hit on a cluster triangle, filled from the hit shader builtins or from
  a ray query (`render_rayquery.glsl`), and the alpha mask test that the any-hit
  shaders and the ray query candidate loops share.

  Assumes the includer declared instances[], geometries[], streaming and included
  attribute_encoding.h and texturing.glsl. A hit shader also declares `gl_ClusterIDNV_`
  and the `barycentrics` hit attribute. CLUSTER_HIT_SKIP_ALPHA_TEST opts out of the
  alpha test and its dependencies.

*/

#ifndef RENDER_CLUSTER_HIT_GLSL
#define RENDER_CLUSTER_HIT_GLSL

#ifndef USE_RAYQUERY
#define USE_RAYQUERY 0
#endif

struct ClusterRayHit
{
  float  hitT;
  uint   instanceID;
  uint   clusterID;
  uint   triangleID;
  vec2   bary;
  // object space, from the CLAS
  vec3   pos0;
  vec3   pos1;
  vec3   pos2;
  mat4x3 objectToWorld;
  mat4x3 worldToObject;
  vec3   wRayOrigin;
  vec3   wRayDir;
  vec3   oRayOrigin;
  vec3   oRayDir;
};

#if !USE_RAYQUERY
ClusterRayHit clusterRayHitBuiltins()
{
  ClusterRayHit hit;
  hit.hitT          = gl_HitTEXT;
  hit.instanceID    = gl_InstanceID;
  hit.clusterID     = uint(gl_ClusterIDNV_);
  hit.triangleID    = gl_PrimitiveID;
  hit.bary          = barycentrics;
  hit.pos0          = gl_HitTriangleVertexPositionsEXT[0];
  hit.pos1          = gl_HitTriangleVertexPositionsEXT[1];
  hit.pos2          = gl_HitTriangleVertexPositionsEXT[2];
  hit.objectToWorld = gl_ObjectToWorldEXT;
  hit.worldToObject = gl_WorldToObjectEXT;
  hit.wRayOrigin    = gl_WorldRayOriginEXT;
  hit.wRayDir       = gl_WorldRayDirectionEXT;
  hit.oRayOrigin    = gl_ObjectRayOriginEXT;
  hit.oRayDir       = gl_ObjectRayDirectionEXT;
  return hit;
}
#endif

#if HAS_ALPHA_TEST && !defined(CLUSTER_HIT_SKIP_ALPHA_TEST)
// true when the alpha mask keeps the hit. `coneWidth` is the ray cone width at the hit,
// `pixelLod` takes the raster's lod at `pixel` instead (a hybrid path camera ray)
bool clusterAlphaTest(ClusterRayHit hit, float coneWidth, bool pixelLod, vec2 pixel)
{
  RenderInstance instance = instances[hit.instanceID];

#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[hit.clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometries[instance.geometryID].preloadedClusters.d[hit.clusterID]);
#endif

  uint8s_in localIndices = Cluster_getTriangleIndices(clusterRef);
  uvec3     triangleIndices = uvec3(localIndices.d[hit.triangleID * 3 + 0], localIndices.d[hit.triangleID * 3 + 1],
                                    localIndices.d[hit.triangleID * 3 + 2]);

  vec3 baryWeight = vec3((1.f - hit.bary.x - hit.bary.y), hit.bary.x, hit.bary.y);

  vec2s_in oTexCoords = Cluster_getVertexTexCoords(clusterRef);
  vec2     uv0        = oTexCoords.d[triangleIndices.x];
  vec2     uv1        = oTexCoords.d[triangleIndices.y];
  vec2     uv2        = oTexCoords.d[triangleIndices.z];
  vec2     oTexCoord  = baryWeight.x * uv0 + baryWeight.y * uv1 + baryWeight.z * uv2;

  // the same ray-cone TexLOD the material sampling uses, so the cutout honors TEXTURE_LOD_MODE too
  float  texelDensity = computeTexelDensity(hit.objectToWorld, hit.pos0, hit.pos1, hit.pos2, uv0, uv1, uv2);
  vec3   wGeoNormal   = normalize(mat3(instance.worldMatrixI) * cross(hit.pos1 - hit.pos0, hit.pos2 - hit.pos0));
  float  incidence    = abs(dot(wGeoNormal, hit.wRayDir));
  TexLOD texLod       = makeConeTexLOD(coneWidth, texelDensity, incidence);
#if USE_HYBRIDPATH
  if(pixelLod)
    texLod = makePixelTexLOD(instanceClipPos(instance, hit.pos0), instanceClipPos(instance, hit.pos1),
                             instanceClipPos(instance, hit.pos2), pixel, uv0, uv1, uv2);
#endif

  uint texIndex = resolveAlphaMaskTextureIndex(instance, clusterRef, hit.triangleID);
  return sampleBindless(texIndex, oTexCoord, texLod).a >= 0.333;
}
#endif

#endif  // RENDER_CLUSTER_HIT_GLSL
