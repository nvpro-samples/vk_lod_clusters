/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Hybrid path: the primary hit from the rasterized visibility buffer. The camera ray
  is intersected with the rasterized triangle and the payload filled the way
  `render_pathtrace_clusters.rchit.glsl` would for a traced primary ray.

  Included by `render_pathtrace_raygen.glsl`, which declares the bindings and `rayHit`.

*/

#ifndef RENDER_HYBRIDPATH_HIT_GLSL
#define RENDER_HYBRIDPATH_HIT_GLSL

layout(set = 0, binding = BINDINGS_VISIBILITY_BUFFER, rg32ui) uniform uimage2D imgVisibilityBuffer;

#if USE_STREAMING && USE_HIT_STREAMING
#include "streaming.glsl"
#endif

// texture lod of the primary hit, the raster's
TexLOD hybridPrimaryTexLod;

// rayHit.hitT < 0 when nothing was rasterized
void hybridPrimaryHit(ivec2 screen, vec3 wRayOrigin, vec3 wRayDir, float coneWidth, float coneSpread, bool registerHit)
{
  rayHit.hitT       = -1.0;
  rayHit.coneWidth  = coneWidth;
  rayHit.coneSpread = coneSpread;

  hybridPrimaryTexLod = makeConeTexLOD(0.0, 0.0, 1.0);

  uvec2 ids = imageLoad(imgVisibilityBuffer, screen).xy;
  if(ids.x == HYBRIDPATH_VISBUFFER_EMPTY)
    return;

  uint instanceID = ids.x;
  uint clusterID  = ids.y >> HYBRIDPATH_VISBUFFER_TRIANGLE_BITS;
  uint triangleID = ids.y & ((1u << HYBRIDPATH_VISBUFFER_TRIANGLE_BITS) - 1u);

  RenderInstance instance = instances[instanceID];
#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometries[instance.geometryID].preloadedClusters.d[clusterID]);
#endif

  vec3s_in  oPositions   = Cluster_getVertexPositions(clusterRef);
  uint8s_in localIndices = Cluster_getTriangleIndices(clusterRef);

  uvec3 triangleIndices =
      uvec3(localIndices.d[triangleID * 3 + 0], localIndices.d[triangleID * 3 + 1], localIndices.d[triangleID * 3 + 2]);

  vec3 oPos0 = oPositions.d[triangleIndices.x];
  vec3 oPos1 = oPositions.d[triangleIndices.y];
  vec3 oPos2 = oPositions.d[triangleIndices.z];
#if HYBRID_POSITION_TRUNCATE_BITS
  // as the CLAS build and the raster
  uint truncateMask = ~((1u << HYBRID_POSITION_TRUNCATE_BITS) - 1u);
  oPos0 = uintBitsToFloat(floatBitsToUint(oPos0) & truncateMask);
  oPos1 = uintBitsToFloat(floatBitsToUint(oPos1) & truncateMask);
  oPos2 = uintBitsToFloat(floatBitsToUint(oPos2) & truncateMask);
#endif

#if ALLOW_VERTEX_TEXCOORD_0
  if((clusterRef.d.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TEX_0) != 0)
  {
    vec2s_in oTexCoords = Cluster_getVertexTexCoords(clusterRef);
    hybridPrimaryTexLod =
        makePixelTexLOD(instanceClipPos(instance, oPos0), instanceClipPos(instance, oPos1), instanceClipPos(instance, oPos2),
                        vec2(screen) + 0.5, oTexCoords.d[triangleIndices.x], oTexCoords.d[triangleIndices.y],
                        oTexCoords.d[triangleIndices.z]);
  }
#endif

  vec3 wPos0 = vec4(oPos0, 1.0) * instance.worldMatrix;
  vec3 we1   = vec4(oPos1 - oPos0, 0.0) * instance.worldMatrix;
  vec3 we2   = vec4(oPos2 - oPos0, 0.0) * instance.worldMatrix;

  // ray / triangle plane (Moeller-Trumbore), barycentrics may leave [0,1] slightly at raster edges
  vec3  pvec = cross(wRayDir, we2);
  float det  = dot(we1, pvec);
  float idet = 1.0 / (abs(det) > 1e-30 ? det : 1e-30);
  vec3  tvec = wRayOrigin - wPos0;
  vec3  qvec = cross(tvec, we1);

#if USE_STREAMING && USE_HIT_STREAMING
  // register the hit for hit-based streaming, as `render_pathtrace_clusters.rchit.glsl` does
  // for a traced primary ray
  if(registerHit)
  {
    mat4x3 objectToWorld = transpose(instance.worldMatrix);
#if USE_HIT_STREAMING_LOD
    float coneToError = build.errorOverDistanceThreshold / view.pixelAngle;
    streamingRegisterHitLod(clusterID, coneWidth * coneToError, coneSpread * coneToError, wRayOrigin, objectToWorld);
    streamingRegisterInstanceDemand(instanceID, clusterID, coneWidth * coneToError, coneSpread * coneToError, wRayOrigin,
                                    view.viewPos.xyz, objectToWorld, build.errorOverDistanceThreshold, view.nearPlane,
                                    build.instanceHitDemand);
#else
    streamingRegisterHit(clusterID);
#endif
  }
#endif

  rayHit.hitT       = max(dot(we2, qvec) * idet, 0.0);
  rayHit.instanceID = instanceID;
  rayHit.clusterID  = clusterID;
  rayHit.triangleID = triangleID;
  rayHit.bary       = vec2(dot(tvec, pvec), dot(wRayDir, qvec)) * idet;

#if PATHTRACE_HIT_POSITIONS
  rayHit.hitPos0 = oPos0;
  rayHit.hitPos1 = oPos1;
  rayHit.hitPos2 = oPos2;
#else
  rayHit.hitGeoNormal = cross(we1, we2);
#endif

#if DEBUG_VISUALIZATION && ALLOW_SHADING
  // wireframe footprint, as in the closest-hit
  {
    float d00   = dot(we1, we1);
    float d01   = dot(we1, we2);
    float d11   = dot(we2, we2);
    float denom = max(d00 * d11 - d01 * d01, 1e-20);
    vec3  gb1   = (d11 * we1 - d01 * we2) / denom;
    vec3  gb2   = (d00 * we2 - d01 * we1) / denom;
    vec3  gb0   = -(gb1 + gb2);
    float coneW = coneWidth + coneSpread * rayHit.hitT;
    rayHit.wireBaryDeltas = coneW * vec3(length(gb0), length(gb1), length(gb2));
  }
#endif
}

#endif  // RENDER_HYBRIDPATH_HIT_GLSL
