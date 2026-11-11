/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Ray query counterparts of the hit shaders, for `render_raytrace_rq.comp.glsl`
  and `render_pathtrace_rq.comp.glsl`. Only alpha masked clusters are not opaque,
  so only their triangles come back as candidates, everything else commits during
  traversal like a hit without any-hit shader.

  Assumes the includer declared the bindings of the hit shaders, `rayHitAO`, and
  included render_cluster_hit.glsl and, with hit streaming, streaming.glsl.

*/

#ifndef RENDER_RAYQUERY_GLSL
#define RENDER_RAYQUERY_GLSL

// the intersection argument has to be a constant
#define RQ_CLUSTER_RAY_HIT(name, committed)                                                                            \
  ClusterRayHit name(rayQueryEXT rq)                                                                                   \
  {                                                                                                                    \
    ClusterRayHit hit;                                                                                                 \
    hit.hitT       = rayQueryGetIntersectionTEXT(rq, committed);                                                       \
    hit.instanceID = rayQueryGetIntersectionInstanceIdEXT(rq, committed);                                              \
    hit.clusterID  = uint(rayQueryGetIntersectionClusterIdNV(rq, committed));                                          \
    hit.triangleID = rayQueryGetIntersectionPrimitiveIndexEXT(rq, committed);                                          \
    hit.bary       = rayQueryGetIntersectionBarycentricsEXT(rq, committed);                                            \
    vec3 pos[3];                                                                                                       \
    rayQueryGetIntersectionTriangleVertexPositionsEXT(rq, committed, pos);                                             \
    hit.pos0          = pos[0];                                                                                        \
    hit.pos1          = pos[1];                                                                                        \
    hit.pos2          = pos[2];                                                                                        \
    hit.objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rq, committed);                                        \
    hit.worldToObject = rayQueryGetIntersectionWorldToObjectEXT(rq, committed);                                        \
    hit.wRayOrigin    = rayQueryGetWorldRayOriginEXT(rq);                                                              \
    hit.wRayDir       = rayQueryGetWorldRayDirectionEXT(rq);                                                           \
    hit.oRayOrigin    = rayQueryGetIntersectionObjectRayOriginEXT(rq, committed);                                      \
    hit.oRayDir       = rayQueryGetIntersectionObjectRayDirectionEXT(rq, committed);                                   \
    return hit;                                                                                                        \
  }

RQ_CLUSTER_RAY_HIT(rqCandidateHit, false)
RQ_CLUSTER_RAY_HIT(rqCommittedHit, true)

// the AO / shadow ray of `render_shading.glsl`: the ray tracer's any-hit, AO closest-hit and miss.
// True when something was hit, `registerHit` for shadow rays, AO rays register nothing.
bool rqTraceOcclusion(uint flags, vec3 origin, float tMin, vec3 dir, float tMax, bool registerHit)
{
  rayQueryEXT rq;
  rayQueryInitializeEXT(rq, accelerationStructureEXT(build.tlasAddress), flags, 0xFF, origin, tMin, dir, tMax);
  while(rayQueryProceedEXT(rq))
  {
#if HAS_ALPHA_TEST
    ClusterRayHit hit = rqCandidateHit(rq);
    // a first-hit footprint, as `render_raytrace_clusters.rahit.glsl` for AO / shadow rays
    if(!clusterAlphaTest(hit, view.pixelAngle * hit.hitT, false, vec2(0)))
      continue;
#endif
    rayQueryConfirmIntersectionEXT(rq);
  }

  if(rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT)
    return false;

#if USE_AO_HITGROUP && USE_STREAMING && USE_HIT_STREAMING && !USE_PATHTRACING
  // as `render_raytrace_clusters_ao.rchit.glsl`
  if(!registerHit)
    return true;
  ClusterRayHit hit = rqCommittedHit(rq);
#if USE_HIT_STREAMING_LOD
  float coneToError = build.errorOverDistanceThreshold / view.pixelAngle;
  streamingRegisterHitLod(hit.clusterID, rayHitAO.coneWidth * coneToError, rayHitAO.coneSpread * coneToError,
                          hit.wRayOrigin, hit.objectToWorld);
  streamingRegisterInstanceDemand(hit.instanceID, hit.clusterID, rayHitAO.coneWidth * coneToError,
                                  rayHitAO.coneSpread * coneToError, hit.wRayOrigin, view.viewPos.xyz, hit.objectToWorld,
                                  build.errorOverDistanceThreshold, view.nearPlane, build.instanceHitDemand);
#else
  streamingRegisterHit(hit.clusterID);
#endif
#endif
  return true;
}

#endif  // RENDER_RAYQUERY_GLSL
