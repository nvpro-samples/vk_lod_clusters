/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  The path tracer's closest-hit, run by `render_pathtrace_clusters.rchit.glsl` and
  by the ray query compute shader `render_pathtrace_rq.comp.glsl`: reports the hit
  in `rayHit`, all shading happens in `render_pathtrace_raygen.glsl`.

*/

#ifndef RENDER_PATHTRACE_CLUSTERS_HIT_GLSL
#define RENDER_PATHTRACE_CLUSTERS_HIT_GLSL

void ptClosestHit(ClusterRayHit hit)
{
#if USE_STREAMING && USE_HIT_STREAMING
  // Register this hit for hit-based streaming so the path tracer drives residency (primary ray and every
  // GI bounce), mirroring render_raytrace_clusters.rchit. The ray-gen writes the propagated ray cone into
  // the payload before each trace.
  uint clusterID  = hit.clusterID;
  uint instanceID = hit.instanceID;
#if USE_HIT_STREAMING_LOD
  // unified pixel-footprint cone -> geometry-error budget (see render_raytrace_clusters.rchit)
  float coneToError = build.errorOverDistanceThreshold / view.pixelAngle;
  streamingRegisterHitLod(clusterID, rayHit.coneWidth * coneToError, rayHit.coneSpread * coneToError, hit.wRayOrigin, hit.objectToWorld);
  streamingRegisterInstanceDemand(instanceID, clusterID, rayHit.coneWidth * coneToError, rayHit.coneSpread * coneToError, hit.wRayOrigin,
                                  view.viewPos.xyz, hit.objectToWorld, build.errorOverDistanceThreshold, view.nearPlane,
                                  build.instanceHitDemand);
#else
  streamingRegisterHit(clusterID);
#endif
#endif

  rayHit.hitT       = hit.hitT;
  rayHit.instanceID = hit.instanceID;
  rayHit.clusterID  = hit.clusterID;
  rayHit.triangleID = hit.triangleID;
  rayHit.bary       = hit.bary;
  // CLAS positions; the ray-gen has no hit builtins to fetch them
#if PATHTRACE_HIT_POSITIONS
  rayHit.hitPos0 = hit.pos0;
  rayHit.hitPos1 = hit.pos1;
  rayHit.hitPos2 = hit.pos2;
#else
  // world-space geometric normal; its length is the triangle world area (ray-cone texture LOD)
  vec3 we1            = hit.objectToWorld * vec4(hit.pos1 - hit.pos0, 0.0);
  vec3 we2            = hit.objectToWorld * vec4(hit.pos2 - hit.pos0, 0.0);
  rayHit.hitGeoNormal = cross(we1, we2);
#endif

#if DEBUG_VISUALIZATION && ALLOW_SHADING
  // Barycentric footprint for the primary-hit wireframe overlay. The ray-gen has no hit builtins, so we
  // convert the ray-cone width at this hit into a per-pixel barycentric delta here (using the triangle's
  // world-space barycentric gradients) and hand it back. Cheap; only reached in debug-viz pipelines.
  {
#if PATHTRACE_HIT_POSITIONS
    vec3 we1 = hit.objectToWorld * vec4(hit.pos1 - hit.pos0, 0.0);
    vec3 we2 = hit.objectToWorld * vec4(hit.pos2 - hit.pos0, 0.0);
#endif
    float d00   = dot(we1, we1);
    float d01   = dot(we1, we2);
    float d11   = dot(we2, we2);
    float denom = max(d00 * d11 - d01 * d01, 1e-20);
    // gradients of the barycentric weights for vertices 1 and 2 w.r.t. world position in the triangle plane
    vec3  gb1   = (d11 * we1 - d01 * we2) / denom;
    vec3  gb2   = (d00 * we2 - d01 * we1) / denom;
    vec3  gb0   = -(gb1 + gb2);
    // cone width (world-space footprint diameter) at this hit; width(t) = coneWidth + coneSpread * t
    float coneW = rayHit.coneWidth + rayHit.coneSpread * hit.hitT;
    rayHit.wireBaryDeltas = coneW * vec3(length(gb0), length(gb1), length(gb2));
  }
#endif
}

#endif  // RENDER_PATHTRACE_CLUSTERS_HIT_GLSL
