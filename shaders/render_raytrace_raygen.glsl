/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Ray generation of the ray tracer, included by `render_raytrace.rgen.glsl` and
  by the ray query compute shader `render_raytrace_rq.comp.glsl`, which defines
  USE_RAYQUERY, the launch id and its own rtTrace().

*/

#ifndef USE_RAYQUERY
#define USE_RAYQUERY 0
#endif

#ifndef RT_LAUNCH_ID
#define RT_LAUNCH_ID gl_LaunchIDEXT.xy
#define RT_LAUNCH_SIZE gl_LaunchSizeEXT.xy
#endif

#if !USE_RAYQUERY
// the primary and the reflected ray, payload `rayHit`
#define rtTrace(flags, origin, tMin, dir, tMax)                                                                        \
  traceRayEXT(accelerationStructureEXT(build.tlasAddress), flags, 0xff, 0, 1, 0, origin, tMin, dir, tMax, 0)
#endif

//////////////////////////////////////////////////////////////

float hitBbox(vec3 rayPos, vec3 rayDir, vec3 bboxMin, vec3 bboxMax)
{
  vec3  invDir = 1.0 / (rayDir);
  vec3  tbot   = invDir * (bboxMin - rayPos);
  vec3  ttop   = invDir * (bboxMax - rayPos);
  vec3  tmin   = min(ttop, tbot);
  vec3  tmax   = max(ttop, tbot);
  float t0     = max(tmin.x, max(tmin.y, tmin.z));
  float t1     = min(tmax.x, min(tmax.y, tmax.z));

  return t1 > max(t0, 0.0) ? t0 : -1.0;
}

// near intersection with a sphere (the outside of its surface)
float hitSphere(vec3 rayPos, vec3 rayDir, vec3 center, float radius)
{
  vec3  oc = rayPos - center;
  float b  = dot(oc, rayDir);
  float c  = dot(oc, oc) - radius * radius;
  float h  = b * b - c;
  if(h < 0.0)
    return -1.0;
  float t = -b - sqrt(h);
  return t > 0.0 ? t : -1.0;
}

// intersection with a flat circular aperture (a lens opening) at 'center',
// facing along 'normal', clipped to 'radius'
float hitDisc(vec3 rayPos, vec3 rayDir, vec3 center, vec3 normal, float radius)
{
  float denom = dot(rayDir, normal);
  if(denom == 0.0)
    return -1.0;
  float t = dot(center - rayPos, normal) / denom;
  if(t <= 0.0)
    return -1.0;
  vec3 p = rayPos + rayDir * t;
  return dot(p - center, p - center) <= radius * radius ? t : -1.0;
}

// direction through the lens: a narrower rectilinear camera magnified by lensZoom (like a zoom
// lens rather than a refracting surface), plus a radial distortion that reaches lensDistortion
// at the rim
vec3 lensZoomDirection(vec3 wDir)
{
  vec3  vDir   = mat3(view.viewMatrix) * wDir;
  vec2  tanA   = vDir.xy / -vDir.z;
  float rimTan = view.lensSize / view.lensDist;
  tanA *= (1.0 + view.lensDistortion * dot(tanA, tanA) / (rimTan * rimTan)) / view.lensZoom;
  return normalize(mat3(view.viewMatrixI) * vec3(tanA, -1.0));
}

void main()
{
#if USE_RAYQUERY
  if(any(greaterThanEqual(RT_LAUNCH_ID, RT_LAUNCH_SIZE)))
    return;
#endif

  // for writing debugging values to stats.debug etc.
  bool center      = RT_LAUNCH_ID == (RT_LAUNCH_SIZE / 2);
  vec2 pixelOffset = vec2(0.5);

#if USE_DLSS
  pixelOffset += view.jitter;
#endif

  ivec2 screen = ivec2(RT_LAUNCH_ID);
  vec2  uv     = (vec2(RT_LAUNCH_ID) + pixelOffset) / vec2(RT_LAUNCH_SIZE);


  vec2 d = uv * 2.0 - 1.0;


  vec4 origin    = view.viewMatrixI * vec4(0, 0, 0, 1);
  vec4 target    = normalize(view.projMatrixI * vec4(d.x, d.y, 1, 1));
  vec4 direction = normalize(view.viewMatrixI * vec4(target.xyz, 0));

  float tMin = view.nearPlane;
  float tMax = view.farPlane;

#if DEBUG_VISUALIZATION && ALLOW_SHADING
  vec2 differentialPixel = vec2(RT_LAUNCH_ID) + pixelOffset;
  rayHit.rayDifferentialX =
      computeViewRayDirection(((differentialPixel + vec2(1.0, 0.0)) / vec2(RT_LAUNCH_SIZE)) * 2.0 - 1.0);
  rayHit.rayDifferentialY =
      computeViewRayDirection(((differentialPixel + vec2(0.0, 1.0)) / vec2(RT_LAUNCH_SIZE)) * 2.0 - 1.0);
#endif

#if USE_DLSS
  float dlssSpecularHitDist = 0.0;
#endif

#if ALLOW_SHADING
  vec3  mirrorCenter = view.wMirrorBox.xyz;
  float mirrorSize   = view.wMirrorBox.w;
  vec3  mirrorHitPoint;
  vec3  mirrorNormal;
  bool  mirrorHit = false;
  float mirrorT;
  bool  fogApplied = false;

  if(view.opticalMode != OPTICAL_MODE_NONE)
  {
    if(view.opticalMode == OPTICAL_MODE_LENS)
    {
      // lens: a flat circular aperture facing the camera, lensDist ahead of the eye
      // along the view direction. See `lensZoomDirection`.
      vec3 lensCenter = origin.xyz + view.viewDir.xyz * view.lensDist;
      vec3 axis       = -view.viewDir.xyz;  // faces the camera
      mirrorT         = hitDisc(origin.xyz, direction.xyz, lensCenter, axis, view.lensSize);
      mirrorHit       = mirrorT > 0;
      if(mirrorHit)
      {
        mirrorHitPoint = origin.xyz + direction.xyz * mirrorT;
        mirrorNormal   = axis;

        tMax = mirrorT;
      }
    }
    else if(view.opticalMode == OPTICAL_MODE_SPHERE)
    {
      // convex spherical mirror: reflect off the outside of the sphere,
      // like a chrome ball — a demagnified wide-angle view of the scene
      mirrorT   = hitSphere(origin.xyz, direction.xyz, mirrorCenter, mirrorSize);
      mirrorHit = mirrorT > 0;
      if(mirrorHit)
      {
        mirrorHitPoint = origin.xyz + direction.xyz * mirrorT;
        mirrorNormal   = normalize(mirrorHitPoint - mirrorCenter);

        tMax = mirrorT;
      }
    }
    else
    {
      mirrorT   = hitBbox(origin.xyz, direction.xyz, mirrorCenter - mirrorSize, mirrorCenter + mirrorSize);
      mirrorHit = mirrorT > 0;
      if(mirrorHit)
      {
        mirrorHitPoint = origin.xyz + direction.xyz * mirrorT;
        mirrorNormal   = mirrorHitPoint - mirrorCenter;
        vec3 signs     = sign(mirrorNormal);
        mirrorNormal   = abs(mirrorNormal);
        mirrorNormal = vec3(equal(mirrorNormal, vec3(max(max(mirrorNormal.x, mirrorNormal.y), mirrorNormal.z)))) * signs;

        tMax = mirrorT;
      }
    }
  }
#endif

  // Ray cone for texture LOD and hit-based streaming LOD: width 0 at the eye, spread = per-pixel angle.
  // The hit-streaming registration scales this shared footprint into its geometry-error budget.
  rayHit.coneWidth  = 0.0;
  rayHit.coneSpread = view.pixelAngle;

  rtTrace(USE_FORCED_TWO_SIDED != 0 ? 0 : gl_RayFlagsCullBackFacingTrianglesEXT, origin.xyz, tMin, direction.xyz, tMax);

  // the lens is see-through, its depth must not occlude what is behind it in the hiz
  bool lensHit = false;
#if ALLOW_SHADING
  if(view.opticalMode != OPTICAL_MODE_NONE)
  {
    if(rayHit.hitT == 0 && mirrorHit)
    {
      bool isLens = view.opticalMode == OPTICAL_MODE_LENS;
      lensHit     = isLens;

      vec3 secondaryOrigin = mirrorHitPoint;
      vec3 secondaryDir    = reflect(direction.xyz, mirrorNormal);

      if(isLens)
      {
        secondaryDir = lensZoomDirection(direction.xyz);
      }

#if DEBUG_VISUALIZATION && ALLOW_SHADING
      if(isLens)
      {
        rayHit.rayDifferentialX = lensZoomDirection(rayHit.rayDifferentialX);
        rayHit.rayDifferentialY = lensZoomDirection(rayHit.rayDifferentialY);
      }
      else
      {
        rayHit.rayDifferentialX = reflect(rayHit.rayDifferentialX, mirrorNormal);
        rayHit.rayDifferentialY = reflect(rayHit.rayDifferentialY, mirrorNormal);
      }
#endif
      // the traveled distance accumulates into the width at the reflection point.
      // A planar mirror leaves the spread unchanged; a curved mirror changes it by
      // twice the normal variation: spread' = spread + 2 * curvature * width. The
      // convex sphere widens the cone, so the demagnified image demands coarser detail.
      float mirrorConeWidth = view.pixelAngle * mirrorT;
      rayHit.coneWidth      = mirrorConeWidth;
      if(isLens)
      {
        // the zoom narrows the spread by the magnification (ignoring the distortion)
        rayHit.coneSpread = view.pixelAngle / view.lensZoom;
      }
      else if(view.opticalMode == OPTICAL_MODE_SPHERE)
      {
        rayHit.coneSpread = view.pixelAngle + 2.0f * mirrorConeWidth / mirrorSize;
      }

      rtTrace(USE_FORCED_TWO_SIDED != 0 ? 0 : gl_RayFlagsCullBackFacingTrianglesEXT, secondaryOrigin, 0,
              secondaryDir, view.farPlane + mirrorT);

#if USE_DLSS
      // rayHit.hitT here is the reflected ray's own hit distance (0 = miss into sky), captured before
      // it's overwritten below with the box's own distance (used for depth/motion instead).
      dlssSpecularHitDist = (rayHit.hitT != 0) ? rayHit.hitT : 65504.0;
#endif

      rayHit.color.xyz *= max(0, dot(mirrorNormal, -direction.xyz)) * 0.5 + 0.5;
      rayHit.color.xyz = applyFog(rayHit.color.xyz, direction.xyz, mirrorT + rayHit.hitT);
      fogApplied       = true;
      rayHit.hitT      = mirrorT;
    }
  }
#endif

  bool  hitValid = rayHit.hitT != 0;
#if ALLOW_SHADING
  if(hitValid && !fogApplied)
  {
    rayHit.color.xyz = applyFog(rayHit.color.xyz, direction.xyz, rayHit.hitT);
  }
#endif
  vec3  hitPos   = origin.xyz + direction.xyz * (hitValid ? rayHit.hitT : view.farPlane * 0.99f);
  // misses at the far plane, which is 0 with reversed-Z
  float hitDepth = 0;

  if(hitValid && !lensHit)
  {
    vec4 screenPos = view.viewProjMatrix * vec4(hitPos, 1);
    hitDepth       = screenPos.z / screenPos.w;
  }

#if !USE_DEPTH_ONLY
  imageStore(imgColor, screen, vec4(rayHit.color.xyz, 1));
#endif
  imageStore(imgRaytracingDepth, screen, vec4(hitDepth, 0.f, 0.f, 0.f));

#if USE_DLSS
  vec2 motionVec = vec2(0);

  motionVec = calculateMotionVector(hitPos, view.viewProjMatrixPrev, view.viewProjMatrix, view.viewportf);

  imageStore(imgDlssMotion, screen, vec4(motionVec.x, motionVec.y, 0, 0));
  imageStore(imgDlssSpecHitDist, screen, vec4(dlssSpecularHitDist, 0, 0, 0));
#if ALLOW_SHADING
  imageStore(imgDlssAlbedo, screen, rayHit.dlssAlbedo);
  imageStore(imgDlssSpecAlbedo, screen, vec4(rayHit.dlssSpecular, 1.0f));
  imageStore(imgDlssNormalRoughness, screen, rayHit.dlssNormalRoughness);
#else
  // without shading there is no albedo/normal/roughness to report; the depth-only
  // variant does not even have a `color` in the payload.
#if !USE_DEPTH_ONLY
  imageStore(imgDlssAlbedo, screen, vec4(rayHit.color.xyz, 1));
#else
  imageStore(imgDlssAlbedo, screen, vec4(0));
#endif
  imageStore(imgDlssSpecAlbedo, screen, vec4(vec3(0), 1.0f));
  imageStore(imgDlssNormalRoughness, screen, vec4(0));
#endif
#endif
}
