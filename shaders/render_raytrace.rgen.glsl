/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"
#if USE_DLSS
#include "dlss_util.h"
#endif

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_RENDERMATERIALS_SSBO, set = 0) buffer renderMaterialsBuffer
{
  RenderMaterial materials[];
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba16f) uniform image2D imgColor;
#if USE_DLSS
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssAlbedo, rgba8)             uniform image2D imgDlssAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecAlbedo, rgba16f)       uniform image2D imgDlssSpecAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssNormalRoughness, rgba16f)  uniform image2D imgDlssNormalRoughness;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssMotion, rg16f)             uniform image2D imgDlssMotion;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecHitDist, r16f)         uniform image2D imgDlssSpecHitDist;
#endif

//////////////////////////////////////////////////////////////

#if DEBUG_VISUALIZATION && ALLOW_SHADING
// This ray-gen declares no bindless texture array; it only needs texturing.glsl's computeViewRayDirection
// for the wireframe-overlay differentials. Opt out of sampleBindless (which references that array).
#define TEXTURING_SKIP_BINDLESS 1
#include "texturing.glsl"
#endif

//////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT RayPayload rayHit;

#if ALLOW_SHADING
#include "sky_fog.glsl"
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
  // for writing debugging values to stats.debug etc.
  bool center      = gl_LaunchIDEXT.xy == (gl_LaunchSizeEXT.xy / 2);
  vec2 pixelOffset = vec2(0.5);

#if USE_DLSS
  pixelOffset += view.jitter;
#endif

  ivec2 screen = ivec2(gl_LaunchIDEXT.xy);
  vec2  uv     = (vec2(gl_LaunchIDEXT.xy) + pixelOffset) / vec2(gl_LaunchSizeEXT.xy);


  vec2 d = uv * 2.0 - 1.0;


  vec4 origin    = view.viewMatrixI * vec4(0, 0, 0, 1);
  vec4 target    = normalize(view.projMatrixI * vec4(d.x, d.y, 1, 1));
  vec4 direction = normalize(view.viewMatrixI * vec4(target.xyz, 0));

  float tMin = view.nearPlane;
  float tMax = view.farPlane;

#if DEBUG_VISUALIZATION && ALLOW_SHADING
  vec2 differentialPixel = vec2(gl_LaunchIDEXT.xy) + pixelOffset;
  rayHit.rayDifferentialX =
      computeViewRayDirection(((differentialPixel + vec2(1.0, 0.0)) / vec2(gl_LaunchSizeEXT.xy)) * 2.0 - 1.0);
  rayHit.rayDifferentialY =
      computeViewRayDirection(((differentialPixel + vec2(0.0, 1.0)) / vec2(gl_LaunchSizeEXT.xy)) * 2.0 - 1.0);
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

  traceRayEXT(accelerationStructureEXT(build.tlasAddress),
              USE_FORCED_TWO_SIDED != 0 ? 0 : gl_RayFlagsCullBackFacingTrianglesEXT, 0xff,
              0, 1,  // hit offset, hit stride
              0,  // miss offset
              origin.xyz, tMin, direction.xyz, tMax,
              0  // rayPayloadNV location qualifier
  );

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

      traceRayEXT(accelerationStructureEXT(build.tlasAddress),
                  USE_FORCED_TWO_SIDED != 0 ? 0 : gl_RayFlagsCullBackFacingTrianglesEXT, 0xff,
                  0, 1,  // hit offset, hit stride
                  0,  // miss offset
                  secondaryOrigin, 0, secondaryDir, view.farPlane + mirrorT,
                  0  // rayPayloadNV location qualifier
      );

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
