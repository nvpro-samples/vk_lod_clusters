/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Basic path tracer. Inspired by vk_gltf_renderer, all shading happens
  here in the ray-generation shader: the closest-hit only reports the hit and
  this shader runs the multi-bounce loop.

  Lighting is the physical sky (evalPhysicalSky) with importance-sampled
  next-event estimation (samplePhysicalSky) and MIS against the BSDF. The BSDF
  reuses this project's simplified material model (computeShading / GGX eval
  helpers) with a stochastic diffuse/specular lobe split for indirect bounces.

  It is 1 sample-per-pixel and relies on DLSS Ray Reconstruction for temporal
  denoising (no accumulation buffer). The final HDR radiance is tone-mapped and
  an auto-exposure value is accumulated from a 16x16 pixel grid using float32
  atomics into the readback buffer.

*/

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float : require

#include "shaderio.h"
#if USE_DLSS
#include "dlss_util.h"  // SHADERIO_eDlss* binding offsets (used below), calculateMotionVector, EnvBRDFApprox2
#endif

//////////////////////////////////////////////////////////////

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

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba8)   uniform image2D imgColor;
#if USE_DLSS
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssAlbedo, rgba8)             uniform image2D imgDlssAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecAlbedo, rgba16f)       uniform image2D imgDlssSpecAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssNormalRoughness, rgba16f)  uniform image2D imgDlssNormalRoughness;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssMotion, rg16f)             uniform image2D imgDlssMotion;
#endif

layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];

//////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT PathRayPayload rayHit;
layout(location = 1) rayPayloadEXT float          rayHitAO;

//////////////////////////////////////////////////////////////

#define SUPPORTS_RT 1
// The ray-generation stage has no object-space transform builtins (gl_ObjectToWorldEXT); skip the
// hit-only helpers in render_shading.glsl.
#define SKIP_OBJECT_SPACE_HELPERS 1

#include "nvshaders/sky_functions.h.slang"
#include "attribute_encoding.h"
#include "texturing.glsl"
#include "render_shading.glsl"
#include "render_pathtrace_hit.glsl"

//////////////////////////////////////////////////////////////

float ptLuminance(vec3 c)
{
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float ptPowerHeuristic(float a, float b)
{
  float a2 = a * a;
  return a2 / max(a2 + b * b, 1e-8);
}

// Basic tone-map operators (from nvshaders/tonemap_functions.h.slang, inlined to avoid the
// tonemap_io enum header). tonemapFilmic/tonemapUncharted2 bake in sRGB; ACES stays linear.
vec3 ptTonemapFilmic(vec3 color)
{
  vec3 temp = max(vec3(0.0), color - vec3(0.004));
  return (temp * (vec3(6.2) * temp + vec3(0.5))) / (temp * (vec3(6.2) * temp + vec3(1.7)) + vec3(0.06));
}

vec3 ptTonemapACES(vec3 color)
{
  const mat3 ACESInputMat  = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
  const mat3 ACESOutputMat = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
  color    = ACESInputMat * color;
  vec3 a   = color * (color + vec3(0.0245786)) - vec3(0.000090537);
  vec3 b   = color * (vec3(0.983729) * color + vec3(0.4329510)) + vec3(0.238081);
  color    = a / b;
  color    = ACESOutputMat * color;
  return clamp(color, vec3(0.0), vec3(1.0));
}

vec3 ptTonemapUncharted2Impl(vec3 color)
{
  const float a = 0.15, b = 0.50, c = 0.10, d = 0.20, e = 0.02, f = 0.30;
  return ((color * (a * color + c * b) + d * e) / (color * (a * color + b) + d * f)) - e / f;
}

vec3 ptTonemapUncharted2(vec3 color)
{
  const float W             = 11.2;
  const float exposure_bias = 2.0;
  color                     = ptTonemapUncharted2Impl(color * exposure_bias);
  vec3 white_scale          = vec3(1.0) / ptTonemapUncharted2Impl(vec3(W));
  return pow(color * white_scale, vec3(1.0 / 2.2));
}

// probability of choosing the specular lobe, from the Fresnel-at-normal reflectance
float specularLobeProbability(vec3 diffuseAlbedo, vec3 f0)
{
  float lumD = ptLuminance(diffuseAlbedo);
  float lumS = ptLuminance(f0);
  return clamp(lumS / max(lumS + lumD, 1e-4), 0.1, 0.9);
}

// combined pdf of the stochastic diffuse+specular BSDF sampler for direction L
float bsdfPdf(vec3 N, vec3 V, vec3 L, float alphaRoughness, float pSpec)
{
  float NdotL = dot(N, L);
  if(NdotL <= 0.0)
    return 0.0;

  vec3  H     = normalize(V + L);
  float NdotH = clamp(dot(N, H), 0.0, 1.0);
  float VdotH = clamp(dot(V, H), 0.0, 1.0);

  float pdfDiff = NdotL * (1.0 / M_PI);
  float pdfSpec = D_GGX(NdotH, alphaRoughness) * NdotH / (4.0 * max(VdotH, 1e-4));

  return (1.0 - pSpec) * pdfDiff + pSpec * pdfSpec;
}

// GGX (NDF) half-vector importance sample in the local frame around N
vec3 sampleGgxDirection(vec3 N, vec3 tx, vec3 ty, vec3 rayDir, float alphaRoughness, float u1, float u2)
{
  float phi     = 2.0 * M_PI * u1;
  float cosTh   = sqrt((1.0 - u2) / (1.0 + (alphaRoughness * alphaRoughness - 1.0) * u2));
  float sinTh   = sqrt(max(0.0, 1.0 - cosTh * cosTh));
  vec3  Hlocal  = vec3(sinTh * cos(phi), sinTh * sin(phi), cosTh);
  vec3  H       = Hlocal.x * tx + Hlocal.y * ty + Hlocal.z * N;
  return reflect(rayDir, H);
}

// cosine-weighted hemisphere sample around N
vec3 sampleCosineDirection(vec3 N, vec3 tx, vec3 ty, float u1, float u2)
{
  float r1 = 2.0 * M_PI * u1;
  float sq = sqrt(1.0 - u2);
  vec3  d  = vec3(cos(r1) * sq, sin(r1) * sq, sqrt(u2));
  return d.x * tx + d.y * ty + d.z * N;
}

// Ray/AABB slab test used by the hacky reflective box (copied from render_raytrace.rgen.glsl).
// Returns the entry distance t0 if the ray enters the box ahead of the origin, otherwise -1.
float hitBbox(vec3 rayPos, vec3 rayDir, vec3 bboxMin, vec3 bboxMax)
{
  vec3  invDir = 1.0 / rayDir;
  vec3  tbot   = invDir * (bboxMin - rayPos);
  vec3  ttop   = invDir * (bboxMax - rayPos);
  vec3  tmin   = min(ttop, tbot);
  vec3  tmax   = max(ttop, tbot);
  float t0     = max(tmin.x, max(tmin.y, tmin.z));
  float t1     = min(tmax.x, min(tmax.y, tmax.z));
  return t1 > max(t0, 0.0) ? t0 : -1.0;
}

// Shadow / visibility ray for the path tracer. Unlike the shared traceShadowRay() (a separate float
// payload at location 1), this reuses PathRayPayload so the alpha-mask any-hit reads the SAME ray cone as
// the primary/indirect rays. We keep closest-hit skipped: hitT is the sentinel - set to occluded here, and
// the miss shader (index 0) flips it to < 0 when the light is reached.
float ptTraceShadowRay(vec3 wPos, vec3 wNormal, vec3 wDir, float coneWidth, float coneSpread)
{
  uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
#if !HAS_ALPHA_TEST
  flags |= gl_RayFlagsOpaqueEXT;
#endif
  rayHit.hitT       = 1.0;  // occluded sentinel; the index-0 miss shader sets it to -1 (unoccluded)
  rayHit.coneWidth  = coneWidth;
  rayHit.coneSpread = coneSpread;
  traceRayEXT(asScene, flags, 0xFF, 0, 1, 0, offsetRay(wPos, wDir, wNormal), 0.001, wDir, 1e7, 0);
  return rayHit.hitT < 0.0 ? 1.0 : 0.0;
}

//////////////////////////////////////////////////////////////

void main()
{
  ivec2 screen      = ivec2(gl_LaunchIDEXT.xy);
  vec2  pixelOffset = vec2(0.5);
#if USE_DLSS
  pixelOffset += view.jitter;
#endif

  vec2 uv = (vec2(gl_LaunchIDEXT.xy) + pixelOffset) / vec2(gl_LaunchSizeEXT.xy);
  vec2 d  = uv * 2.0 - 1.0;

  vec4 origin    = view.viewMatrixI * vec4(0, 0, 0, 1);
  vec4 target    = normalize(view.projMatrixI * vec4(d.x, d.y, 1, 1));
  vec4 direction = normalize(view.viewMatrixI * vec4(target.xyz, 0));

  vec3 rayOrigin = origin.xyz;
  vec3 rayDir    = direction.xyz;

  uint  seed        = xxhash32(uvec3(gl_LaunchIDEXT.xy, view.frame));
  vec3  radiance    = vec3(0);
  vec3  throughput  = vec3(1);
  float lastBsdfPdf = 0.0;

  // Ray-cone texture LOD (Akenine-Moeller et al., "Improved Ray Cones", JCGT 2021): the ray footprint is
  // a world-space cone described by a width and a spread angle. Width starts at 0 at the eye; the spread
  // starts as the per-pixel angle and widens at each rough/diffuse scatter (see the expansion below).
  float coneWidth  = 0.0;
  float coneSpread = view.pixelAngle;

  // first-hit outputs
  bool hitValid            = false;
  vec3 firstHitPos         = rayOrigin + rayDir * (view.farPlane * 0.99f);
  vec4 dlssAlbedo          = vec4(0);
  vec4 dlssNormalRoughness = vec4(0);
  vec3 dlssSpecular        = vec3(0);

  float tMin = view.nearPlane;
  float tMax = view.farPlane;

  uint rayFlags = USE_FORCED_TWO_SIDED != 0 ? 0 : gl_RayFlagsCullBackFacingTrianglesEXT;
 
  // do less bounces in visualization modes, given the flat detail.
  int maxBounces = view.visualize == VISUALIZE_SHADED
               ? max(view.pathtraceNumBounces, 1)
               : min(max(view.pathtraceNumBounces, 1), 2);

  // Hacky reflective box (ported from the raster / ray tracer, see render_raytrace.rgen.glsl): only the
  // primary camera ray reflects off an axis-aligned box; path-traced bounces ignore it. When the box's
  // front face is reached with no geometry in front of it, the primary ray is redirected along the
  // mirror reflection and normal path tracing continues from there, so the reflection gets full GI.
  if(view.useMirrorBox != 0)
  {
    vec3  mirrorCenter = view.wMirrorBox.xyz;
    float mirrorSize   = view.wMirrorBox.w;
    float mirrorT      = hitBbox(rayOrigin, rayDir, mirrorCenter - mirrorSize, mirrorCenter + mirrorSize);
    if(mirrorT > 0.0)
    {
      // Probe the camera->face segment; only reflect when nothing occludes the box (matches the RT's
      // `hitT == 0` check). The probe hit is discarded - the loop below re-traces for the actual shading.
      rayHit.hitT       = -1.0;
      rayHit.coneWidth  = coneWidth;
      rayHit.coneSpread = coneSpread;
      traceRayEXT(asScene, rayFlags, 0xff, 0, 1, 0, rayOrigin, tMin, rayDir, mirrorT, 0);
      if(rayHit.hitT < 0.0)
      {
        // Axis-aligned face normal of the box: the dominant axis of (hitPoint - center).
        vec3 mirrorHitPoint = rayOrigin + rayDir * mirrorT;
        vec3 mirrorNormal   = mirrorHitPoint - mirrorCenter;
        vec3 signs          = sign(mirrorNormal);
        mirrorNormal        = abs(mirrorNormal);
        mirrorNormal = vec3(equal(mirrorNormal, vec3(max(max(mirrorNormal.x, mirrorNormal.y), mirrorNormal.z)))) * signs;

        throughput *= max(0.0, dot(mirrorNormal, -rayDir)) * 0.5 + 0.5;  // edge darkening, same as the RT
        rayOrigin   = mirrorHitPoint;
        rayDir      = reflect(rayDir, mirrorNormal);
        tMin        = 1e-4;
        firstHitPos = rayOrigin + rayDir * (view.farPlane * 0.99f);  // miss fallback along the reflected ray
      }
    }
  }

  for(int bounce = 0; bounce < maxBounces; bounce++)
  {
    // hand the current ray cone (at the ray origin) to the any-hit / hit shaders for this trace
    rayHit.hitT       = -1.0;
    rayHit.coneWidth  = coneWidth;
    rayHit.coneSpread = coneSpread;
    traceRayEXT(asScene, rayFlags, 0xff, 0, 1, 0, rayOrigin, tMin, rayDir, tMax, 0);

    // ---------------------------------------------------------
    // Environment (physical sky) on miss
    // ---------------------------------------------------------
    if(rayHit.hitT < 0.0)
    {
      vec3  envColor = evalPhysicalSky(view.skyPhysical, rayDir);
      float misW     = 1.0;
      if(bounce > 0)
      {
        float envPdf = samplePhysicalSkyPDF(view.skyPhysical, rayDir);
        misW         = ptPowerHeuristic(lastBsdfPdf, envPdf);
      }
      radiance += throughput * misW * envColor;
      break;
    }

    // ---------------------------------------------------------
    // Reconstruct hit + material
    // ---------------------------------------------------------
    PathHit hit = getHitAttributes(rayHit.instanceID, rayHit.clusterID, rayHit.triangleID, rayHit.bary, rayOrigin, rayDir, rayHit.hitT);

    // grow the ray cone from the previous point to this hit, then derive the texture LOD for this surface
    coneWidth += coneSpread * rayHit.hitT;
    TexLOD texLod = makeConeTexLOD(coneWidth, hit.texelDensity, abs(dot(rayDir, hit.wGeoNormal)));

    bool firstHit = (bounce == 0);

    vec3            N = hit.wNormal;
    ShadingMaterial mat;
    if(view.visualize == VISUALIZE_SHADED)
    {
      mat = loadMaterial(hit.materialID, hit.oTexCoord, N, hit.wTangent, texLod);
    }
    else
    {
      vec3  viz  = visualizeColor(hit.visData, rayHit.instanceID);
      float vl   = dot(viz, vec3(0.2126, 0.7152, 0.0722));
      mat.albedo = clamp(mix(vec3(vl), viz, 1.4), 0.0, 1.0);   // 1.4 = saturation boost
      mat.roughness = 0.4;
      mat.metallic  = 0.0;
      mat.emissive  = vec3(0);
      mat.occlusion = 1.0;
    }

    // For low-tessellated geometry the interpolated (or normal-mapped) shading normal can tilt far
    // enough that the mirror-reflected view direction points below the geometric surface, producing
    // black specular spots and bounce rays that immediately self-intersect. Snap the shading normal
    // back to the geometric normal in that case.
    if(dot(reflect(rayDir, N), hit.wGeoNormal) < 0.0)
      N = hit.wGeoNormal;

    vec3  V     = -rayDir;
    float NdotV = clamp(dot(N, V), 0.0, 1.0);

    if(firstHit)
    {
      hitValid    = true;
      firstHitPos = hit.wPos;

#if USE_DLSS
      dlssAlbedo          = vec4(mat.albedo, 0);
      dlssNormalRoughness = vec4(N, mat.roughness);
      dlssSpecular        = EnvBRDFApprox2(vec3(1), mat.roughness, dot(N, V));
#endif

      // mouse picking readback
      if(gl_LaunchIDEXT.xy == view.mousePosition)
      {
        vec4  projected            = view.viewProjMatrix * vec4(hit.wPos, 1.f);
        float depth                = projected.z / projected.w;
        readback.clusterTriangleId = packPickingValue((rayHit.clusterID << 8) | rayHit.triangleID, depth);
        readback.instanceId        = packPickingValue(rayHit.instanceID, depth);
        readback.materialId        = packPickingValue(hit.materialID, depth);
      }
    }

    // emissive
    radiance += throughput * mat.emissive;

    // Camera flashlight: an additional light on top of the sky, applied on the first (camera-visible)
    // surface only - a port of the raster/RT light mixer, but additive rather than blended.
    // view.lightMixer scales it relative to the sky (1 == flashVsSky of sky brightness). The
    // camera-visible point is unoccluded by construction, so no shadow ray is needed.
    if(firstHit && view.lightMixer > 0.0)
    {
      const float flashVsSky = 0.5;  // flashlight capped at 50% of the sky brightness
      vec3  flashDir      = normalize(view.wLightPos.xyz - hit.wPos);
      float skyBrightness = ptLuminance(evalPhysicalSky(view.skyPhysical, normalize(view.wUpDir.xyz)));
      radiance += throughput * (view.lightMixer * skyBrightness * flashVsSky) * computeShading(mat, N, flashDir, V, NdotV);
    }

    // material lobes
    float perceptualRoughness = max(mat.roughness, 0.04f);
    float alphaRoughness      = perceptualRoughness * perceptualRoughness;
    vec3  diffuseAlbedo       = mat.albedo * (1.0 - mat.metallic);
    vec3  f0                  = mix(vec3(0.04), mat.albedo, mat.metallic);
    float pSpec               = specularLobeProbability(diffuseAlbedo, f0);

    vec3 tx, ty;
    computeDefaultBasis(N, tx, ty);

    // ---------------------------------------------------------
    // Next-event estimation: importance-sample the physical sky
    // ---------------------------------------------------------
    {
      SkySamplingResult sky = samplePhysicalSky(view.skyPhysical, vec2(rand(seed), rand(seed)));
      if(sky.pdf > 0.0 && dot(N, sky.direction) > 0.0)
      {
        vec3  f    = computeShading(mat, N, sky.direction, V, NdotV);  // BRDF * NdotL
        float bp   = bsdfPdf(N, V, sky.direction, alphaRoughness, pSpec);
        float misW = ptPowerHeuristic(sky.pdf, bp);
        if(misW > 0.0 && (f.x + f.y + f.z) > 0.0)
        {
          float visibility = ptTraceShadowRay(hit.wShadowPos, hit.wGeoNormal, sky.direction, coneWidth, coneSpread);
          radiance += throughput * (sky.radiance / sky.pdf) * f * misW * visibility;
        }
      }
    }

    // ---------------------------------------------------------
    // BSDF sampling: choose the next direction (diffuse or specular lobe)
    // ---------------------------------------------------------
    vec3 L;
    if(rand(seed) < pSpec)
    {
      L = sampleGgxDirection(N, tx, ty, rayDir, alphaRoughness, rand(seed), rand(seed));
    }
    else
    {
      L = sampleCosineDirection(N, tx, ty, rand(seed), rand(seed));
    }

    if(dot(N, L) <= 0.0)
      break;

    float bp = bsdfPdf(N, V, L, alphaRoughness, pSpec);
    if(bp <= 0.0)
      break;

    vec3 f = computeShading(mat, N, L, V, NdotV);  // BRDF * NdotL
    throughput *= f / bp;
    lastBsdfPdf = bp;

    // Ray-cone spread expansion (RTXPT / JCGT 2021, §3): widen the cone by the scattered lobe's angular
    // size, derived from the BSDF sample pdf. Interpreting the lobe as a uniform spherical cap of solid
    // angle 1/pdf gives plane angle 2*acos(1 - (1/pdf)/(2*pi)); the 0.3 factor is a conservative
    // underestimate (1-spp + DLSS denoise handle the rest, and it avoids overblur). Tight glossy lobes
    // (high pdf) add almost nothing so reflections stay sharp; broad diffuse lobes (low pdf) widen a lot,
    // pushing indirect texture reads to coarse mips which suppresses noise and aliasing.
    coneSpread = min(coneSpread + 0.3 * 2.0 * acos(clamp(1.0 - (1.0 / bp) / (2.0 * M_PI), -1.0, 1.0)), 2.0 * M_PI);

    // continue the path
    rayOrigin = offsetRay(hit.wPos, L, hit.wGeoNormal);
    rayDir    = L;
    tMin      = 1e-4;
    tMax      = view.farPlane;

    // Russian roulette
    if(bounce >= 2)
    {
      float p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 1.0);
      if(rand(seed) > p)
        break;
      throughput /= p;
    }

    if(max(throughput.x, max(throughput.y, throughput.z)) <= 0.0)
      break;
  }

  // ---------------------------------------------------------
  // Firefly clamp
  // ---------------------------------------------------------
  if(view.pathtraceFireflyClamp > 0.0)
  {
    float lum = ptLuminance(radiance);
    if(lum > view.pathtraceFireflyClamp)
      radiance *= view.pathtraceFireflyClamp / lum;
  }

  // ---------------------------------------------------------
  // Auto-exposure accumulation (pre-tonemap), one pixel per 16x16 tile
  // ---------------------------------------------------------
  // Accumulate log-luminance for a geometric-mean auto-exposure (the CPU converts back with exp2).
  if((gl_LaunchIDEXT.x & 15u) == 0u && (gl_LaunchIDEXT.y & 15u) == 0u)
  {
    atomicAdd(readback.autoExposureLumaSum, log2(max(ptLuminance(radiance), 1e-3)));
    atomicAdd(readback.autoExposureSampleCount, 1u);
  }

  // ---------------------------------------------------------
  // Tone map (basic) + store
  // ---------------------------------------------------------
  vec3 mapped = radiance * view.pathtraceExposure;
  if(view.pathtraceTonemap == 1)
  {
    mapped = toSrgb(ptTonemapACES(mapped));
  }
  else if(view.pathtraceTonemap == 2)
  {
    mapped = ptTonemapUncharted2(mapped);
  }
  else if(view.pathtraceTonemap == 3)
  {
    // Clip: no tone curve, just clamp + sRGB (matches the raster / RT shading look)
    mapped = toSrgb(clamp(mapped, vec3(0.0), vec3(1.0)));
  }
  else
  {
    mapped = ptTonemapFilmic(mapped);
  }

  float hitDepth = 1.0;
  if(hitValid)
  {
    vec4 screenPos = view.viewProjMatrix * vec4(firstHitPos, 1);
    hitDepth       = screenPos.z / screenPos.w;
  }

  imageStore(imgColor, screen, vec4(mapped, 1));
  imageStore(imgRaytracingDepth, screen, vec4(hitDepth, 0.f, 0.f, 0.f));

#if USE_DLSS
  vec2 motionVec = calculateMotionVector(firstHitPos, view.viewProjMatrixPrev, view.viewProjMatrix, view.viewportf);
  imageStore(imgDlssMotion, screen, vec4(motionVec.x, motionVec.y, 0, 0));

  if(hitValid)
  {
    // Shaded and visualization modes both feed the base-color/normal/roughness guides captured at the
    // first hit (for viz the base color is the palette color), so DLSS demodulates + denoises properly.
    imageStore(imgDlssAlbedo, screen, dlssAlbedo);
    imageStore(imgDlssSpecAlbedo, screen, vec4(dlssSpecular, 1.0f));
    imageStore(imgDlssNormalRoughness, screen, dlssNormalRoughness);
  }
  else
  {
    imageStore(imgDlssAlbedo, screen, vec4(mapped, 1));
    imageStore(imgDlssSpecAlbedo, screen, vec4(vec3(0), 1.0f));
    imageStore(imgDlssNormalRoughness, screen, vec4(0));
  }
#endif
}
