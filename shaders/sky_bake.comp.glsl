/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  Bakes the physical sky into the lighting raster and ray tracing use for the
  shaded and grey visualization, so they match the path tracer.

  The sun cone is split off the same way the path tracer's sky sampler does
  (half-angle 1.5 * 0.00465 * sunDiskScale) and handled as a directional light.
  The remaining sky dome is integrated by `push.mode`:

  - `SKY_BAKE_ENV`:        one mip of the GGX prefiltered environment, roughness
                           `mip / (SKY_ENV_MIPS - 1)`, one thread per texel.
  - `SKY_BAKE_IRRADIANCE`: cosine weighted average radiance, one thread per texel.
                           Times albedo it is the lambertian response.
  - `SKY_BAKE_SUN`:        a single workgroup integrates the sun cone into
                           `FrameConstants::skyLighting`.

  The sky is analytic, so it is evaluated directly instead of sampling a
  higher-resolution source.

  Follows nvpro_core2's environment prefiltering
  https://github.com/nvpro-samples/nvpro_core2/blob/main/nvshaders/hdr_prefilter_glossy.slang
  https://github.com/nvpro-samples/nvpro_core2/blob/main/nvshaders/hdr_prefilter_diffuse.slang
  which uses the split-sum approximation of [Karis 2013, "Real Shading in Unreal Engine 4"].
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"
#include "nvshaders/sky_functions.h.slang"

layout(local_size_x = SKY_BAKE_WORKGROUP, local_size_y = SKY_BAKE_WORKGROUP) in;

layout(scalar, push_constant) uniform pushData
{
  SkyBakePush push;
};

layout(binding = BINDINGS_SKYBAKE_ENV, set = 0, rgba16f) uniform writeonly image2DArray imgEnv[SKY_ENV_MIPS];
layout(binding = BINDINGS_SKYBAKE_IRRADIANCE, set = 0, rgba16f) uniform writeonly image2DArray imgIrradiance;
layout(scalar, binding = BINDINGS_SKYBAKE_FRAME_SSBO, set = 0) buffer frameConstantsBuffer
{
  FrameConstants view;
};

const uint SAMPLES = 1024;

float sunConeCos()
{
  return cos(1.5 * 0.00465 * push.sky.sunDiskScale);
}

vec3 evalSkyDome(vec3 dir)
{
  if(push.sky.sunDiskScale > 1e-5 && dot(dir, push.sky.sunDirection) >= sunConeCos())
    return vec3(0);
  return evalPhysicalSky(push.sky, dir);
}

vec2 hammersley(uint i, uint n)
{
  return vec2((float(i) + 0.5) / float(n), float(bitfieldReverse(i)) * 2.3283064365386963e-10);
}

void basis(vec3 n, out vec3 t, out vec3 b)
{
  vec3 up = abs(n.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
  t       = normalize(cross(up, n));
  b       = cross(n, t);
}

// Vulkan cube face order +X -X +Y -Y +Z -Z
vec3 cubeDirection(uvec3 texel, uint size)
{
  vec2 uv = (vec2(texel.xy) + 0.5) / float(size) * 2.0 - 1.0;
  switch(texel.z)
  {
    case 0: return normalize(vec3(1, -uv.y, -uv.x));
    case 1: return normalize(vec3(-1, -uv.y, uv.x));
    case 2: return normalize(vec3(uv.x, 1, uv.y));
    case 3: return normalize(vec3(uv.x, -1, -uv.y));
    case 4: return normalize(vec3(uv.x, -uv.y, 1));
    default: return normalize(vec3(-uv.x, -uv.y, -1));
  }
}

shared vec3 s_sum[SKY_BAKE_WORKGROUP * SKY_BAKE_WORKGROUP];

void main()
{
  if(push.mode == SKY_BAKE_SUN)
  {
    // uniform samples over the sun cone, times its solid angle
    vec3 t, b;
    basis(push.sky.sunDirection, t, b);
    float zMin = sunConeCos();

    uint thread = gl_LocalInvocationIndex;
    uint count  = SKY_BAKE_WORKGROUP * SKY_BAKE_WORKGROUP;
    vec3 sum    = vec3(0);
    for(uint i = thread; i < SAMPLES; i += count)
    {
      vec3 local = sampleSphericalCap(zMin, hammersley(i, SAMPLES));
      sum += evalPhysicalSky(push.sky, local.x * t + local.y * b + local.z * push.sky.sunDirection);
    }
    s_sum[thread] = sum;
    barrier();

    if(thread == 0)
    {
      for(uint i = 1; i < count; i++)
        sum += s_sum[i];

      float angle      = 1.5 * 0.00465 * push.sky.sunDiskScale;
      float solidAngle = angle < 0.001 ? M_PI * angle * angle : 2.0 * M_PI * (1.0 - zMin);

      view.skyLighting.sunIrradiance = push.sky.sunDiskScale > 1e-5 ? sum * (solidAngle / float(SAMPLES)) : vec3(0);
      view.skyLighting.upLuminance   = dot(evalPhysicalSky(push.sky, push.upDir), vec3(0.2126, 0.7152, 0.0722));
    }
    return;
  }

  uint size = push.mode == SKY_BAKE_ENV ? (SKY_ENV_SIZE >> push.mip) : SKY_IRRADIANCE_SIZE;
  uvec3 texel = gl_GlobalInvocationID;
  if(texel.x >= size || texel.y >= size)
    return;

  vec3 N = cubeDirection(texel, size);
  vec3 t, b;
  basis(N, t, b);

  vec3 result = vec3(0);
  if(push.mode == SKY_BAKE_IRRADIANCE)
  {
    for(uint i = 0; i < SAMPLES; i++)
    {
      vec2  xi  = hammersley(i, SAMPLES);
      float r   = sqrt(xi.x);
      float phi = 2.0 * M_PI * xi.y;
      vec3  L   = r * cos(phi) * t + r * sin(phi) * b + sqrt(max(0.0, 1.0 - xi.x)) * N;
      result += evalSkyDome(L);
    }
    result /= float(SAMPLES);
    imageStore(imgIrradiance, ivec3(texel), vec4(result, 1));
  }
  else if(push.mip == 0)
  {
    imageStore(imgEnv[0], ivec3(texel), vec4(evalSkyDome(N), 1));
  }
  else
  {
    // split sum: N = V = R, GGX importance sampled and NdotL weighted
    float roughness = float(push.mip) / float(SKY_ENV_MIPS - 1);
    float alpha     = roughness * roughness;
    float weight    = 0;
    for(uint i = 0; i < SAMPLES; i++)
    {
      vec2  xi    = hammersley(i, SAMPLES);
      float phi   = 2.0 * M_PI * xi.x;
      float cosTh = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
      float sinTh = sqrt(max(0.0, 1.0 - cosTh * cosTh));
      vec3  H     = sinTh * cos(phi) * t + sinTh * sin(phi) * b + cosTh * N;
      vec3  L     = reflect(-N, H);
      float NdotL = dot(N, L);
      if(NdotL > 0.0)
      {
        result += evalSkyDome(L) * NdotL;
        weight += NdotL;
      }
    }
    imageStore(imgEnv[push.mip], ivec3(texel), vec4(result / max(weight, 1e-4), 1));
  }
}
