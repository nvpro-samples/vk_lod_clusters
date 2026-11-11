/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ENVIRONMENT_GLSL
#define ENVIRONMENT_GLSL

// The environment: the physical sky, or an HDRI loaded by `nvvk::HdrIbl`. The HDRI is a
// y-up lat-long whose alpha holds the per-steradian pdf of its alias table.
// Needs nvshaders/sky_functions.h.slang.
//
// The HDRI lookup and sampling are a GLSL port of nvpro_core2's Slang-only
// https://github.com/nvpro-samples/nvpro_core2/blob/main/nvshaders/hdr_env_sampling.h.slang

layout(set = 0, binding = BINDINGS_ENV_HDRI_TEX) uniform sampler2D texEnvHdri;

#if ENV_HDRI_SAMPLING
// nvpro_core2's shaderio::EnvAccel
struct EnvHdriAccel
{
  uint  alias;
  float q;
};
layout(scalar, set = 0, binding = BINDINGS_ENV_HDRI_ACCEL_SSBO) readonly buffer envHdriAccelBuffer
{
  EnvHdriAccel envHdriAccel[];
};
#endif

// world to the HDRI's y-up frame, turned around its up axis, then tilted
vec3 envHdriFromWorld(EnvHdri hdri, SkyPhysicalParameters sky, vec3 dir)
{
  vec3  d = sky.yIsUp == 1 ? dir : vec3(dir.x, dir.z, -dir.y);
  float c = cos(hdri.rotation);
  float s = sin(hdri.rotation);
  d       = vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
  c       = cos(hdri.elevation);
  s       = sin(hdri.elevation);
  return vec3(d.x, c * d.y - s * d.z, s * d.y + c * d.z);
}

vec3 envHdriToWorld(EnvHdri hdri, SkyPhysicalParameters sky, vec3 d)
{
  float c = cos(hdri.elevation);
  float s = sin(hdri.elevation);
  d       = vec3(d.x, c * d.y + s * d.z, -s * d.y + c * d.z);
  c       = cos(hdri.rotation);
  s       = sin(hdri.rotation);
  d       = vec3(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
  return sky.yIsUp == 1 ? d : vec3(d.x, -d.z, d.y);
}

// matches nvpro_core2's environmentSampleCoords
vec2 envHdriUv(vec3 d)
{
  float u = (atan(d.z, d.x) + M_PI) * (0.5 * M_1_PI);
  float v = acos(clamp(d.y, -1.0, 1.0)) * M_1_PI;
  return vec2(u, v);
}

// radiance in rgb, the pdf of sampleEnvironment in alpha (HDRI only)
vec4 evalEnvironmentLod(SkyPhysicalParameters sky, EnvHdri hdri, vec3 dir, float lod)
{
  if(hdri.enabled == 0)
    return vec4(evalPhysicalSky(sky, dir), 0);
  vec4 texel = textureLod(texEnvHdri, envHdriUv(envHdriFromWorld(hdri, sky, dir)), lod);
  return vec4(texel.xyz * hdri.intensity, texel.w);
}

vec3 evalEnvironment(SkyPhysicalParameters sky, EnvHdri hdri, vec3 dir)
{
  return evalEnvironmentLod(sky, hdri, dir, 0.0).xyz;
}

#if ENV_HDRI_SAMPLING
// HDRI importance sampling, a port of nvpro_core2's environmentSample (alias method,
// https://arxiv.org/pdf/1901.05423.pdf section 2.6): a texel, then a direction uniform in its
// solid angle. Returns radiance and the per-steradian pdf.
vec4 sampleEnvironmentHdri(SkyPhysicalParameters sky, EnvHdri hdri, vec3 xi, out vec3 dir)
{
  uint size = hdri.width * hdri.height;
  uint idx  = min(uint(xi.x * float(size)), size - 1u);

  EnvHdriAccel entry = envHdriAccel[idx];
  if(xi.y < entry.q)
  {
    xi.y /= entry.q;
  }
  else
  {
    idx  = entry.alias;
    xi.y = (xi.y - entry.q) / (1.0 - entry.q);
  }

  uint  px       = idx % hdri.width;
  uint  py       = idx / hdri.width;
  float u        = (float(px) + xi.y) / float(hdri.width);
  float phi      = u * (2.0 * M_PI) - M_PI;
  float stepTheta = M_PI / float(hdri.height);
  float theta0   = float(py) * stepTheta;
  float cosTheta = mix(cos(theta0), cos(theta0 + stepTheta), xi.z);
  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

  vec3 local = vec3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
  dir        = envHdriToWorld(hdri, sky, local);

  vec4 texel = textureLod(texEnvHdri, vec2(u, acos(cosTheta) * M_1_PI), 0.0);
  return vec4(texel.xyz * hdri.intensity, texel.w);
}
#endif

#endif
