/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SKY_FOG_GLSL
#define SKY_FOG_GLSL

// physical sky lighting of raster and ray tracing, see `sky_bake.comp.glsl`
layout(set = 0, binding = BINDINGS_SKY_ENV_TEX) uniform samplerCube texSkyEnv;
layout(set = 0, binding = BINDINGS_SKY_IRRADIANCE_TEX) uniform samplerCube texSkyIrradiance;

// Haze between the eye and a surface `distance` away along `wDir`. Fades into the sky behind it,
// so distant geometry blends into the background. The baked sky has no sun disk, so a fogged
// surface in front of the sun does not show it.
vec3 applyFog(vec3 color, vec3 wDir, float distance)
{
  if(view.fogDensity <= 0.0)
    return color;

  vec3  skyFog  = textureLod(texSkyEnv, wDir, 1.0).xyz;
  float opacity = 1.0 - exp(-view.fogDensity * distance);
  return mix(color, skyFog, opacity);
}

#endif
