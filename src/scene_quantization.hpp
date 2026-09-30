/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#include <glm/glm.hpp>

namespace lodclusters {

// Drops mantissa bits so the values compress better.
// based on meshopt_quantizeFloat
// https://github.com/zeux/meshoptimizer/blob/master/src/quantization.cpp
inline float quantizeFloat(float value, uint32_t dropBits)
{
  union
  {
    uint32_t u32;
    float    f32;
  } un;

  un.f32      = value;
  uint32_t ui = un.u32;

  const int32_t mask  = (1 << (dropBits)) - 1;
  const int32_t round = (1 << (dropBits)) >> 1;

  int32_t  e   = ui & 0x7f800000;
  uint32_t rui = (ui + round) & ~mask;

  // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
  ui = e == 0x7f800000 ? ui : rui;

  // flush denormals to zero
  ui = e == 0 ? 0 : ui;

  un.u32 = ui;
  return un.f32;
}

inline glm::vec2 quantizeFloat(const glm::vec2& vec, uint32_t dropBits)
{
  glm::vec2 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  return res;
}

inline glm::vec3 quantizeFloat(const glm::vec3& vec, uint32_t dropBits)
{
  glm::vec3 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  res.z = quantizeFloat(vec.z, dropBits);
  return res;
}

inline glm::vec4 quantizeFloat(const glm::vec4& vec, uint32_t dropBits)
{
  glm::vec4 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  res.z = quantizeFloat(vec.z, dropBits);
  res.w = quantizeFloat(vec.w, dropBits);
  return res;
}

}  // namespace lodclusters
