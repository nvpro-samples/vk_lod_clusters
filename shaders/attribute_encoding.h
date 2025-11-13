/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _ATTRIBUTE_ENCODING_H_
#define _ATTRIBUTE_ENCODING_H_

#define ATTRENC_PI float(3.14159265358979323846264338327950288)

// normal total storage must be even number
#define ATTRENC_NORMAL_BITS 22

// tangent total storage is equal
// angle uses -1 bits
#define ATTRENC_TANGENT_BITS 10

#ifdef __cplusplus

namespace shaderio {

#define ATTRENC_INLINE inline
#define ATTRENC_OUT(a) a&
#define ATTRENC_ATAN2F atan2f
#define ATTRENC_INLINE inline
#define ATTRENC_FLOOR glm::floor
#define ATTRENC_CLAMP glm::clamp
#define ATTRENC_ABS glm::abs

static_assert(ATTRENC_NORMAL_BITS % 2 == 0);

#else

#define ATTRENC_INLINE
#define ATTRENC_FLOOR floor
#define ATTRENC_CLAMP clamp
#define ATTRENC_ABS abs
#define ATTRENC_INLINE
#define ATTRENC_OUT(a) out a
#define ATTRENC_ATAN2F atan

#endif


// several oct functions from http://jcgt.org/published/0003/02/01/paper.pdf
ATTRENC_INLINE vec2 oct_signNotZero(vec2 v)
{
  return vec2((v.x >= 0.0f) ? +1.0f : -1.0f, (v.y >= 0.0f) ? +1.0 : -1.0f);
}
ATTRENC_INLINE vec3 oct_to_vec(vec2 e)
{
  vec3 v = vec3(e.x, e.y, 1.0f - ATTRENC_ABS(e.x) - ATTRENC_ABS(e.y));
  if(v.z < 0.0f)
  {
    vec2 os = oct_signNotZero(e);
    v.x     = (1.0f - ATTRENC_ABS(e.y)) * os.x;
    v.y     = (1.0f - ATTRENC_ABS(e.x)) * os.y;
  }
  return normalize(v);
}

ATTRENC_INLINE vec2 vec_to_oct(vec3 v)
{
  // Project the sphere onto the octahedron, and then onto the xy plane
  vec2 p = vec2(v.x, v.y) * (1.0f / (ATTRENC_ABS(v.x) + ATTRENC_ABS(v.y) + ATTRENC_ABS(v.z)));
  // Reflect the folds of the lower hemisphere over the diagonals
  return (v.z <= 0.0f) ? (vec2(1.0f - ATTRENC_ABS(p.y), 1.0f - ATTRENC_ABS(p.x)) * oct_signNotZero(p)) : p;
}

ATTRENC_INLINE vec2 vec_to_oct_precise(vec3 v, int bits)
{
  vec2 s = vec_to_oct(v);  // Remap to the square
                           // Each snorm's max value interpreted as an integer,
                           // e.g., 127.0 for snorm8
  float M = float(1 << ((bits)-1)) - 1.0f;
  // Remap components to snorm(n/2) precision...with floor instead
  // of round (see equation 1)
  s                        = ATTRENC_FLOOR(ATTRENC_CLAMP(s, -1.0f, +1.0f) * M) * (1.0f / M);
  vec2  bestRepresentation = s;
  float highestCosine      = dot(oct_to_vec(s), v);
  // Test all combinations of floor and ceil and keep the best.
  // Note that at +/- 1, this will exit the square... but that
  // will be a worse encoding and never win.
  for(int i = 0; i <= 1; ++i)
  {
    for(int j = 0; j <= 1; ++j)
    {
      // This branch will be evaluated at compile time
      if((i != 0) || (j != 0))
      {
        // Offset the bit pattern (which is stored in floating
        // point!) to effectively change the rounding mode
        // (when i or j is 0: floor, when it is one: ceiling)
        vec2  candidate = vec2(i, j) * (1 / M) + s;
        float cosine    = dot(oct_to_vec(candidate), v);
        if(cosine > highestCosine)
        {
          bestRepresentation = candidate;
          highestCosine      = cosine;
        }
      }
    }
  }
  return bestRepresentation;
}

ATTRENC_INLINE vec3 normal_unpack(uint32_t packed)
{
  const uint32_t mask = (1 << (ATTRENC_NORMAL_BITS / 2)) - 1;

  uvec2 pv = uvec2(packed, (packed >> 11)) & uvec2(mask);
  vec2  v  = (vec2(pv) / float(mask)) * 2.0f - 1.0f;

  return oct_to_vec(v);
}

ATTRENC_INLINE uint32_t normal_pack(vec3 normal)
{
  vec2           v    = vec_to_oct_precise(normal, (ATTRENC_NORMAL_BITS / 2));
  const uint32_t mask = (1 << (ATTRENC_NORMAL_BITS / 2)) - 1;

  v = (v + 1.0f) * 0.5f * float(mask) + 0.5f;

  uint32_t packed = uint32_t(v.x) & mask;
  packed |= (uint32_t(v.y) & mask) << 11;

  return packed;
}

// Tangent packing based on "3 BYTE TANGENT FRAMES"
// https://advances.realtimerendering.com/s2020/RenderingDoomEternal.pdf

//-----------------------------------------------------------------------------
// Builds an orthonormal basis: given only a normal vector, returns a
// tangent and bitangent.
//
// This uses the technique from "Improved accuracy when building an orthonormal
// basis" by Nelson Max, https://jcgt.org/published/0006/01/02.
// Any tangent-generating algorithm must produce at least one discontinuity
// when operating on a sphere (due to the hairy ball theorem); this has a
// small ring-shaped discontinuity at normal.z == -0.99998796.
//-----------------------------------------------------------------------------
ATTRENC_INLINE void tangent_orthonormalBasis(vec3 normal, ATTRENC_OUT(vec3) tangent, ATTRENC_OUT(vec3) bitangent)
{
  if(normal.z < -0.99998796F)  // Handle the singularity
  {
    tangent   = vec3(0.0F, -1.0F, 0.0F);
    bitangent = vec3(-1.0F, 0.0F, 0.0F);
    return;
  }
  float a   = 1.0F / (1.0F + normal.z);
  float b   = -normal.x * normal.y * a;
  tangent   = vec3(1.0F - normal.x * normal.x * a, b, -normal.x);
  bitangent = vec3(b, 1.0f - normal.y * normal.y * a, -normal.y);
}

ATTRENC_INLINE uint32_t tangent_pack(vec3 normal, vec4 tangent)
{
  const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;

  vec3 autoTangent;
  vec3 autoBitangent;

  tangent_orthonormalBasis(normal, autoTangent, autoBitangent);

  float angle = ATTRENC_ATAN2F(dot(autoTangent, vec3(tangent)), dot(autoBitangent, vec3(tangent))) / ATTRENC_PI;

  float    angleUnorm = min(max((angle + 1.0f) * 0.5f, 0.0f), 1.0f);
  uint32_t angleBits  = uint32_t(angleUnorm * float(mask) + 0.5f);
  uint32_t encoded    = uint32_t((angleBits << 1) | ((tangent.w > 0.0f ? 1 : 0)));
  return encoded;
}

ATTRENC_INLINE vec4 tangent_unpack(vec3 normal, uint32_t encoded)
{
  const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;

  uint32_t signBit   = encoded & 1;
  uint32_t angleBits = (encoded >> 1) & mask;

  float angleUnorm = float(angleBits) / float(mask);

  float angle = ((angleUnorm * 2.0f) - 1.0f) * ATTRENC_PI;

  vec3 autoTangent;
  vec3 autoBitangent;
  tangent_orthonormalBasis(normal, autoTangent, autoBitangent);

  vec3  tangent = cos(angle) * autoBitangent + sin(angle) * autoTangent;
  float w       = signBit == 1 ? 1.0f : -1.0f;

  return vec4(tangent, w);
}

#undef ATTRENC_ABS
#undef ATTRENC_FLOOR
#undef ATTRENC_CLAMP
#undef ATTRENC_INLINE
#undef ATTRENC_PI
#undef ATTRENC_OUT

#ifdef __cplusplus
}
#endif

#endif