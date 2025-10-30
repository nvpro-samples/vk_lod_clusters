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

#ifndef _TANGENT_ENCODING_H_
#define _TANGENT_ENCODING_H_

#define TANGENT_PI float(3.14159265358979323846264338327950288)

#ifdef __cplusplus
namespace shaderio {
using namespace glm;

#define TANGENT_INLINE inline
#define TANGENT_OUT(a) a&

#else

#define TANGENT_INLINE
#define TANGENT_OUT(a) out a

#endif


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
TANGENT_INLINE void tangent_orthonormalBasis(vec3 normal, TANGENT_OUT(vec3) tangent, TANGENT_OUT(vec3) bitangent)
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


#ifdef __cplusplus

TANGENT_INLINE uint16_t tangent_pack(vec3 normal, vec4 tangent)
{
  vec3 autoTangent;
  vec3 autoBitangent;

  tangent_orthonormalBasis(normal, autoTangent, autoBitangent);

  float angle = atan2f(dot(autoTangent, vec3(tangent)), dot(autoBitangent, vec3(tangent))) / TANGENT_PI;

  float    angleUnorm = min(max(angle * 0.5f + 0.5f, 0.0f), 1.0f);
  uint32_t angleBits  = uint32_t(angleUnorm * float((1 << 15) - 1));
  uint16_t encoded    = uint16_t((angleBits << 1) | ((tangent.w > 0.0f ? 1 : 0)));
  return encoded;
}

}  // namespace shaderio

#else
vec4 tangent_unpack(vec3 normal, uint16_t encoded)
{
  uint32_t signBit   = encoded & 1;
  uint32_t angleBits = encoded >> 1;

  float angleUnorm = float(angleBits) / float((1 << 15) - 1);

  float angle = (angleUnorm - 0.5f) * 2.0f * TANGENT_PI;

  vec3 autoTangent;
  vec3 autoBitangent;
  tangent_orthonormalBasis(normal, autoTangent, autoBitangent);

  vec3  tangent = cos(angle) * autoBitangent + sin(angle) * autoTangent;
  float w       = signBit == 1 ? 1.0f : -1.0f;

  return vec4(tangent, w);
}
#endif

#undef TANGENT_PI
#undef TANGENT_INLINE
#undef TANGENT_OUT

#endif