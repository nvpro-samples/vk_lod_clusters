/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/
/*

  Utility code for frustum and occlusion culling of 
  bounding boxes
  
*/

const float c_epsilon    = 1.2e-07f;
const float c_depthNudge = 2.0/float(1<<24);

bool intersectSize(vec4 clipMin, vec4 clipMax, float threshold)
{
  vec2 rect = (clipMax.xy - clipMin.xy) * 0.5 * viewLast.viewportf.xy;
  vec2 clipThreshold = vec2(threshold);
  return any(greaterThan(rect,clipThreshold));
}

vec4 getClip(vec4 hPos, out bool valid) {
  valid = !(-c_epsilon < hPos.w && hPos.w < c_epsilon);
  return vec4(hPos.xyz / abs(hPos.w), hPos.w);
}

uint getCullBits(vec4 hPos)
{
  uint cullBits = 0;
  cullBits |= hPos.x < -hPos.w ?  1 : 0;
  cullBits |= hPos.x >  hPos.w ?  2 : 0;
  cullBits |= hPos.y < -hPos.w ?  4 : 0;
  cullBits |= hPos.y >  hPos.w ?  8 : 0;
  cullBits |= hPos.z <  0      ? 16 : 0;
  cullBits |= hPos.z >  hPos.w ? 32 : 0;
  cullBits |= hPos.w <= 0      ? 64 : 0; 
  return cullBits;
}

vec4 getBoxCorner(vec3 bboxMin, vec3 bboxMax, int n)
{
  bvec3 useMax = bvec3((n & 1) != 0, (n & 2) != 0, (n & 4) != 0);
  return vec4(mix(bboxMin, bboxMax, useMax),1);
}

bool intersectFrustum(vec3 bboxMin, vec3 bboxMax, mat4 worldTM, out vec4 oClipmin, out vec4 oClipmax, out bool oClipvalid)
{
  mat4 worldViewProjTM = viewLast.viewProjMatrix * worldTM;
  bool valid;
  // clipspace bbox
  vec4 hPos     = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, 0);
  vec4 clip     = getClip(hPos, valid);
  uint bits     = getCullBits(hPos);
  vec4 clipMin  = clip;
  vec4 clipMax  = clip;
  bool clipValid = valid;
  
  [[unroll]]
  for (int n = 1; n < 8; n++){
    hPos  = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, n);
    clip  = getClip(hPos, valid);
    bits &= getCullBits(hPos);

    clipMin = min(clipMin,clip);
    clipMax = max(clipMax,clip);

    clipValid = clipValid && valid;
  }
  
  oClipvalid = clipValid;
  oClipmin = vec4(clamp(clipMin.xy, vec2(-1), vec2(1)), clipMin.zw);
  oClipmax = vec4(clamp(clipMax.xy, vec2(-1), vec2(1)), clipMax.zw);

  //return true;
  return bits == 0;
}

#ifndef CULLING_NO_HIZ
bool intersectHiz(vec4 clipMin, vec4 clipMax)
{
  clipMin.xy = clipMin.xy * 0.5 + 0.5;
  clipMax.xy = clipMax.xy * 0.5 + 0.5;
  
  clipMin.xy *= viewLast.hizSizeFactors.xy;
  clipMax.xy *= viewLast.hizSizeFactors.xy;
   
  clipMin.xy = min(clipMin.xy, viewLast.hizSizeFactors.zw);
  clipMax.xy = min(clipMax.xy, viewLast.hizSizeFactors.zw);
  
  vec2  size = (clipMax.xy - clipMin.xy);
  float maxsize = max(size.x, size.y) * viewLast.hizSizeMax;
  float miplevel = ceil(log2(maxsize));

  float depth = textureLod(texHizFar, ((clipMin.xy + clipMax.xy)*0.5),miplevel).r;
  bool result = clipMin.z <= depth + c_depthNudge;

  return result;
}
#endif

////////////////////////////////////////////////////////////////////////////

#define RasterVertex vec4
#define RasterVertex_cullBits(r)  floatBitsToUint(r.w)

vec2 getScreenPos(vec4 hPos)
{
  return vec2(((hPos.xy/hPos.w) * 0.5 + 0.5) * view.viewportf.xy);
}

RasterVertex getRasterVertex(vec4 hPos)
{
  RasterVertex vtx;
  vtx.xy    = getScreenPos(hPos);
  vtx.z     = hPos.z/hPos.w;
  vtx.w     = uintBitsToFloat(getCullBits(hPos));

  return vtx;
}

void pixelBboxEpsilon(inout vec2 pixelMin, inout vec2 pixelMax)
{
  // apply some safety around the bbox to take into account fixed point rasterization
  // (our rasterization grid is 1/256)

  const float epsilon = (1.0 / 256);
  pixelMin -= epsilon;
  pixelMax += epsilon;
  pixelMin = round(pixelMin);
  pixelMax = round(pixelMax);
}

bool pixelBboxCull(vec2 pixelMin, vec2 pixelMax){
  // bbox culling
  bool cull = ( ( pixelMin.x == pixelMax.x) || ( pixelMin.y == pixelMax.y));
  return cull;
}

bool pixelViewportCull(vec2 pixelMin, vec2 pixelMax)
{
  return ((pixelMax.x < 0) || (pixelMin.x >= view.viewportf.x) || (pixelMax.y < 0) || (pixelMin.y >= view.viewportf.y));
}

bool testTriangle(vec2 a, vec2 b, vec2 c, bool frustum, out vec2 pixelMin, out vec2 pixelMax, out float triArea)
{
  // back face culling
  vec2 ab = b.xy - a.xy;
  vec2 ac = c.xy - a.xy;
  float cross_product = ab.y * ac.x - ab.x * ac.y;   

  triArea = cross_product;
#if !USE_TWO_SIDED
  if (cross_product < 0) return false;
#endif

  // compute the min and max in each X and Y direction
  pixelMin = min(a,min(b,c));
  pixelMax = max(a,max(b,c));

  pixelBboxEpsilon(pixelMin, pixelMax);

  // we may not test against frustum / viewport, given the hierarchical culling was doing that on a cluster level already
  if (frustum && pixelViewportCull(pixelMin, pixelMax)) return false;

  if (pixelBboxCull(pixelMin, pixelMax)) return false;

  return true;
}

bool testTriangle(RasterVertex a, RasterVertex b, RasterVertex c, out vec2 pixelMin, out vec2 pixelMax, out float triArea)
{
#if 1
  // skip detailed tests, given sw-raster is only triggered if cluster is within frustum / no complex clip
  if (true)
#else
  if ((RasterVertex_cullBits(a) & RasterVertex_cullBits(b) & RasterVertex_cullBits(c)) == 0 &&
    // don't attempt to to rasterize specially clipped triangles 
    (((RasterVertex_cullBits(a) | RasterVertex_cullBits(b) | RasterVertex_cullBits(c)) & (16 | 32 |64)) == 0))
#endif
  {
    return testTriangle(a.xy,b.xy,c.xy, false, pixelMin, pixelMax, triArea);
  }
  return false;
}

bool testTriangle(vec4 ha, vec4 hb, vec4 hc)
{
  RasterVertex a = getRasterVertex(ha);
  RasterVertex b = getRasterVertex(hb);
  RasterVertex c = getRasterVertex(hc);

  vec2 pixelMin;
  vec2 pixelMax;
  float triArea;

  return testTriangle(a.xy, b.xy, c.xy, false, pixelMin, pixelMax, triArea);
}
