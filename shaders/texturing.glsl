/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/


///////////////////////////////////////////////////


uint resolveClusterLocalMaterialID(Cluster_in clusterRef, uint triangleID)
{
  uint localMaterialID = clusterRef.d.localMaterialID;
  if(localMaterialID == SHADERIO_PER_TRIANGLE_MATERIALS)
  {
    uint8s_in triMaterials = Cluster_getTriangleMaterials(clusterRef);
    localMaterialID        = uint(triMaterials.d[triangleID]);
  }

  return localMaterialID & SHADERIO_LOCAL_MATERIAL_MASK;
}

bool resolveTriangleTwoSided(RenderInstance instance, Cluster_in clusterRef, uint triangleID)
{
  if(instance.multiMaterial == 0)
  {
    return instance.twoSided != 0;
  }

  if(clusterRef.d.localMaterialID != SHADERIO_PER_TRIANGLE_MATERIALS)
  {
    return (clusterRef.d.stateBits & SHADERIO_CLUSTER_TRIANGLE_TWOSIDED) != 0;
  }

  uint8s_in triMaterials = Cluster_getTriangleMaterials(clusterRef);
  uint triMaterial       = uint(triMaterials.d[triangleID]);
  return (triMaterial & SHADERIO_CLUSTER_TRIANGLE_TWOSIDED) != 0;
}

uint resolveMaterialID(RenderInstance instance, Cluster_in clusterRef, uint triangleID)
{
  if(instance.multiMaterial == 0)
  {
    return instance.materialID;
  }

  uint localMaterialID = resolveClusterLocalMaterialID(clusterRef, triangleID);
  return instance.materialID + localMaterialID;
}

uint resolveAlphaMaskTextureIndex(RenderInstance instance, Cluster_in clusterRef, uint triangleID)
{
  if(instance.multiMaterial == 0)
  {
    return instance.alphaMaskTexture;
  }

  uint localMaterialID = resolveClusterLocalMaterialID(clusterRef, triangleID);
  return materials[instance.materialID + localMaterialID].alphaMaskTexture;
}

///////////////////////////////////////////////////

vec2 interpolateTexCoord(vec3 barycentrics, vec2 uv0, vec2 uv1, vec2 uv2)
{
  return barycentrics.x * uv0 +
         barycentrics.y * uv1 +
         barycentrics.z * uv2;
}

float computeTexelDensity(mat4x3 objectToWorld, vec3 pos0, vec3 pos1, vec3 pos2, vec2 uv0, vec2 uv1, vec2 uv2)
{
  vec3  we1    = objectToWorld * vec4(pos1 - pos0, 0.0);
  vec3  we2    = objectToWorld * vec4(pos2 - pos0, 0.0);
  float wArea  = length(cross(we1, we2));
  vec2  duv1   = uv1 - uv0;
  vec2  duv2   = uv2 - uv0;
  float uvArea = abs(duv1.x * duv2.y - duv1.y * duv2.x);

  return sqrt(max(uvArea, 1e-20) / max(wArea, 1e-20));
}

float computeRayFootprintGrad(float hitT, float pixelAngle, float texelDensity)
{
  return hitT * pixelAngle * texelDensity * view.texGradScale;
}

float computeRayFootprintGrad2(float hitT,
                               float pixelAngle,
                               mat4x3 objectToWorld,
                               vec3 rayDirection,
                               vec3 pos0,
                               vec3 pos1,
                               vec3 pos2,
                               vec2 uv0,
                               vec2 uv1,
                               vec2 uv2)
{
  vec3  we1       = objectToWorld * vec4(pos1 - pos0, 0.0);
  vec3  we2       = objectToWorld * vec4(pos2 - pos0, 0.0);
  vec3  wNormal   = cross(we1, we2);
  float wArea     = length(wNormal);
  float wAreaSafe = max(wArea, 1e-20);
  vec2  duv1      = uv1 - uv0;
  vec2  duv2      = uv2 - uv0;
  float uvArea    = abs(duv1.x * duv2.y - duv1.y * duv2.x);
  float incidence = abs(dot(wNormal, rayDirection)) / wAreaSafe;

  return hitT * pixelAngle * sqrt(max(uvArea, 1e-20) / wAreaSafe) * view.texGradScale / max(incidence, 1e-4);
}

vec3 computeViewRayDirection(vec2 clipPos)
{
  vec4 target = normalize(view.projMatrixI * vec4(clipPos.x, clipPos.y, 1, 1));
  return normalize((view.viewMatrixI * vec4(target.xyz, 0)).xyz);
}

float computeRasterFootprintGrad(float triArea, vec2 uv0, vec2 uv1, vec2 uv2)
{
  vec2  duv1   = uv1 - uv0;
  vec2  duv2   = uv2 - uv0;
  float uvArea = abs(duv1.x * duv2.y - duv1.y * duv2.x);

  return sqrt(max(uvArea, 1e-20) / max(triArea, 1e-20)) * view.texGradScale;
}

#if USE_ANISOTROPIC_GRADIENT
struct RayTriangleBarycentricBasis
{
  vec3  v0;
  vec3  e1;
  vec3  e2;
  vec3  normal;
  float d00;
  float d01;
  float d11;
  float invDen;
};

bool computeRayTriangleBarycentricBasis(vec3 v0, vec3 v1, vec3 v2, out RayTriangleBarycentricBasis basis)
{
  basis.v0     = v0;
  basis.e1     = v1 - v0;
  basis.e2     = v2 - v0;
  basis.normal = cross(basis.e1, basis.e2);
  basis.d00    = dot(basis.e1, basis.e1);
  basis.d01    = dot(basis.e1, basis.e2);
  basis.d11    = dot(basis.e2, basis.e2);

  float areaSqr = dot(basis.normal, basis.normal);
  float den     = basis.d00 * basis.d11 - basis.d01 * basis.d01;

  if (areaSqr < 1e-20 || abs(den) < 1e-20)
  {
    return false;
  }

  basis.invDen = 1.0 / den;
  return true;
}

bool intersectRayTriangleBarycentrics(RayTriangleBarycentricBasis basis, vec3 origin, vec3 direction, out vec3 barycentrics)
{
  float nDotDir = dot(basis.normal, direction);

  if (abs(nDotDir) < 1e-20)
  {
    barycentrics = vec3(0);
    return false;
  }

  float t = dot(basis.normal, basis.v0 - origin) / nDotDir;
  vec3  p = origin + t * direction;

  vec3  vp  = p - basis.v0;
  float d20 = dot(vp, basis.e1);
  float d21 = dot(vp, basis.e2);
  float v   = (basis.d11 * d20 - basis.d01 * d21) * basis.invDen;
  float w   = (basis.d00 * d21 - basis.d01 * d20) * basis.invDen;
  barycentrics = vec3(1.0 - v - w, v, w);
  return true;
}

bool computeRayDifferentialTextureGradients(vec3 origin,
                                            vec3 directionX,
                                            vec3 directionY,
                                            vec3 pos0,
                                            vec3 pos1,
                                            vec3 pos2,
                                            vec2 uv0,
                                            vec2 uv1,
                                            vec2 uv2,
                                            vec2 uv,
                                            out vec2 texGradDdx,
                                            out vec2 texGradDdy)
{
  RayTriangleBarycentricBasis basis;
  if (!computeRayTriangleBarycentricBasis(pos0, pos1, pos2, basis))
  {
    texGradDdx = vec2(0);
    texGradDdy = vec2(0);
    return false;
  }

  vec3 baryX;
  vec3 baryY;
  bool validX = intersectRayTriangleBarycentrics(basis, origin, directionX, baryX);
  bool validY = intersectRayTriangleBarycentrics(basis, origin, directionY, baryY);

  if (!validX || !validY)
  {
    texGradDdx = vec2(0);
    texGradDdy = vec2(0);
    return false;
  }

  texGradDdx = (interpolateTexCoord(baryX, uv0, uv1, uv2) - uv) * view.texGradScale;
  texGradDdy = (interpolateTexCoord(baryY, uv0, uv1, uv2) - uv) * view.texGradScale;
  return true;
}

void computeRasterTextureGradients(vec2 pos0,
                                   vec2 pos1,
                                   vec2 pos2,
                                   vec2 uv0,
                                   vec2 uv1,
                                   vec2 uv2,
                                   float invTriArea,
                                   float winding,
                                   out vec2 texGradDdx,
                                   out vec2 texGradDdy)
{
  vec3 baryDdx = vec3(pos2.y - pos1.y, pos0.y - pos2.y, pos1.y - pos0.y) * (winding * invTriArea);
  vec3 baryDdy = vec3(pos1.x - pos2.x, pos2.x - pos0.x, pos0.x - pos1.x) * (winding * invTriArea);

  texGradDdx = interpolateTexCoord(baryDdx, uv0, uv1, uv2) * view.texGradScale;
  texGradDdy = interpolateTexCoord(baryDdy, uv0, uv1, uv2) * view.texGradScale;
}
#endif
