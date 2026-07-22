/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


///////////////////////////////////////////////////
// Texture LOD selection for ray / path tracing
//
// A ray cone is tracked as a world-space footprint width plus a spread angle
// (Akenine-Moeller et al., "Improved Shader and Texture Level of Detail using
// Ray Cones", JCGT 2021 / Ray Tracing Gems 2019). At a hit the cone width is
// turned into a texture LOD. Two ways to feed the sampler are supported, chosen
// at compile time by TEXTURE_LOD_MODE (see shaderio.h):
//   - TEXLODMODE_GRAD: a UV-space gradient handed to textureGrad(); the hardware
//     folds in each texture's own resolution, so differently sized maps of one
//     material pick the correct mip for free.
//   - TEXLODMODE_LOD: an explicit, resolution-independent lambda handed to
//     textureLod(); sampleBindless() adds 0.5*log2(w*h) per texture.

// TexLOD carries only the payload the compile-time TEXTURE_LOD_MODE actually needs: an explicit lambda
// for TEXLODMODE_LOD, or a UV gradient otherwise (TEXLODMODE_GRAD hands it to textureGrad; the raster
// TEXLODMODE_IMPLICIT path ignores it and lets the hardware derivatives pick the mip).
struct TexLOD
{
#if TEXTURE_LOD_MODE == TEXLODMODE_LOD
  float lodBase;  // resolution-independent lambda; 0.5*log2(w*h) added per texture at sample time
#else
  vec2 gradX;     // UV-space gradient handed to textureGrad()
  vec2 gradY;
#endif
};

// LOD for the rasterizer / hardware-derivative path; the payload is unused (see sampleBindless()).
TexLOD texLodImplicit()
{
  TexLOD tl;
#if TEXTURE_LOD_MODE == TEXLODMODE_LOD
  tl.lodBase = 0.0;
#else
  tl.gradX = vec2(0);
  tl.gradY = vec2(0);
#endif
  return tl;
}

// World-space ray-cone width -> normalized-UV footprint, grazing-corrected (paper Eq. 16/24 normal term).
// Shared by makeConeTexLOD (ray-gen material sampling) and the alpha-mask any-hit, so both derive the
// identical footprint from the cone.
//   coneWidth    : world-space width of the cone at the hit (accumulated spreadAngle*distance)
//   texelDensity : sqrt(uvArea/worldArea) for the hit triangle (computeTexelDensity), normalized-UV per world length
//   incidence    : |dot(rayDir, surfaceNormal)|, the grazing term (footprint stretches at grazing)
float coneFootprintUV(float coneWidth, float texelDensity, float incidence)
{
  // grazing term uses sqrt(incidence) ("moreDetailOnSlopes", JCGT 2021): sharper on slopes than the
  // paper's linear 1/incidence, which tends to over-blur at grazing angles.
  return max(coneWidth * texelDensity * view.texGradScale / sqrt(max(incidence, 1e-4)), 1e-20);
}

// Build a TexLOD from an isotropic ray-cone footprint. The GRAD/LOD choice is the compile-time TEXTURE_LOD_MODE.
TexLOD makeConeTexLOD(float coneWidth, float texelDensity, float incidence)
{
  TexLOD tl;
  float footUV = coneFootprintUV(coneWidth, texelDensity, incidence);
#if TEXTURE_LOD_MODE == TEXLODMODE_LOD
  tl.lodBase = log2(footUV);  // per-texture 0.5*log2(w*h) added in sampleBindless()
#else
  tl.gradX = vec2(footUV, 0.0);
  tl.gradY = vec2(0.0, footUV);
#endif
  return tl;
}

#if (HAS_TEXTURED_MATERIALS || HAS_ALPHA_TEST) && !defined(TEXTURING_SKIP_BINDLESS)
vec4 sampleBindless(uint texIndex, vec2 uv, TexLOD texLod)
{
#if TEXTURE_LOD_MODE == TEXLODMODE_GRAD
  return textureGrad(bindlessTextures[nonuniformEXT(texIndex)], uv, texLod.gradX, texLod.gradY);
#elif TEXTURE_LOD_MODE == TEXLODMODE_LOD
  // lodBase is resolution independent; fold in this texture's own dimensions here so different-sized maps
  // of one material (base color vs. normal vs. metallic-roughness) each land on the right mip.
  vec2  ts  = vec2(textureSize(bindlessTextures[nonuniformEXT(texIndex)], 0));
  float lod = texLod.lodBase + 0.5 * log2(max(ts.x * ts.y, 1.0));
  return textureLod(bindlessTextures[nonuniformEXT(texIndex)], uv, lod);
#else  // TEXLODMODE_IMPLICIT
  return texture(bindlessTextures[nonuniformEXT(texIndex)], uv);
#endif
}
#endif

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

// Screen-space texture gradients from a rasterized triangle (software compute-raster alpha test; the
// hardware fragment path gets these from the GPU). Equivalent to ddx/ddy of the interpolated texcoords.
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
