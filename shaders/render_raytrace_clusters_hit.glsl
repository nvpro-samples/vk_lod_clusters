/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*

  Shader Description
  ==================

  The closest-hit shading of clusters in ray tracing, run by
  `render_raytrace_clusters.rchit.glsl` and by the ray query compute shader
  `render_raytrace_rq.comp.glsl`. Writes `rayHit`.

  Assumes the includer declared the bindings and payloads of the closest-hit and
  included texturing.glsl, render_shading.glsl and render_cluster_hit.glsl.

*/

#ifndef RENDER_RAYTRACE_CLUSTERS_HIT_GLSL
#define RENDER_RAYTRACE_CLUSTERS_HIT_GLSL

void rtClosestHit(ClusterRayHit hit)
{
  rayHit.hitT = hit.hitT;

#if !USE_DEPTH_ONLY
  // get IDs
  uint clusterID  = hit.clusterID;
  uint instanceID = hit.instanceID;
  uint triangleID = hit.triangleID;
  
#if USE_HIT_STREAMING_LOD
  // ray cone width at this hit, also continued by the secondary rays below.
  // abs: a converging cone (negative spread after a concave mirror) shrinks
  // through its focus and grows again — past the focus the continued cone
  // diverges, hence the spread sign flip.
  float hitConeWidthSigned = rayHit.coneWidth + rayHit.coneSpread * hit.hitT;
  float hitConeWidth       = abs(hitConeWidthSigned);
  float hitConeSpread      = hitConeWidthSigned < 0 ? -rayHit.coneSpread : rayHit.coneSpread;
#endif

#if USE_STREAMING && USE_HIT_STREAMING
#if USE_HIT_STREAMING_LOD
  // The unified ray cone is in pixel-footprint units (spread = view.pixelAngle), shared with texture LOD;
  // hit-based streaming LOD needs geometry-error-budget units (spread = errorOverDistanceThreshold). Both
  // are proportional to tan(fov/2)/height, so convert the footprint with this per-frame ratio.
  float coneToError = build.errorOverDistanceThreshold / view.pixelAngle;
  streamingRegisterHitLod(clusterID, rayHit.coneWidth * coneToError, rayHit.coneSpread * coneToError, hit.wRayOrigin, hit.objectToWorld);
  streamingRegisterInstanceDemand(instanceID, clusterID, rayHit.coneWidth * coneToError, rayHit.coneSpread * coneToError, hit.wRayOrigin,
                                  view.viewPos.xyz, hit.objectToWorld, build.errorOverDistanceThreshold, view.nearPlane,
                                  build.instanceHitDemand);
#else
  streamingRegisterHit(clusterID);
#endif
#endif

  RenderInstance instance = instances[instanceID];
  Geometry       geometry = geometries[instance.geometryID];

  // Fetch cluster header
#if USE_STREAMING
  // dereference the cluster from the resident cluster table
  uint64_t clusterAddress = streaming.resident.clusters.d[clusterID];
#else
  // access the cluster data directly from the preloaded array
  uint64_t clusterAddress = geometry.preloadedClusters.d[clusterID];
#endif
  Cluster_in clusterRef = Cluster_in(clusterAddress);
  Cluster    cluster    = clusterRef.d;

  uint visData = clusterID;
  uint materialID = instance.materialID;

#if ALLOW_SHADING
  materialID = resolveMaterialID(instance, clusterRef, triangleID);

  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(Cluster_in(clusterRef)));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(Cluster_in(clusterRef)));

  uvec3 triangleIndices =
      uvec3(localIndices.d[triangleID * 3 + 0], localIndices.d[triangleID * 3 + 1], localIndices.d[triangleID * 3 + 2]);

  vec3 baryWeight = vec3((1.f - hit.bary.x - hit.bary.y), hit.bary.x, hit.bary.y);

  vec3 oPos = baryWeight.x * hit.pos0 + baryWeight.y * hit.pos1
              + baryWeight.z * hit.pos2;
  vec3 wPos = vec3(hit.objectToWorld * vec4(oPos, 1.0));
  // shadow-ray origin: defaults to the hit point, replaced by the terminator-corrected position below
  // when smooth per-vertex normals are used.
  vec3 wShadowPos = wPos;

  if(view.visualize == VISUALIZE_LOD || view.visualize == VISUALIZE_GROUP)
  {
    if(view.visualize == VISUALIZE_LOD)
    {
      visData = floatBitsToUint(float(cluster.lodLevel) * instances[instanceID].maxLodLevelRcp);
    }
    else
    {
      uvec2 baseAddress = unpackUint2x32(clusterAddress - cluster.groupChildIndex * Cluster_size);
      visData           = baseAddress.x ^ baseAddress.y;
    }
  }
  else if(view.visualize == VISUALIZE_TRIANGLE)
  {
    visData = clusterID * 256 + uint(triangleID);
  }
  else if(view.visualize == VISUALIZE_MATERIAL)
  {
    visData = materialID ^ 0x14325231;
  }

  vec4 wTangent  = vec4(1);
  vec2 oTexCoord = vec2(1);
  vec3 oNormal;
  bool backFacing = false;

  mat3 worldMatrixI = transpose(mat3(instance.worldMatrixI));

#if ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS
  uint32s_in oNormals   = Cluster_getVertexNormals(clusterRef);
  vec2s_in   oTexCoords = Cluster_getVertexTexCoords(clusterRef);
#endif
  {
    // always compute geometric normal
    vec3 e0 = hit.pos1 - hit.pos0;
    vec3 e1 = hit.pos2 - hit.pos0;
    oNormal = (cross(e0, e1));

    backFacing = dot(oNormal, hit.oRayDir) > 0;
  }
  // keep the geometric normal; oNormal is replaced by the interpolated shading normal in the branch below
  vec3 oGeoNormal = oNormal;
#if ALLOW_VERTEX_NORMALS
  if(view.facetShading == 0 && (cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_NORMAL) != 0)
  {
    uvec3 triNormalsPacked =
        uvec3(oNormals.d[triangleIndices.x], oNormals.d[triangleIndices.y], oNormals.d[triangleIndices.z]);
    vec3 triNormals[3];

    triNormals[0] = normal_unpack(triNormalsPacked.x);
    triNormals[1] = normal_unpack(triNormalsPacked.y);
    triNormals[2] = normal_unpack(triNormalsPacked.z);

    oNormal = baryWeight.x * triNormals[0] + baryWeight.y * triNormals[1] + baryWeight.z * triNormals[2];

    // Shadow terminator offset (Hanika 2021), computed in object space and transformed to world.
    // sideFlip keeps the offset on the visible side for back-face hits on two-sided meshes.
    float sideFlip = backFacing ? -1.0 : 1.0;
    vec3  oShadow  = pointOffset(oPos, hit.pos0, hit.pos1,
                                 hit.pos2, triNormals[0] * sideFlip,
                                 triNormals[1] * sideFlip, triNormals[2] * sideFlip, baryWeight);
    wShadowPos = vec3(hit.objectToWorld * vec4(oShadow, 1.0));

#if ALLOW_VERTEX_TANGENTS
    if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TANGENT) != 0)
    {
      vec4 tangent0 = tangent_unpack(triNormals[0], triNormalsPacked.x >> ATTRENC_NORMAL_BITS);
      wTangent.w    = tangent0.w;

      vec3 oTangent = baryWeight.x * tangent0.xyz
                      + baryWeight.y * tangent_unpack(triNormals[1], triNormalsPacked.y >> ATTRENC_NORMAL_BITS).xyz
                      + baryWeight.z * tangent_unpack(triNormals[2], triNormalsPacked.z >> ATTRENC_NORMAL_BITS).xyz;

      wTangent.xyz = normalize(oTangent * worldMatrixI);
    }
#endif
  }
#endif
  // per-vertex texcoords, also kept for the ray-cone texel-density term below
  vec2 uv0 = vec2(0);
  vec2 uv1 = vec2(0);
  vec2 uv2 = vec2(0);
#if ALLOW_VERTEX_TEXCOORD_0
  if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TEX_0) != 0)
  {
    uv0       = oTexCoords.d[triangleIndices.x];
    uv1       = oTexCoords.d[triangleIndices.y];
    uv2       = oTexCoords.d[triangleIndices.z];
    oTexCoord = baryWeight.x * uv0 + baryWeight.y * uv1 + baryWeight.z * uv2;
  }
#endif

  vec3 wNormal    = normalize(vec3(oNormal * worldMatrixI));
  vec3 wGeoNormal = normalize(vec3(oGeoNormal * worldMatrixI));
  if(backFacing)
  {
    wNormal    = -wNormal;
    wGeoNormal = -wGeoNormal;
  }

  // For low-tessellated geometry the interpolated shading normal can tilt far enough that the mirror-
  // reflected view direction points below the geometric surface, producing black specular spots and
  // shadow-ray self-intersection. Snap the shading normal back to the geometric normal in that case.
  if(dot(reflect(hit.wRayDir, wNormal), wGeoNormal) < 0.0)
    wNormal = wGeoNormal;

  vec4 shaded;
  {
    float ambientOcclusion =
        ambientOcclusion(wPos, wNormal, view.ambientOcclusionSamples, view.ambientOcclusionRadius);

    float sunContribution  = 1.0;
    vec3  directionToLight = view.skyPhysical.sunDirection;
    if(view.doShadow == 1)
      sunContribution = traceShadowRay(wShadowPos, wGeoNormal, directionToLight
#if USE_HIT_STREAMING_LOD
                                       , hitConeWidth, hitConeSpread
#endif
      );

    // Ray-cone texture LOD: the cone (width/spread at the ray origin) arrives via the payload and is
    // propagated to this hit - width = coneWidth + coneSpread * hit.hitT. For the primary ray this is
    // spreadAngle*distance, and it accumulates across the mirror reflection. Turns the footprint into a
    // texture LOD so material minification stops sampling mip 0.
    TexLOD texLod = texLodImplicit();
#if HAS_TEXTURED_MATERIALS
    {
      float coneWidth    = rayHit.coneWidth + rayHit.coneSpread * hit.hitT;
      float texelDensity = computeTexelDensity(hit.objectToWorld, hit.pos0,
                                               hit.pos1,
                                               hit.pos2, uv0, uv1, uv2);
      float incidence    = abs(dot(hit.wRayDir, wGeoNormal));
      texLod             = makeConeTexLOD(coneWidth, texelDensity, incidence);
    }
#endif

    shaded = shading(instanceID, materialID, wPos, wNormal, wTangent, oTexCoord, visData, sunContribution, ambientOcclusion, texLod
#if USE_DLSS
                     ,
                     rayHit.dlssAlbedo, rayHit.dlssSpecular, rayHit.dlssNormalRoughness
#endif
    );
  }
#else
  //uint triangleCountMinusOne = clusterRef.d.triangleCountMinusOne;
  uint  triangleCountMinusOne = CLUSTER_TRIANGLE_COUNT - 1;
  float relative              = (float(hit.triangleID) / float(triangleCountMinusOne)) * 0.25 + 0.75;
  vec4  shaded                = vec4(colorizeID(visData) * relative, 1.0);
#endif


#if DEBUG_VISUALIZATION && ALLOW_SHADING
  if(view.doWireframe != 0)
  {
    vec3 derivativeTargetX = hit.worldToObject * vec4(hit.wRayOrigin + rayHit.rayDifferentialX, 1);
    vec3 derivativeDirX    = derivativeTargetX.xyz - hit.oRayOrigin;
    vec3 derivativeX = intersectRayTriangle(hit.oRayOrigin, derivativeDirX, hit.pos0,
                                            hit.pos1, hit.pos2);
    derivativeX      = abs(derivativeX - baryWeight);


    vec3 derivativeTargetY = hit.worldToObject * vec4(hit.wRayOrigin + rayHit.rayDifferentialY, 1);
    vec3 derivativeDirY    = derivativeTargetY.xyz - hit.oRayOrigin;
    vec3 derivativeY = intersectRayTriangle(hit.oRayOrigin, derivativeDirY, hit.pos0,
                                            hit.pos1, hit.pos2);
    derivativeY      = abs(derivativeY - baryWeight);

    vec3 derivative = max(derivativeX, derivativeY);

    rayHit.color.xyz = addWireframe(shaded.xyz, baryWeight, true, derivative);
  }
  else
#endif
  {
    rayHit.color.xyz = shaded.xyz;
  }

  if(RT_LAUNCH_ID == view.mousePosition)
  {
#if !ALLOW_SHADING
    vec3 wPos = hit.wRayOrigin + hit.wRayDir * hit.hitT;
#endif
    vec4  projected            = (view.viewProjMatrix * vec4(wPos, 1.f));
    float depth                = projected.z / projected.w;
    readback.clusterTriangleId = packPickingValue((clusterID << 8) | triangleID, depth);
    readback.instanceId        = packPickingValue(instanceID, depth);
    readback.materialId        = packPickingValue(materialID, depth);
  }
#endif
}

#endif  // RENDER_RAYTRACE_CLUSTERS_HIT_GLSL
