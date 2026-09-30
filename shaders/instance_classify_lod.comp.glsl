/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  
  Shader Description
  ==================
  
  Only used for TARGETS_RAY_TRACING && USE_BLAS_REUSE

  This compute shader classifies the lod range
  of each instance and updates the geometry's lod
  histogram information accordingly.

  Blas caching only needs `lodLevelMin` and the geometry's `cachedLevel`,
  blas sharing also needs `lodLevelMax` and the histogram.

  It also ensure that each tlas instance is initialized
  to use the pre-built low detail blas. This blas assignment
  may later be overridden in `instance_assign_blas.comp.glsl`

  A thread represents one instance.
  
  The follow up procedure to this is
  `geometry_blas_sharing.comp.glsl`
*/

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_RENDERMATERIALS_SSBO, set = 0) buffer renderMaterialsBuffer
{
  RenderMaterial materials[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};


////////////////////////////////////////////

layout(local_size_x=INSTANCES_CLASSIFY_LOD_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

////////////////////////////////////////////

void main()
{
  uint instanceID   = getGlobalInvocationIndex(gl_GlobalInvocationID);
  uint instanceLoad = min(build.numRenderInstances-1, instanceID);
  bool isValid      = instanceID == instanceLoad;
  
  RenderInstance instance = instances[instanceLoad];
  uint geometryID = instance.geometryID;
  Geometry geometry = geometries[geometryID];
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;

#if USE_INSTANCE_OCCLUSION_CULLING
  bool useOcclusion = true;
#else
  bool useOcclusion = false;
#endif
  
  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, geometry.bbox.lo, geometry.bbox.hi, transpose(instance.worldMatrix), clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, CULL_MIN_PIXEL_SIZE) && intersectHiz(clipMin, clipMax, 0)));
  
  uint visibilityState = isVisible ? INSTANCE_VISIBLE_BIT : 0;
  
  if (isValid)
  {
    // setup evaluation of lod metric
    mat4x3 worldMatrix  = transpose(instance.worldMatrix);
    mat4x3 worldMatrixI = transpose(instance.worldMatrixI);
    float uniformScale  = computeUniformScale(worldMatrix);
    float errorScale    = 1.0;
  #if USE_CULLING && !USE_FORCED_INVISIBLE_CULLING
    // instance is not primary visible, apply different error scale
    if (visibilityState == 0) errorScale = build.culledErrorScale;
  #endif
  
    mat4 transform = build.traversalViewMatrix * toMat4(worldMatrix);
    vec3 oViewPos  = (worldMatrixI * vec4(view.viewPos.xyz,1));
    
    // `classifyInstanceLod` iterates the geometry's per lod level root children
    // and determines which lod levels this instance may use.
    //
    // An instance may span multiple lod levels, meaning it has cluster
    // groups from different lod levels.
    // This result depends on distance and orientation of the instance towards
    // the camera.
    //
    //   camera ->             [instance lodLevelMin  .... lodLevelMax]
    //
    // lodLevelMax is only required for blas sharing.

    bool geometryUsesBlasSharing = testForBlasSharing(geometry);
    uint geometryLodLevelMax     = geometry.lodLevelsCount - 1;

    uint lodLevelMin = 0;
    uint lodLevelMax = geometryLodLevelMax;

  #if USE_CULLING && USE_FORCED_INVISIBLE_CULLING
    if(isVisible)
  #endif
    {
      InstanceLodClassification classification =
          classifyInstanceLod(geometry, mat4x3(transform), oViewPos, uniformScale, errorScale, 0, geometryUsesBlasSharing);

      lodLevelMin = classification.lodLevelMin;
      lodLevelMax = classification.lodLevelMax;

      if (visibilityState == 0 && build.sharingPushCulled != 0)
      {
        // For invisible instances we might want to artificially push out
        // the minimum lod level that this instance will think it requires.
        // That way we increase the instance's likelihood to share another blas.
        
        lodLevelMin = min(lodLevelMin + 1, lodLevelMax);
      }
      
      // If the minimum lod level used is actually the maximum lod level available
      // for this geometry, it means the instance only uses the lowest detail
      // cluster group / pre-built blas.
      bool lowestDetailOnly = lodLevelMin == geometryLodLevelMax;
      
      if (TRAVERSAL_ALLOW_LOW_DETAIL_BLAS && lowestDetailOnly)
      {
        // uses the pre-built blas, influences neither sharing nor caching
      }
      else
      {
      #if USE_BLAS_CACHING && !USE_BLAS_SHARING
        // this instance can use the cached blas, accumulate the lowest lod level any
        // such instance needs, the streaming age filter keeps those levels alive.
        // With sharing the same value is derived from the histogram in
        // `geometry_blas_sharing.comp.glsl`, which spares us this atomic.
        if (lodLevelMin >= uint(geometry.discreteLodLevel))
        {
          atomicMin(build.geometryCachedInfos.d[geometryID].cachedLevel, lodLevelMin);
        }
      #endif

        if (geometryUsesBlasSharing)
        {
          // fill geometry histogram
          atomicAdd(build.geometryHistograms.d[geometryID].lodLevelMinHistogram[lodLevelMin], 1);
          atomicAdd(build.geometryHistograms.d[geometryID].lodLevelMaxHistogram[lodLevelMax], 1);

          // we want to find the instance with the highest lod min level (meaning it likely has less detail)
          // for each lod max level

          // pack lod min in upper 5 most significant bits, and instanceID in lower 27
          // this should give us a "stable" result on a static camera
          uint packedLodInstance = (lodLevelMin << 27) | instanceID & 0x7FFFFFF;
          atomicMax(build.geometryHistograms.d[geometryID].lodLevelMaxPackedInstance[lodLevelMax], packedLodInstance);

          // The histogram is evaluated in the
          // `geometry_blas_sharing.comp.glsl` kernel
        }
      }
    }

    // used during `traversal_run.comp.glsl` to allow lower detail for "invisible" instances
    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);
    
    // used in `traversal_init_blas_reuse.comp.glsl` to drive the actual decision
    // which blas an instance should use
    build.instanceBuildInfos.d[instanceID].lodLevelMin            = uint8_t(lodLevelMin);
    build.instanceBuildInfos.d[instanceID].lodLevelMax            = uint8_t(lodLevelMax);
    build.instanceBuildInfos.d[instanceID].geometryLodLevelMax    = uint16_t(geometryLodLevelMax);
    build.instanceBuildInfos.d[instanceID].geometryID             = geometryID;
    
    // ensure that the tlas instances are always initialized to the pre-built low detail blas and
    // renderable in some way.
    // May be overwritten at a later time in `instance_assign_blas.comp.glsl`
    
    // We might want to remove the instance completely if not visible, or just use the low detail blas
  #if USE_CULLING && USE_FORCED_INVISIBLE_CULLING && FORCE_INVISIBLE_CULLED_REMOVES_INSTANCE
    if(!isVisible && build.frameIndex != 0){
      // first frame must always have a valid BLAS due to TLAS BUILD, other frames are TLAS UPDATE
      build.tlasInstances.d[instanceID].blasReference             = 0;
    }
    else
  #endif
    {
      build.tlasInstances.d[instanceID].blasReference             = geometry.lowDetailBlasAddress;
    }

  }
}