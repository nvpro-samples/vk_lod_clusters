/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _SHADERIO_H_
#define _SHADERIO_H_

#include "shaderio_core.h"
#include "shaderio_scene.h"
#include "shaderio_streaming.h"
#include "shaderio_building.h"
#include "nvshaders/sky_io.h.slang"

/////////////////////////////////////////

#define VISUALIZE_SHADED 0
#define VISUALIZE_GREY 1
#define VISUALIZE_VIS_BUFFER 2
#define VISUALIZE_MATERIAL 3
#define VISUALIZE_CLUSTER 4
#define VISUALIZE_GROUP 5
#define VISUALIZE_LOD 6
#define VISUALIZE_TRIANGLE 7
#define VISUALIZE_BLAS 8
#define VISUALIZE_BLAS_CACHED 9
#define VISUALIZE_DEPTH_ONLY 10

// Texture LOD selection for material sampling, resolved entirely at compile time via TEXTURE_LOD_MODE.
// Raster uses hardware quad derivatives; the ray tracer / path tracer turn the ray-cone footprint into
// either a texture gradient or an explicit lambda. The default is derived from TARGETS_RASTERIZATION
// (set by every renderer) so raster shaders get IMPLICIT and ray/path traced shaders get a cone mode;
// define TEXTURE_LOD_MODE explicitly to override (e.g. TEXLODMODE_LOD for the ray/path tracers).
#define TEXLODMODE_GRAD 0      // convert the ray-cone footprint to a UV gradient and use textureGrad()
#define TEXLODMODE_LOD 1       // compute an explicit lambda and use textureLod()
#define TEXLODMODE_IMPLICIT 2  // use texture() with hardware derivatives (rasterization)

#ifndef TEXTURE_LOD_MODE
#if TARGETS_RASTERIZATION
#define TEXTURE_LOD_MODE TEXLODMODE_IMPLICIT
#else
#define TEXTURE_LOD_MODE TEXLODMODE_GRAD
#endif
#endif

#define MESHSHADER_BBOX_VERTICES 8
#define MESHSHADER_BBOX_LINES 12
#define MESHSHADER_BBOX_THREADS 4

/////////////////////////////////////////

#define BINDINGS_FRAME_UBO 0
#define BINDINGS_READBACK_SSBO 1
#define BINDINGS_GEOMETRIES_SSBO 2
#define BINDINGS_RENDERINSTANCES_SSBO 3
#define BINDINGS_RENDERMATERIALS_SSBO 4
#define BINDINGS_SCENEBUILDING_SSBO 5
#define BINDINGS_SCENEBUILDING_UBO 6
#define BINDINGS_HIZ_TEX 7
#define BINDINGS_STREAMING_UBO 8
#define BINDINGS_STREAMING_SSBO 9
#define BINDINGS_TLAS 10
#define BINDINGS_RAYTRACING_DEPTH 11
#define BINDINGS_RASTER_ATOMIC 12
// DLSS buffers start here as well
#define BINDINGS_RENDER_TARGET 13

/////////////////////////////////////////

#define BUILD_SETUP_TRAVERSAL_RUN 1
#define BUILD_SETUP_TRAVERSAL_RUN_PASS_COMBINED 2
#define BUILD_SETUP_TRAVERSAL_RUN_PASS_NODES_ONLY 3
#define BUILD_SETUP_DRAW 4
#define BUILD_SETUP_BLAS_INSERTION 5

/////////////////////////////////////////

#define STREAM_SETUP_COMPACTION_OLD_NO_UNLOADS 0
#define STREAM_SETUP_COMPACTION_STATUS 1
#define STREAM_SETUP_ALLOCATOR_FREEINSERT 2
#define STREAM_SETUP_ALLOCATOR_STATUS 3
#define STREAM_SETUP_UPDATE_GEOMETRY_INDICES 4

/////////////////////////////////////////

#define TRAVERSAL_PRESORT_WORKGROUP 128
#define TRAVERSAL_INIT_WORKGROUP 64
#define TRAVERSAL_RUN_WORKGROUP 64
#define TRAVERSAL_GROUPS_WORKGROUP 64
#define TRAVERSAL_BLAS_MERGING_WORKGROUP 64
#define BLAS_SETUP_INSERTION_WORKGROUP 64
#define BLAS_INSERT_CLUSTERS_WORKGROUP 64
#define INSTANCES_ASSIGN_BLAS_WORKGROUP 64
#define INSTANCES_CLASSIFY_LOD_WORKGROUP 64
#define GEOMETRY_BLAS_SHARING_WORKGROUP 64
#define BLAS_CACHING_SETUP_BUILD_WORKGROUP 64
#define BLAS_CACHING_SETUP_COPY_WORKGROUP 64

// must be power of 2
#define STREAM_UPDATE_SCENE_WORKGROUP 64
#define STREAM_UPDATE_CLAS_GEOMETRY_INDICES_WORKGROUP 64
#define STREAM_AGEFILTER_GROUPS_WORKGROUP 128
#define STREAM_COMPACTION_NEW_CLAS_WORKGROUP 128
#define STREAM_COMPACTION_OLD_CLAS_WORKGROUP 64
#define STREAM_ALLOCATOR_LOAD_GROUPS_WORKGROUP 64
#define STREAM_ALLOCATOR_UNLOAD_GROUPS_WORKGROUP 64
#define STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP 64
#define STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP 64
#define STREAM_ALLOCATOR_SETUP_INSERTION_WORKGROUP 64


/////////////////////////////////////////

// not exposed in UI, but can be modified and tested via
// shader reload ("R" key)

// for ray tracing only:
// if USE_FORCED_INVISIBLE_CULLING is active and this setting is active as well, then instances
// are removed from the TLAS if invisible. Otherwise they use the low detail BLAS.
// Both options yield different sorts of artifacts, but removing yields better performance.
#define FORCE_INVISIBLE_CULLED_REMOVES_INSTANCE 1

/////////////////////////////////////////

#ifdef __cplusplus
namespace shaderio {
using namespace glm;

#else

#ifndef ALLOW_SHADING
#define ALLOW_SHADING 1
#endif

#ifndef ALLOW_VERTEX_NORMALS
#define ALLOW_VERTEX_NORMALS 1
#endif

#ifndef ALLOW_VERTEX_TEXCOORDS
#define ALLOW_VERTEX_TEXCOORDS 1
#endif

#ifndef ALLOW_VERTEX_TEXCOORD_1
#define ALLOW_VERTEX_TEXCOORD_1 1
#endif

#ifndef ALLOW_VERTEX_TANGENTS
#define ALLOW_VERTEX_TANGENTS 1
#endif

#ifndef USE_SW_RASTER
#define USE_SW_RASTER 0
#endif

#ifndef USE_RENDER_STATS
#define USE_RENDER_STATS 1
#endif

#ifndef USE_MEMORY_STATS
#define USE_MEMORY_STATS 1
#endif

#ifndef USE_CULLING
#define USE_CULLING 1
#endif

#ifndef USE_INSTANCE_OCCLUSION_CULLING
#define USE_INSTANCE_OCCLUSION_CULLING 1
#endif

#ifndef USE_NODE_OCCLUSION_CULLING
#define USE_NODE_OCCLUSION_CULLING 1
#endif

#ifndef USE_CLUSTER_OCCLUSION_CULLING
#define USE_CLUSTER_OCCLUSION_CULLING 1
#endif

#ifndef USE_FORCED_INVISIBLE_CULLING
#define USE_FORCED_INVISIBLE_CULLING 1
#endif

#ifndef USE_TWO_PASS_CULLING
#define USE_TWO_PASS_CULLING 1
#endif

// only effective in NV_mesh_shader
#ifndef USE_PRIMITIVE_CULLING
#define USE_PRIMITIVE_CULLING 1
#endif

#ifndef USE_INSTANCE_SORTING
#define USE_INSTANCE_SORTING 1
#endif

#ifndef USE_BLAS_SHARING
#define USE_BLAS_SHARING 1
#endif

#ifndef USE_BLAS_MERGING
#define USE_BLAS_MERGING 1
#endif

#ifndef USE_BLAS_CACHING
#define USE_BLAS_CACHING 1
#endif

#ifndef USE_STREAMING
#define USE_STREAMING 0
#endif

#ifndef USE_TWO_SIDED
#define USE_TWO_SIDED 1
#endif

#ifndef USE_FORCED_TWO_SIDED
#define USE_FORCED_TWO_SIDED 0
#endif

#ifndef USE_PERSISTENT_TRAVERSAL_KERNEL
#define USE_PERSISTENT_TRAVERSAL_KERNEL 0
#endif

#ifndef MAX_VISIBLE_CLUSTERS
#define MAX_VISIBLE_CLUSTERS 1024
#endif

#ifndef TARGETS_RASTERIZATION
#define TARGETS_RASTERIZATION 1
#endif

#define TARGETS_RAY_TRACING (!(TARGETS_RASTERIZATION))

#ifndef USE_DLSS
#define USE_DLSS 0
#endif

#ifndef USE_DLSS_GUIDE_BUFFERS
#define USE_DLSS_GUIDE_BUFFERS USE_DLSS
#endif

#ifndef HAS_ALPHA_TEST
#define HAS_ALPHA_TEST 1
#endif

#ifndef HAS_TEXTURED_MATERIALS
#define HAS_TEXTURED_MATERIALS 1
#endif

#ifndef CLUSTER_VERTEX_COUNT
#define CLUSTER_VERTEX_COUNT 128
#endif

#ifndef CLUSTER_TRIANGLE_COUNT
#define CLUSTER_TRIANGLE_COUNT 128
#endif

#define GEOMETRY_INDICES_TASKS_PER_WORKGROUP (STREAM_UPDATE_CLAS_GEOMETRY_INDICES_WORKGROUP / SUBGROUP_SIZE)

struct RayPayload
{
#if DEBUG_VISUALIZATION && ALLOW_SHADING
  // Ray directions through neighboring pixels, used to draw wireframe overlays (barycentric differentials).
  vec3 rayDifferentialX;
  vec3 rayDifferentialY;
#endif
#if !USE_DEPTH_ONLY
  vec3 color;
#endif
  float hitT;
  // Ray cone carried into traversal (matches PathRayPayload): width/spread at the ray ORIGIN, so the
  // closest-hit material sampling and the alpha any-hit share the propagated footprint (and it accumulates
  // across the mirror reflection), plus streaming-LOD hooks can read the footprint. width(t) = coneWidth + coneSpread*t.
  float coneWidth;
  float coneSpread;
#if USE_DLSS && ALLOW_SHADING
  vec4 dlssNormalRoughness;
  vec4 dlssAlbedo;
  vec3 dlssSpecular;
#endif
};

// Minimal payload for the path tracer: the closest-hit only reports the hit,
// all shading happens in the ray-generation shader. hitT < 0 signals a miss.
struct PathRayPayload
{
  float hitT;
  uint  instanceID;
  uint  clusterID;
  uint  triangleID;
  vec2  bary;
  // Ray cone carried into traversal (Akenine-Moeller ray cones) so the alpha-mask any-hit - and future
  // streaming-LOD hooks in the hit shaders - use the SAME propagated footprint as the ray-gen material
  // sampling. Width/spread are taken at the ray ORIGIN; a hit at distance t has width = coneWidth + coneSpread * t.
  float coneWidth;
  float coneSpread;
};

#endif

struct FrameConstants
{
  // jittered version for DLSS
  mat4 viewProjMatrixRender;

  mat4 projMatrix;
  mat4 projMatrixI;
  mat4 viewProjMatrix;
  mat4 viewProjMatrixI;
  mat4 viewMatrix;
  mat4 viewMatrixI;
  vec4 viewPos;
  vec4 viewDir;
  vec4 viewPlane;

  // for motion vectors
  mat4 viewProjMatrixPrev;

  ivec2 viewport;
  vec2  viewportf;

  vec2 viewPixelSize;
  vec2 viewClipSize;

  vec3  wLightPos;
  float lightMixer;

  vec3  wUpDir;
  float sceneSize;

  vec4 wMirrorBox;

  uint  colorXor;
  uint  useMirrorBox;
  uint  visualize;
  float fov;

  float   nearPlane;
  float   farPlane;
  float   ambientOcclusionRadius;
  int32_t ambientOcclusionSamples;

  vec4 hizSizeFactors;
  vec4 nearSizeFactors;

  vec2 hizSize;
  vec2 jitter;

  uint  dbgUint;
  float dbgFloat;
  uint  frame;
  uint  doShadow;

  vec4 bgColor;

  uvec2 mousePosition;
  float wireThickness;
  float wireSmoothing;

  vec3 wireColor;
  uint wireStipple;

  vec3  wireBackfaceColor;
  float wireStippleRepeats;

  float wireStippleLength;
  uint  doWireframe;
  uint  visFilterInstanceID;
  uint  visFilterClusterID;

  float texGradScale;
  float pixelAngle;
  int   facetShading;

  SkySimpleParameters skyParams;

  // physical sky used by the basic path tracer
  SkyPhysicalParameters skyPhysical;

  // basic path tracing
  int   pathtraceNumBounces;
  float pathtraceFireflyClamp;
  float pathtraceExposure;      // final exposure multiplier used by the shader (renderer-driven, e.g. auto-exposure)
  float pathtraceExposureBias;  // user exposure compensation in EV stops
  int   pathtraceAutoExposure;  // 0/1 toggle for grid-sampled auto-exposure
  int   pathtraceTonemap;       // operator: 0 = Filmic, 1 = ACES, 2 = Uncharted2
};

struct Readback
{
  uint     numRenderClusters;
  uint     numRenderClustersSW;
  uint     numRenderClustersAlpha;
  uint     numRenderClustersAlphaSW;
  uint     numTraversalTasks;
  uint     numTraversedTasks;
  uint     numBlasBuilds;
  uint     numRenderedClusters;
  uint     numRenderedClustersSW;
  uint     numRenderedClustersAlpha;
  uint     numRenderedClustersAlphaSW;
  uint64_t numRenderedTriangles;
  uint64_t numRenderedTrianglesSW;
  uint64_t numRenderedTrianglesAlpha;
  uint64_t numRenderedTrianglesAlphaSW;
  uint64_t numRasteredTriangles;
  uint64_t numRasteredTrianglesSW;
  uint64_t numRasteredTrianglesAlpha;
  uint64_t numRasteredTrianglesAlphaSW;

  uint64_t blasActualSizes;

  // path tracer auto-exposure: grid-sampled luminance accumulation (float32 atomics)
  float autoExposureLumaSum;
  uint  autoExposureSampleCount;

#ifdef __cplusplus
  uint32_t clusterTriangleId;
  uint32_t _packedDepth0;

  uint32_t instanceId;
  uint32_t _packedDepth1;

  uint32_t materialId;
  uint32_t _packedDepth2;
#else
  uint64_t clusterTriangleId;
  uint64_t instanceId;
  uint64_t materialId;
#endif

  uint64_t debugU64;

  int  debugI;
  uint debugUI;
  uint debugF;

  uint debugA[64];
  uint debugB[64];
  uint debugC[64];
};


#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_H_
