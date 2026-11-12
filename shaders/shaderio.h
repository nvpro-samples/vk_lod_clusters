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
#include "shaderio_ptlas.h"
#include "nvshaders/sky_io.h.slang"
#include "nvshaders/tonemap_io.h.slang"

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
#define VISUALIZE_BLAS_REUSE 9
#define VISUALIZE_DEPTH_ONLY 10
#define VISUALIZE_DISCRETE_LOD 11
#define VISUALIZE_PARTITION 12
#define VISUALIZE_PARTITION_ACTIVE 13

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

// Hit-streaming demand propagation strategy: how the persistent per-group demand
// (`Geometry::groupDemand`) floods coarse->fine for faster stream-in. Selected via
// STREAM_PROPAGATION_MODE (recompiles); see docs/hitbased_streaming.md.
#define STREAM_PROPAGATION_RESIDENT_LEVELS 0   // resident-seeded, single hop (~1 level per stream round-trip)
#define STREAM_PROPAGATION_RESIDENT_SUBTREE 1  // resident-seeded work-proportional multi-pass BFS (demanded subtrees)
#define STREAM_PROPAGATION_ALL_LEVELS 2        // level-wise full sweep over every group, coarsest->finest
#define STREAM_PROPAGATION_ALL_LEVELS_ROLLING                                                                          \
  3  // like ALL_LEVELS but host dispatches one reverse level per frame; compiles as ALL_LEVELS

#ifndef STREAM_PROPAGATION_MODE
#define STREAM_PROPAGATION_MODE STREAM_PROPAGATION_RESIDENT_SUBTREE
#endif

// optical effect applied to the mirror box (see FrameConstants::opticalMode)
#define OPTICAL_MODE_NONE 0
#define OPTICAL_MODE_BOX 1
#define OPTICAL_MODE_SPHERE 2
#define OPTICAL_MODE_LENS 3

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
#define BINDINGS_RAYTRACING_DEPTH 10
#define BINDINGS_RASTER_ATOMIC 11
// DLSS buffers start here as well
#define BINDINGS_RENDER_TARGET 12
// physical sky lighting for raster and ray tracing, baked by `sky_bake.comp.glsl`
#define BINDINGS_SKY_ENV_TEX 20
#define BINDINGS_SKY_IRRADIANCE_TEX 21
// HDRI environment, lat-long with the sampling pdf in alpha, and its alias table, see `environment.glsl`
#define BINDINGS_ENV_HDRI_TEX 22
#define BINDINGS_ENV_HDRI_ACCEL_SSBO 23
// hybrid path: rasterized ids, see `HYBRIDPATH_VISBUFFER_*`
#define BINDINGS_VISIBILITY_BUFFER 24

// hybrid path visibility buffer texel: x = instanceID, y = clusterID << 8 | triangle,
// cleared to empty where nothing was rasterized
#define HYBRIDPATH_VISBUFFER_EMPTY 0xFFFFFFFFu
#define HYBRIDPATH_VISBUFFER_TRIANGLE_BITS 8

// dedicated descriptor set of the clas allocator visualization
#define BINDINGS_ALLOCATOR_VIS_IMAGE 0

// dedicated descriptor set of the tonemap pass, also uses FRAME_UBO and READBACK_SSBO
#define BINDINGS_TONEMAP_IN 2
#define BINDINGS_TONEMAP_OUT 3
// HBAO composite, see `hbao_apply.glsl`
#define BINDINGS_TONEMAP_HBAO_UBO 4
#define BINDINGS_TONEMAP_HBAO_DEPTH 5
#define BINDINGS_TONEMAP_HBAO_RESULT 6

// dedicated descriptor set of the sky bake
#define BINDINGS_SKYBAKE_ENV 0
#define BINDINGS_SKYBAKE_IRRADIANCE 1
// the frame constants, to write `FrameConstants::skyLighting`
#define BINDINGS_SKYBAKE_FRAME_SSBO 2
// also reads BINDINGS_ENV_HDRI_TEX

#define SKY_ENV_SIZE 128
#define SKY_ENV_MIPS 6
#define SKY_IRRADIANCE_SIZE 16
#define SKY_BAKE_WORKGROUP 8
#define SKY_BAKE_ENV 0
#define SKY_BAKE_IRRADIANCE 1
#define SKY_BAKE_SUN 2

#define TONEMAP_WORKGROUP 16
// --rayquery, the ray tracer's image generation as compute, square tiles
#define RAYQUERY_WORKGROUP 8
#define TONEMAP_MODE_FULL 0
// visualization modes of raster and ray tracing, written as-is
#define TONEMAP_MODE_BYPASS 1
// visualization modes of the path tracer, exposed but without a curve that would desaturate the palette
#define TONEMAP_MODE_CLIP 2

/////////////////////////////////////////

#define BUILD_SETUP_TRAVERSAL_RUN 1
#define BUILD_SETUP_TRAVERSAL_RUN_PASS_COMBINED 2
#define BUILD_SETUP_TRAVERSAL_RUN_PASS_NODES_ONLY 3
#define BUILD_SETUP_DRAW 4
#define BUILD_SETUP_BLAS_INSERTION 5
#define BUILD_SETUP_PTLAS_OPS 6
#define BUILD_SETUP_PARTITION_ACTIVE 7
// faster hit-streaming demand BFS (post-traversal), see stream_hit_demand_tree.comp.glsl
#define BUILD_SETUP_HIT_DEMAND_INIT 8       // reset the demand queue counters (before seeding)
#define BUILD_SETUP_HIT_DEMAND_RUN 9        // first frontier range from the seeds
#define BUILD_SETUP_HIT_DEMAND_RUN_PASS 10  // advance the frontier one BFS level
#define BUILD_SETUP_HYBRID_DRAW 11          // hybrid path: second raster pass

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
#define TRAVERSAL_REJECT_CLUSTERS_WORKGROUP 64
#define HYBRIDPATH_CULL_WORKGROUP 64
#define TRAVERSAL_BLAS_MERGING_WORKGROUP 64
#define BLAS_SETUP_INSERTION_WORKGROUP 64
#define BLAS_DEDUP_SCAN_WORKGROUP 64
// number of buckets a blas candidate tries before it gives up and builds its own blas
#define BLAS_DEDUP_PROBE_COUNT 8
#define BLAS_INSERT_CLUSTERS_WORKGROUP 64
#define INSTANCES_ASSIGN_BLAS_WORKGROUP 64
#define INSTANCES_CLASSIFY_LOD_WORKGROUP 64
#define PARTITION_SETUP_WORKGROUP 64
#define GEOMETRY_CACHED_RETAIN_WORKGROUP 64
#define INSTANCE_HIT_SCALE_WORKGROUP 64
// instance_classify_lod, traversal_init_blas_reuse and instance_assign_blas all use 64,
// so one indirect dispatch serves the partition early-out for all three
#define PARTITION_ACTIVE_INSTANCES_WORKGROUP 64
#define PARTITION_BLOCK_COMPACT_WORKGROUP 64
// The partition-sorted instance array is split into blocks of this size; the early-out
// compacts block ids rather than instance ids, so the per-frame work is proportional to
// the block count (thousands) rather than the instance count (millions). Smaller blocks
// track the active partitions more tightly but leave more blocks to compact.
#define PARTITION_INSTANCE_BLOCK_SIZE 1024
#define GEOMETRY_BLAS_SHARING_WORKGROUP 64
#define BLAS_CACHING_SETUP_BUILD_WORKGROUP 64
#define BLAS_CACHING_SETUP_COPY_WORKGROUP 64

// must be power of 2
#define STREAM_UPDATE_SCENE_WORKGROUP 64
#define STREAM_UPDATE_CLAS_GEOMETRY_INDICES_WORKGROUP 64
#define STREAM_AGEFILTER_GROUPS_WORKGROUP 128
// smaller than the age-filter groups: the demand-tree kernel distributes each
// group's finer children across the subgroup, so 2 subgroups per workgroup keeps
// the per-subgroup packing tight (see stream_hit_demand_tree.comp.glsl)
#define STREAM_HIT_DEMAND_TREE_WORKGROUP 64
#define STREAM_COMPACTION_NEW_CLAS_WORKGROUP 128
#define STREAM_COMPACTION_OLD_CLAS_WORKGROUP 64
#define STREAM_ALLOCATOR_LOAD_GROUPS_WORKGROUP 64
#define STREAM_ALLOCATOR_UNLOAD_GROUPS_WORKGROUP 64
#define STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP 64
#define STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP 64
#define STREAM_ALLOCATOR_SETUP_INSERTION_WORKGROUP 64
#define STREAM_ALLOCATOR_VIS_WORKGROUP 64


/////////////////////////////////////////

// not exposed in UI, but can be modified and tested via
// shader reload ("R" key)

// in the second pass of two pass culling, passing instances below this size are culled. Much
// bigger than the lossless `FrameConstants::cullMinPixelSize`: re-traversal only refreshes lod and
// visibility over a few pixels, yet costs as much as any other instance.
#define CULL_SECOND_PASS_MIN_PIXEL_SIZE 8.0

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

#ifndef USE_HYBRID_SW_RASTER
#define USE_HYBRID_SW_RASTER 0
#endif

// compute raster: clusters per workgroup, 32 threads each. More in flight hide their loads.
#define SW_RASTER_CLUSTERS_PER_WORKGROUP 4

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

// the host pre-ands these with USE_CULLING, like USE_PRIMITIVE_CULLING below,
// so they can be tested on their own
#ifndef USE_TWO_PASS_CULLING
#define USE_TWO_PASS_CULLING 1
#endif

// two pass culling records what pass 0 rejected, so the second pass only
// re-tests those against the updated hiz instead of traversing from scratch.
// pre-anded with USE_TWO_PASS_CULLING as well
#ifndef USE_TWO_PASS_REJECT_LISTS
#define USE_TWO_PASS_REJECT_LISTS 1
#endif

// rasterization: a bounding sphere frustum test before the instance loads
#ifndef USE_INSTANCE_PRECULL
#define USE_INSTANCE_PRECULL 0
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

#ifndef USE_BLAS_REUSE
#define USE_BLAS_REUSE (USE_BLAS_SHARING || USE_BLAS_CACHING)
#endif

#ifndef USE_BLAS_DEDUP
#define USE_BLAS_DEDUP 0
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

// persistent producer/consumer variant of the hit-demand BFS descent
// (stream_hit_demand_tree.comp.glsl); reuses the traversal queue counters.
#ifndef USE_PERSISTENT_HIT_DEMAND_KERNEL
#define USE_PERSISTENT_HIT_DEMAND_KERNEL 0
#endif

#ifndef MAX_VISIBLE_CLUSTERS
#define MAX_VISIBLE_CLUSTERS 1024
#endif

#ifndef USE_HIT_STREAMING
#define USE_HIT_STREAMING 1
#endif

#ifndef TARGETS_RASTERIZATION
#define TARGETS_RASTERIZATION 0
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
  // across the mirror reflection). The hit-based streaming-LOD hooks read the SAME footprint, so geometry
  // LOD ranging and texture LOD are driven by one unified ray cone. width(t) = coneWidth + coneSpread*t.
  float coneWidth;
  float coneSpread;
#if USE_DLSS && ALLOW_SHADING
  vec4 dlssNormalRoughness;
  vec4 dlssAlbedo;
  vec3 dlssSpecular;
#endif
};

// Path tracer hit payload: what the closest-hit carries to the ray-gen (which has no hit builtins).
//  0 (default): the world-space geometric normal only (1 vec3) - smaller payload; the shadow-ray
//               origin is the hit point (no Hanika smooth-normal terminator offset).
//  1          : the three object-space triangle positions (3 vec3) - enables the terminator offset.
#ifndef PATHTRACE_HIT_POSITIONS
#define PATHTRACE_HIT_POSITIONS 0
#endif

// Minimal payload for the path tracer: the closest-hit only reports the hit,
// all shading happens in the ray-generation shader. hitT < 0 signals a miss.
struct PathRayPayload
{
  float hitT;
  uint  instanceID;
  uint  clusterID;
  uint  triangleID;
  vec2  bary;
  // Ray cone carried into traversal (Akenine-Moeller ray cones) so the alpha-mask any-hit - and the
  // streaming-LOD hooks in the hit shaders - use the SAME propagated footprint as the ray-gen material
  // sampling. Width/spread are taken at the ray ORIGIN; a hit at distance t has width = coneWidth + coneSpread * t.
  float coneWidth;
  float coneSpread;
#if PATHTRACE_HIT_POSITIONS
  // object-space triangle positions from the CLAS (closest-hit); enable the shadow-terminator offset
  vec3 hitPos0;
  vec3 hitPos1;
  vec3 hitPos2;
#else
  // world-space geometric normal (cross of the triangle's world edges): direction is the geometric
  // normal, length is the triangle world area (drives the ray-cone texture LOD)
  vec3 hitGeoNormal;
#endif
#if DEBUG_VISUALIZATION && ALLOW_SHADING
  // per-pixel barycentric footprint (|d bary| for vertices 0/1/2) derived from the ray cone at the hit,
  // so the ray-gen can draw the wireframe overlay on the primary hit (see render_pathtrace.rgen.glsl).
  vec3 wireBaryDeltas;
#endif
};

#if USE_HIT_STREAMING_LOD
// payload of AO/shadow rays (location 1), otherwise a plain float
struct RayPayloadAO
{
  // 1.0 when the ray hit something (or was left untouched), 0.0 on miss
  float hit;
  // ray cone continued from the surface the secondary ray starts at (width at origin + spread, world space),
  // sharing the unified footprint used for texture LOD.
  float coneWidth;
  float coneSpread;
};
#endif

#endif

// replaces the physical sky when enabled
struct EnvHdri
{
  uint  enabled;
  float intensity;
  // around the up axis, then tilted towards the horizon, radians
  float rotation;
  float elevation;
  uint  width;
  uint  height;
};

struct SkyLighting
{
  // sun disk integrated over the path tracer's sun cone, for a surface facing it
  vec3 sunIrradiance;
  // flashlight scale, matches the path tracer
  float upLuminance;
};

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

  // for motion vectors
  mat4 viewProjMatrixPrev;

  vec2 viewportf;
  vec2 hizSize;
  // exact screen extent in hiz uv, scales and clamps lookups
  vec2 hizSizeFactors;
  // sub-pixel cull next to the hiz occlusion test, boxes below this size in render pixels in both
  // dimensions are culled. 1 is lossless, larger values drop small objects.
  float cullMinPixelSize;
  uint  _pad0;

  vec4 viewPos;
  vec4 viewDir;
  vec4 viewPlane;

  ivec2 viewport;

  vec3  wLightPos;
  float lightMixer;

  vec3  wUpDir;
  float sceneSize;

  vec4 wMirrorBox;

  uint  colorXor;
  uint  visualize;
  float fov;

  float   nearPlane;
  float   farPlane;
  float   ambientOcclusionRadius;  // world-space
  int32_t ambientOcclusionSamples;

  vec2 jitter;

  uint  dbgUint;
  float dbgFloat;
  uint  frame;
  uint  doShadow;

  float timeSec;
  uint  pickedInstanceID;
  uint  pickedClusterID;
  float shadowRayMinT;
  float shadowRayDistanceBias;

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
  // Rasterization-only "solo" filter driven by "Misc Settings > Advanced":
  uint visFilterInstanceID;
  uint visFilterClusterID;

  float texGradScale;
  float texLodBias;       // user bias plus log2(renderSize/targetSize) when DLSS upscales
  float texLodBiasScale;  // exp2(texLodBias), for the gradient paths
  float pixelAngle;
  int   facetShading;
  // optical effect for the mirror box: OPTICAL_MODE_BOX (flat box mirror),
  // OPTICAL_MODE_SPHERE (convex chrome ball) or OPTICAL_MODE_LENS (a
  // camera-anchored magnifier)
  uint opticalMode;

  // OPTICAL_MODE_LENS: magnification of the view through the lens, a narrower
  // rectilinear camera like a zoom lens, and an optional radial distortion
  // (< 0 barrel, > 0 pincushion) relative to the lens rim
  float lensZoom;
  float lensDistortion;

  // the magnifier lens is anchored to the camera; its aperture radius and the
  // distance it is kept ahead of the eye along the view direction (its center
  // is derived from these, independent of wMirrorBox)
  float lensSize;
  float lensDist;

  // lights all visualizations except the raw ones, in all renderers
  SkyPhysicalParameters skyPhysical;

  // basic path tracing
  int   pathtraceNumBounces;
  float pathtraceFireflyClamp;
  // last path vertex is lit like ray tracing: sun shadow ray plus baked sky times AO
  int pathtraceAmbientTail;
  // ambient that the AO rays find occluded still receives this fraction, a stand-in for bounce light
  float pathtraceAmbientFill;

  TonemapperData tonemapper;
  uint           tonemapMode;
  // grey and palette visualizations dim the sky lighting for more contrast
  float skyAmbientScale;
  // rasterization has no sun shadows, a dimmer sky keeps the sun's shaping
  float rasterSkyScale;
  // material of the physically lit grey and palette visualizations
  float debugRoughness;
  float debugSpecular;
  // palette visualizations: constant gain on the linear palette, 0 scales each color to full value
  float debugPaletteGain;
  // grey and palette visualizations: multipliers on the tonemapper's grading
  float debugContrast;
  float debugSaturation;

  // haze along the primary ray, per world unit, 0 disables
  float fogDensity;

  EnvHdri envHdri;

  // written by `sky_bake.comp.glsl` when the sky changes, the host upload stops before it.
  // must stay last.
  SkyLighting skyLighting;
};

struct SkyBakePush
{
  SkyPhysicalParameters sky;
  EnvHdri               envHdri;
  // bumped per loaded HDRI, so a new file re-bakes
  uint envHdriVersion;
  vec3 upDir;
  uint mode;
  uint mip;
};

struct Readback
{
  uint numRenderClusters;
  uint numRenderClustersSW;
  uint numRenderClustersAlpha;
  uint numRenderClustersAlphaSW;
  uint numTraversalTasks;
  uint numHitDemandTasks;  // peak demand-BFS queue usage (shares the traversalNodeInfos array)
  uint numTraversedTasks;
  uint numRejectInstances;
  uint numRejectNodes;
  uint numRejectGroups;
  uint numBlasBuilds;
  uint numBlasCandidates;
  uint numBlasDeduped;
  uint numBlasDedupProbes;
  uint numBlasDedupMissed;
  uint numRenderedClusters;
  uint numRenderedClustersSW;
  uint numRenderedClustersAlpha;
  uint numRenderedClustersAlphaSW;
  uint numDiscreteInstances;
  // hybrid raster lists: first pass, rejected by last frame's hiz, second pass
  uint     numHybridClusters;
  uint     numHybridRejectClusters;
  uint     numHybridClusters2;
  uint     numHybridClustersSW;
  uint64_t numRenderedTriangles;
  uint64_t numRenderedTrianglesSW;
  uint64_t numRenderedTrianglesAlpha;
  uint64_t numRenderedTrianglesAlphaSW;
  uint64_t numRasteredTriangles;
  uint64_t numRasteredTrianglesSW;
  uint64_t numRasteredTrianglesAlpha;
  uint64_t numRasteredTrianglesAlphaSW;

  uint64_t blasActualSizes;

  // ptlas
  uint numPtlasUpdates;
  uint numPtlasDirtyPartitions;
  // all instances within the dirty partitions
  uint numPtlasDirtyInstances;
  // partition early-out: what actually got processed this frame
  uint numActivePartitions;
  uint numActiveInstances;
  // non zero when every partition was forced to be processed regardless of being stable
  uint partitionInvalidateAllCount;
  // instances whose filtered hit lod scale changed, see instance_hit_scale.comp.glsl
  uint numInstanceHitScaleChanges;

  // auto-exposure: grid-sampled luminance accumulation in the tonemap pass (float32 atomics)
  float autoExposureLumaSum;
  uint  autoExposureSampleCount;

#ifdef __cplusplus
  // the uint64_t on the GLSL side are 8 byte aligned, also in scalar layout
  alignas(8) uint32_t clusterTriangleId;
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
