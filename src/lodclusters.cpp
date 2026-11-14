/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <bit>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <span>

#include <volk.h>
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/camera.hpp>

#include "lodclusters.hpp"
#include <nvshaders/tonemap_functions.h.slang>
#include "partitioned_acceleration_structures.hpp"
#if USE_DLSS
#include "../shaders/dlss_util.h"
#endif

bool g_verbose = false;

namespace lodclusters {

LodClusters::LodClusters(const Info& info)
    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";

  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);

  m_screenshotTimeStamp =
      fmt::format("{:%Y_%m_%d_%H_%M_%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));

  m_info.parameterRegistry->add({"scene", "scene file to load (.gltf, .glb, .cfg)"}, {".gltf", ".glb", ".cfg"}, &m_sceneFilePathDropNew);
  m_info.parameterRegistry->add({"hdrifile", "HDRI environment (.hdr) lighting all renderers instead of the physical sky"},
                                {".hdr"}, &m_envHdriFile);
  m_info.parameterRegistry->add({"hdri", "light with the loaded HDRI, otherwise the physical sky. default true"},
                                &m_tweak.useEnvHdri);
  m_info.parameterRegistry->add({"hdriintensity", "HDRI radiance scale. default 1"}, &m_tweak.envHdriIntensity);
  m_info.parameterRegistry->add({"hdrirotation", "HDRI rotation around the up axis in degrees, RTX Mega Geometry's \"envmap rotation\" + 180. default 0"},
                                &m_tweak.envHdriRotationDeg);
  m_info.parameterRegistry->add({"hdrielevation", "HDRI tilt towards the horizon in degrees. default 0"}, &m_tweak.envHdriElevationDeg);
  m_info.parameterRegistry->add({"renderer", "0 rasterization, 1 ray tracing, 2 path tracing, 3 hybrid: path tracing from rasterized primary visibility. default 1"},
                                (int*)&m_tweak.renderer);
  m_info.parameterRegistry->add({"verbose", "verbose logging"}, &g_verbose, true);
  m_info.parameterRegistry->add({"resetstats", "reset the timer statistics automatically. default false"}, &m_tweak.autoResetTimers);
  m_info.parameterRegistry->add({"resetstatsdelay",
                                 "frames to skip after a config/sequence change before the profiler starts accumulating "
                                 "(sample-side, unclamped; needs resetstats). 0 maps to the profiler default (8)"},
                                &m_tweak.autoResetTimersDelay);
  m_info.parameterRegistry->add({"memoryreportframes", "log a memory report every N frames. 0 disabled (default)"},
                                &m_memoryReportInterval);
  m_info.parameterRegistry->add({"supersample", "render resolution: 1 none, 2 doubles each dimension, or a target height like 1080. default 2"},
                                &m_tweak.supersample);
  m_info.parameterRegistry->add({"debugui", "show the debug window with shader readback values"}, &m_showDebugUI);
  m_info.parameterRegistry->add({"sequencescreenshot", "save screenshot at end of each sequence. 0 disabled (default), 1 full window, 2 rendered viewport"},
                                (int*)&m_sequenceScreenshotMode, true);
  m_info.parameterRegistry->add({"screenshotframegap", "save a screenshot every n frames, and after the last headless frame. 0 disabled (default)"},
                                &m_screenshotFrameInterval);
  m_info.parameterRegistry->addVector(
      {"headlesspicking", "pick at x y relative to the viewport (0..1) instead of the mouse and log the next hit info. default -1 -1 off"},
      &m_headlessPicking);
  m_info.parameterRegistry->add({"screenshotmode", "screenshot content of screenshotframegap: 1 full window, 2 rendered viewport (default)"},
                                (int*)&m_screenshotMode);

  m_info.parameterRegistry->add({"force16bitdispatch",
                                 "use the 16 bit compute launch grid even when the device does not need it, for "
                                 "testing that path. command line / config only, it is applied at startup"},
                                &m_resources.m_force16bitDispatch);
  m_info.parameterRegistry->add({"dumpspirv", "dumps compiled spirv into working directory"}, &m_resources.m_dumpSpirv);
  m_info.parameterRegistry->add({"dumpinternal", "dumps the driver internal pipeline representation into working directory"},
                                &m_resources.m_dumpInternal);
  m_info.parameterRegistry->add({"camerastring", "initial camera, copy/paste from the Misc Settings -> Camera widget"}, &m_cameraString);
  m_info.parameterRegistry->add({"cameraspeed", "camera movement speed, 0 derives it from the scene size"}, &m_cameraSpeed);
  m_info.parameterRegistry->add({"cameragltf", "index of the glTF camera to start with, default 0"}, &m_cameraGltf);
  m_info.parameterRegistry->add({"cameraclipnear", "default camera near clip plane as fraction of the scene diagonal, negative is absolute, 0 derives it from the scene size"},
                                &m_cameraClipNear);
  m_info.parameterRegistry->add({"cameraclipfar", "default camera far clip plane as fraction of the scene diagonal, negative is absolute, 0 derives it from the scene size"},
                                &m_cameraClipFar);
  registerCameraPathParameters();
  m_frameCapture.registerParameters(*m_info.parameterRegistry);
  m_info.parameterRegistry->addVector({"sundirection", "sun direction"}, &m_frameConfig.frameConstants.skyPhysical.sunDirection);

  m_info.parameterRegistry->add({"streamingreset", "reset the streaming state once, e.g. at the start of a parameter sequence"},
                                &m_streamingResetRequest, true);
  m_info.parameterRegistry->add({"streaming", "stream geometry on demand, 0 preloads the whole scene and can need a lot of memory. default true"},
                                &m_tweak.useStreaming);
  m_info.parameterRegistry->add({"clasallocator", "persistent device-side CLAS allocator, otherwise move based compaction. default true"},
                                &m_streamingConfig.usePersistentClasAllocator);
  m_info.parameterRegistry->add({"persistentpositions", "ray tracing keeps the vertex positions resident like rasterization does and builds the CLAS from them, to compare memory. The hybrid renderer always does. default false"},
                                &m_streamingKeepPositions);
  m_info.parameterRegistry->add({"gridcopies", "instance the entire scene on a grid this many times. default 1"},
                                &m_sceneGridConfig.numCopies);
  m_info.parameterRegistry->add({"gridconfig", "grid config in 6 bits: 0..2 enabled axis, 3..5 enabled rotation. default 13"},
                                &m_sceneGridConfig.gridBits);
  m_info.parameterRegistry->add({"gridunique", "fraction (0..1) of grid copies with their own geometries, 1 = all unique, stresses streaming and memory. default 0"},
                                &m_sceneGridConfig.uniqueGeometries);
  m_info.parameterRegistry->add({"clusterconfig", "cluster size: 0 64T_64V, 1 64T_128V, 2 64T_192V, 3 96T_96V, 4 128T_128V, 5 128T_256V, 6 256T_256V. default 4"},
                                (int*)&m_tweak.clusterConfig);
  m_info.parameterRegistry->add({"clustergroupsize", "clusters per lod group, decimated together and sharing one error. default 32"},
                                &m_sceneConfig.clusterGroupSize);
  m_info.parameterRegistry->add({"simplifymaterialweight", "weight of the material in the simplification error metric, 0 disables. default 0.1"},
                                &m_sceneConfig.simplifyMaterialWeight);
  m_info.parameterRegistry->add({"simplifyuvweight", "weight of texcoords in the simplification error metric, 0 disables. default 0.5"},
                                &m_sceneConfig.simplifyTexCoordWeight);
  m_info.parameterRegistry->add({"simplifynormalweight", "weight of normals in the simplification error metric, 0 disables. default 0.5"},
                                &m_sceneConfig.simplifyNormalWeight);
  m_info.parameterRegistry->add({"simplifytangentweight", "weight of tangents in the simplification error metric, 0 disables. default 0"},
                                &m_sceneConfig.simplifyTangentWeight);
  m_info.parameterRegistry->add({"simplifytangentsignweight",
                                 "weight of the tangent sign in the simplification error metric, 0 disables. default 0.2"},
                                &m_sceneConfig.simplifyTangentSignWeight);
  m_info.parameterRegistry->add({"attributes", "vertex attribute bitmask: 1 normal, 2 tangent, 4 texcoord 0, 8 texcoord 1. default 1"},
                                &m_sceneConfig.enabledAttributes);
  m_info.parameterRegistry->add({"multimaterials", "allow a mesh to have multiple materials. default false"},
                                &m_sceneConfig.enableMultiMaterials);
  m_info.parameterRegistry->add({"loderrormergeprevious", "lod error propagation: scales the previous error in max(previous * factor, error). >= 1, default 1.5"},
                                &m_sceneConfig.lodErrorMergePrevious);
  m_info.parameterRegistry->add({"loderrormergeadditive", "lod error propagation: adds this much of the current error after the maximum. default 0"},
                                &m_sceneConfig.lodErrorMergeAdditive);
  m_info.parameterRegistry->add({"loderroredgelimit", "limit the lod error by edge length, to drop subpixel triangles despite high attribute error. default 1"},
                                &m_sceneConfig.lodErrorEdgeLimit);
  m_info.parameterRegistry->add({"simplifyerrorclamped",
                                 "clamp the attribute error to the position error scale, avoids overly conservative "
                                 "lod picking. default true"},
                                &m_sceneConfig.simplifyErrorClamped);
  m_info.parameterRegistry->add({"simplifypreservefolds", "try to keep fold lines between opposite-facing triangles, costs a bit of processing time. default false"},
                                &m_sceneConfig.simplifyPreserveFolds);
  m_info.parameterRegistry->add({"simplifydilateall",
                                 "dilate open cluster borders to compensate the area loss of simplification, meant "
                                 "for foliage rather than everything. default false"},
                                &m_sceneConfig.simplifyDilateBordersAll);
  m_info.parameterRegistry->add({"simplifydilatetwosided",
                                 "enable simplifydilateall for geometries that use a two-sided material, which "
                                 "foliage typically does. default true"},
                                &m_sceneConfig.simplifyDilateBordersTwoSided);
  m_info.parameterRegistry->add({"optimizeclusterslevel", "triangle order optimization within a cluster, 0 off, higher trades processing time for compression ratio. default 1"},
                                &m_sceneConfig.optimizeClustersLevel);
  m_info.parameterRegistry->add({"lodnodewidth", "preferred children per lod node, the maximum is always 32. default 8"},
                                &m_sceneConfig.preferredNodeWidth);
  m_info.parameterRegistry->add({"loddecimationfactor", "triangle reduction factor per lod step. default 0.5"},
                                &m_sceneConfig.lodLevelDecimationFactor);
  m_info.parameterRegistry->add({"meshoptpreferrt", "build lod clusters preferring ray tracing over rasterization. default true"},
                                &m_sceneConfig.meshoptPreferRayTracing);
  m_info.parameterRegistry->add({"meshoptfillweight", "when preferring ray tracing, 0 is SAH optimized, higher fills clusters. default 0.5"},
                                &m_sceneConfig.meshoptFillWeight);
  m_info.parameterRegistry->add({"loderror", "lod pixel error threshold. default 1"}, &m_frameConfig.lodPixelError);
  m_info.parameterRegistry->add({"adaptiveerror", "scale the pixel error by the streaming load factor. default false"},
                                &m_frameConfig.adaptiveError);
  m_info.parameterRegistry->add({"shadowray", "cast shadow rays in ray tracing. default 1"},
                                &m_frameConfig.frameConstants.doShadow);
  m_info.parameterRegistry->add({"shadowraymint", "shadow ray start offset in world units. default 0.001"},
                                &m_frameConfig.frameConstants.shadowRayMinT);
  m_info.parameterRegistry->add({"shadowraydistancebias",
                                 "scale the shadow ray start offset with the distance to the camera, which compensates "
                                 "the lod error growing with distance. 0 uses a constant offset. default 1"},
                                &m_frameConfig.frameConstants.shadowRayDistanceBias);
  m_info.parameterRegistry->add({"ao", "ambient occlusion, same setting as hbao. default true"}, &m_tweak.hbaoActive);  // use same as hbao
  m_info.parameterRegistry->add({"aoradius", "ray traced ambient occlusion radius as percentage to scene bounding radius, negative is absolute world-space radius. default 0.1"},
                                &m_tweak.aoRadius);
  m_info.parameterRegistry->add({"hbao", "ambient occlusion in rasterization (HBAO). default true"}, &m_tweak.hbaoActive);
  m_info.parameterRegistry->add({"fogdistance", "distance at which haze halves the light of the shaded visualization, as fraction of the scene diagonal, negative is absolute. scaled by the sky's haze, 0 disables (default)"},
                                &m_tweak.fogDistance);
  m_info.parameterRegistry->add({"debugexposure", "grey and palette visualizations: exposure boost in stops. default 0.6"},
                                &m_tweak.debugExposure);
  m_info.parameterRegistry->add({"debugskyscale", "grey and palette visualizations: sky lighting scale against the sun. default 0.35"},
                                &m_tweak.debugSkyScale);
  m_info.parameterRegistry->add({"rasterskyscale", "rasterization: sky lighting scale, compensates for the missing sun shadows. default 0.6"},
                                &m_tweak.rasterSkyScale);
  m_info.parameterRegistry->add({"debugroughness", "grey and palette visualizations: material roughness. default 0.5"},
                                &m_tweak.debugRoughness);
  m_info.parameterRegistry->add({"debugspecular", "grey and palette visualizations: specular strength, 1 is F0 0.04. default 1"},
                                &m_tweak.debugSpecular);
  m_info.parameterRegistry->add({"debugpalettegain", "palette visualizations: gain on the linear palette, 0 scales each color to full value. default 1.5"},
                                &m_tweak.debugPaletteGain);
  m_info.parameterRegistry->add({"debugcontrast", "grey and palette visualizations: contrast multiplier. default 1.2"},
                                &m_tweak.debugContrast);
  m_info.parameterRegistry->add({"debugsaturation", "palette visualizations: saturation multiplier. default 1.15"},
                                &m_tweak.debugSaturation);
  m_info.parameterRegistry->add({"hbaoradius", "HBAO radius as percentage to scene bounding radius, negative is absolute world-space radius. default 0.05"},
                                &m_tweak.hbaoRadius);
  m_info.parameterRegistry->add({"hbaoblur", "blur the HBAO result. default true"}, &m_frameConfig.hbaoSettings.blur);
  m_info.parameterRegistry->add({"hbaoquarter", "HBAO at half width and height, upsampled depth-aware. default true"},
                                &m_tweak.hbaoQuarter);
  m_info.parameterRegistry->add({"claspositionbits", "CLAS position mantissa bits to drop to save memory, 0 to 22. default 0"},
                                &m_streamingConfig.clasPositionTruncateBits);
  m_info.parameterRegistry->add({"maxtransfermegabytes", "per-frame transfer budget in MiB, negative is percentage of device local heap (-25 == 25 %)"},
                                &m_memoryBudgetArgs.maxTransferMegaBytes);
  m_info.parameterRegistry->add({"asynctransfer", "use the asynchronous transfer queue for streaming uploads. default false"},
                                &m_streamingConfig.useAsyncTransfer);
  m_info.parameterRegistry->add({"decoupledtransfer", "let asynchronous transfers span multiple frames, needs asynctransfer. default false"},
                                &m_streamingConfig.useDecoupledAsyncTransfer);
  m_info.parameterRegistry->add({"maxblascachingmegabytes",
                                 "blas cache budget in MiB, negative is percentage of device local heap (-25 == 25 %)"},
                                &m_memoryBudgetArgs.maxBlasCachingMegaBytes);
  m_info.parameterRegistry->add({"maxclasmegabytes", "clas budget in MiB, negative is percentage of device local heap (-25 == 25 %)"},
                                &m_memoryBudgetArgs.maxClasMegaBytes);
  m_info.parameterRegistry->add({"startclasmegabytes", "initial clas allocation in MiB, negative is percentage of device local heap (-25 == 25 %)"},
                                &m_memoryBudgetArgs.startClasMegaBytes);
  m_info.parameterRegistry->add({"clasgrowmegabytes", "clas growth step in MiB, negative is percentage of device local heap (-25 == 25 %)"},
                                &m_memoryBudgetArgs.clasGrowMegaBytes);
  m_info.parameterRegistry->add({"maxgeomegabytes", "streamed geometry budget in MiB, negative is percentage of device local heap (-25 == 25 %)"},
                                &m_memoryBudgetArgs.maxGeometryMegaBytes);
  m_info.parameterRegistry->add({"maxresidentgroups", "maximum resident cluster groups. default 65536"},
                                &m_streamingConfig.maxGroups);
  m_info.parameterRegistry->add({"maxframeloadrequests", "maximum group loads per frame. default 128"},
                                &m_streamingConfig.maxPerFrameLoadRequests);
  m_info.parameterRegistry->add({"maxframeunloadrequests", "maximum group unloads per frame. default 1024"},
                                &m_streamingConfig.maxPerFrameUnloadRequests);
  m_info.parameterRegistry->add({"streamingunloadthreshold", "only unload above this memory load factor, 0 to 0.75. default 0"},
                                &m_frameConfig.streamingUnloadThreshold);
  m_info.parameterRegistry->add({"streamingagethreshold", "unload groups traversal has not used for this many frames. default 16"},
                                &m_frameConfig.streamingAgeThreshold);
  m_info.parameterRegistry->add({"cullederrorscale", "scale the pixel error for instances without primary visibility in ray tracing. >= 1, default 2"},
                                &m_frameConfig.culledErrorScale);
  m_info.parameterRegistry->add({"culling", "occlusion and frustum culling. default true"}, &m_rendererConfig.useCulling);
  m_info.parameterRegistry->add({"cullminpixelsize", "cull boxes smaller than this in render pixels in both dimensions, 1 is lossless. default 1"},
                                &m_frameConfig.frameConstants.cullMinPixelSize);
  m_info.parameterRegistry->add({"primitiveculling", "primitive culling in the NV mesh shader, rasterization and hybrid path. default false"},
                                &m_rendererConfig.usePrimitiveCulling);
  m_info.parameterRegistry->add({"twopassculling", "two pass culling in rasterization, otherwise only last frame's hiz. The hybrid path always culls in two passes. default false"},
                                &m_rendererConfig.useTwoPassCulling);
  m_info.parameterRegistry->add({"twopassrejectlists", "second pass continues from what the first pass rejected, rather than traversing again. Rasterization only. default true"},
                                &m_rendererConfig.useTwoPassRejectLists);
  m_info.parameterRegistry->add({"culledgrouplod", "ray tracing applies the culled error scale to occluded parts of visible instances, always on for the hybrid path. default false"},
                                &m_rendererConfig.useCulledGroupLod);
  m_info.parameterRegistry->add({"hybridmatchcheck", "hybrid path: trace the primary ray too, logs an error with the next sequence step or memory report where it disagrees with the raster. default false"},
                                &m_rendererConfig.useHybridMatchCheck);
  m_info.parameterRegistry->add({"dlss", "enable DLSS, super resolution in rasterization and denoising in ray tracing. default false"},
                                &m_rendererConfig.useDlss);
  m_info.parameterRegistry->add({"dlssquality", "0 max performance, 1 balanced, 2 max quality, 3 ultra performance, 5 DLAA. default 2"},
                                (int*)&m_rendererConfig.dlssQuality);
  m_info.parameterRegistry->add({"dlssdelayload", "render without DLSS until it is loaded in the background, ignored headless and with sequence files. default true"},
                                &m_dlssDelayLoad);
  m_info.parameterRegistry->add(
      {.name            = "pathtrace",
       .help            = "older form of --renderer 2: turns ray tracing into path tracing and back. default false",
       .callbackSuccess = [this](const nvutils::ParameterBase* const) { m_pathtraceParsed = true; }},
      &m_rendererConfig.usePathtrace);
  m_info.parameterRegistry->add({"rayquery", "ray tracing renderer: compute shader with ray queries instead of the ray tracing pipeline. default false"},
                                &m_rendererConfig.useRayQuery);
  m_info.parameterRegistry->add({"hybriddiscretelod", "hybrid path: in the caching tail rasterize a whole cached level, its cached blas matches. default true"},
                                &m_rendererConfig.useHybridDiscreteLod);
  m_info.parameterRegistry->add({"hybridcachinglevels", "hybrid path: --cachinglevels in place of the ray tracing one. default 4"},
                                &m_rendererConfig.hybridCachingLevels);
  m_info.parameterRegistry->add({"hybridblasdedup", "hybrid path: --blasdedup in place of the ray tracing one. default true"},
                                &m_rendererConfig.useHybridBlasDedup);
  m_info.parameterRegistry->add({"pathtracebounces", "indirect bounces, 0 (direct lighting only) to 7. default 1"},
                                &m_frameConfig.frameConstants.pathtraceNumBounces);
  m_info.parameterRegistry->add({"pathtraceambient", "light the last path vertex with the baked sky and AO rays. default true"},
                                &m_frameConfig.frameConstants.pathtraceAmbientTail);
  m_info.parameterRegistry->add({"pathtraceambientfill", "ambient kept in occluded directions, 0 to 1. default 0.15"},
                                &m_frameConfig.frameConstants.pathtraceAmbientFill);
  m_info.parameterRegistry->add({"tonemap", "0 Filmic, 1 Uncharted2, 2 Clip (no curve), 3 ACES, 4 AgX, 5 Khronos PBR"},
                                &m_frameConfig.frameConstants.tonemapper.method);
  m_info.parameterRegistry->add({"autoexposure", "adapt exposure to scene luminance, default 1. Disable for deterministic captures"},
                                &m_frameConfig.frameConstants.tonemapper.autoExposure);
  m_info.parameterRegistry->add({"exposure", "exposure multiplier, scaled by auto-exposure when enabled. default 1"},
                                &m_frameConfig.frameConstants.tonemapper.exposure);
  m_info.parameterRegistry->add({"pathtracetonemap", "deprecated - use tonemap"},
                                &m_frameConfig.frameConstants.tonemapper.method);
  m_info.parameterRegistry->add({"pathtraceautoexposure", "deprecated - use autoexposure"},
                                &m_frameConfig.frameConstants.tonemapper.autoExposure);
  m_info.parameterRegistry->add({"pathtraceexposure", "deprecated - use exposure"},
                                &m_frameConfig.frameConstants.tonemapper.exposure);
  m_info.parameterRegistry->add({"pathtracefireflyclamp", "clamp per-bounce radiance, 0 disables. default 20"},
                                &m_frameConfig.frameConstants.pathtraceFireflyClamp);
  m_info.parameterRegistry->add({"lightmixer", "sun strength, the camera flashlight fades in over the sky as it lowers. default 0.85"},
                                &m_frameConfig.frameConstants.lightMixer);
  m_info.parameterRegistry->add({"hitstreaming", "use ray tracing hits to guide streaming in the ray and path tracing renderers. default false"},
                                &m_rendererConfig.useHitStreaming);
  m_info.parameterRegistry->add({"hitstreaminglod", "additionally clamp the lod level based on the ray cone widths of hits, requires hitstreaming. default true"},
                                &m_rendererConfig.useHitStreamingLod);
  m_info.parameterRegistry->add({"hitstreamingshadowdepth",
                                 "path depth up to which sun shadow rays register their hits, requires hitstreaming. 1 from the primary hits (default), 2 also from the first bounce"},
                                &m_rendererConfig.hitStreamingShadowDepth);
  m_info.parameterRegistry->add({"hitstreaminglodfactor", "scale on the acceptable error derived from hit ray cone widths, > 1 allows finer detail. default 1"},
                                &m_frameConfig.hitLodWidthFactor);
  m_info.parameterRegistry->add({"hitstreamingcamerascale",
                                 "hit lod: also open what the camera metric at this multiple of the error wants, 0 disables. default 0"},
                                &m_frameConfig.hitLodCameraScale);
  m_info.parameterRegistry->add({"blassharing", "share BLAS between distant instances, see autosharing. default true"},
                                &m_rendererConfig.useBlasSharing);
  m_info.parameterRegistry->add({"blasmerging", "merged BLAS for close instances when streaming, caps dynamic blas at 2 per geometry. default true"},
                                &m_rendererConfig.useBlasMerging);
  m_info.parameterRegistry->add({"blascaching", "cached BLAS from the highest fully resident lod level, when streaming. default true"},
                                &m_rendererConfig.useBlasCaching);
  m_info.parameterRegistry->add({"cachinglevels", "cached BLAS lod levels per geometry, 1 to 15: more keep a chain from the coarsest tail level up to the finest resident one, refined one level per frame. default 1"},
                                &m_rendererConfig.cachingLevels);
  m_info.parameterRegistry->add({"discretelod", "rasterization: skip the lod traversal for instances that use a single fully resident lod level. default false"},
                                &m_rendererConfig.useDiscreteLod);
  m_info.parameterRegistry->add({"discreteenabledlevels", "allow discrete lod in the last N lod levels, 0 to 32. default 8"},
                                &m_frameConfig.discreteEnabledLevels);
  m_info.parameterRegistry->add({"discretelodrange", "allow discrete lod when an instance spans at most N lod levels, a value at or above the geometry's lod level count imposes no limit. default 16"},
                                &m_frameConfig.discreteLodRange);
  m_info.parameterRegistry->add({"partitions", "group instances into spatial partitions, required by the early-out and the ptlas. default false"},
                                &m_rendererConfig.usePartitions);
  m_info.parameterRegistry->add({"ptlas", "partitioned tlas, updates only instances whose blas changed. default false"},
                                &m_rendererConfig.usePtlas);
  m_info.parameterRegistry->add({"partitionlayout", "how instances are grouped into partitions: 0 fixed grid, 1 derived grid, 2 derived bvh median, 3 derived bvh sah, 4 derived quadtree, 5 node regex (see partitionnodes). default 1"},
                                &m_rendererConfig.partitionLayout);
  m_info.parameterRegistry->add({"partitiongridcellsize", "fixed grid: world units per cell, aligned to the shifted origin, takes precedence over partitiongridcells. 0 disables. default 0"},
                                &m_rendererConfig.partitionGridCellSize, 0.0f);
  {
    const float originMin[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    const float originMax[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    m_info.parameterRegistry->addArray({"partitiongridorigin", "x y z: fixed grid: shifts the world origin the partitiongridcellsize cells are aligned to. default 0 0 0"},
                                       3, &m_rendererConfig.partitionGridOrigin.x, originMin, originMax);
  }
  m_info.parameterRegistry->add({"partitiongridcells", "fixed grid: cells along each axis over the scene bounds, 1 to 4096. default 16"},
                                &m_rendererConfig.partitionGridCellsPerAxis, 1u);
  m_info.parameterRegistry->add({"partitionderivedtarget",
                                 "derived layouts: instances a partition should hold, the bvh and quadtree leaf size or the initial grid resolution. default 64"},
                                &m_rendererConfig.partitionDerivedTarget);
  m_info.parameterRegistry->add({"partitionderivedmax", "derived grid: refines the resolution until no partition holds more instances. default 1024"},
                                &m_rendererConfig.partitionDerivedMaxInstances);
  m_info.parameterRegistry->add({"partitionderivedmaxcount",
                                 "derived layouts: budget for the number of partitions, stops the bvh and quadtree splits and the grid refinement. 0 only respects the device limit. default 4096"},
                                &m_rendererConfig.partitionDerivedMaxCount);
  m_info.parameterRegistry->add({"ptlasexplicitaabb", "provide instance bounds to the ptlas, rather than deriving them from the blas. default false"},
                                &m_rendererConfig.usePtlasExplicitAabb);
  m_info.parameterRegistry->add({"partitionearlyout", "skip the per-instance kernels for partitions whose instances all sit on a persistent blas. default true"},
                                &m_rendererConfig.usePartitionEarlyOut);
  m_info.parameterRegistry->add({"blasdedup", "skip building BLAS that reference the same clusters as another BLAS in the same frame. default false"},
                                &m_rendererConfig.useBlasDedup);
  m_info.parameterRegistry->add({"sharingpushculled", "push culled instances by one lod level for sharing. default true"},
                                &m_frameConfig.sharingPushCulled);
  m_info.parameterRegistry->add({"sharingenabledlevels", "allow BLAS sharing in the last N lod levels, 0 to 32. default 8"},
                                &m_frameConfig.sharingEnabledLevels);
  m_info.parameterRegistry->add({"sharingtolerantlevels", "share BLAS despite a lod level mismatch in the last N levels, 0 to 32. default 7"},
                                &m_frameConfig.sharingTolerantLevels);
  m_info.parameterRegistry->add({"cachingenabledlevels", "allow BLAS caching in the last N lod levels, 0 to 16. default 8"},
                                &m_frameConfig.cachingEnabledLevels);
  m_info.parameterRegistry->add({"cachingpushlevels", "extra lod levels an instance may give up to reach the cached blas, 0 is exact. default 0"},
                                &m_frameConfig.cachingPushLevels);
  m_info.parameterRegistry->add({"cachingstickylevels", "extra lod levels granted to an instance already on the cached blas, keeps it there. default 0"},
                                &m_frameConfig.cachingStickyLevels);
  m_info.parameterRegistry->add({"hybridcachedtolerance", "hybrid path: a visible instance keeps a cached blas within N lod levels of its cut, a known raster mismatch. default -1 off"},
                                &m_frameConfig.hybridCachedTolerance);
  m_info.parameterRegistry->add({"partitionrevalidateframes",
                                 "reprocess a stable partition at least every N frames, also the cached blas retention window. default 64"},
                                &m_frameConfig.partitionRevalidateFrames);
  m_info.parameterRegistry->add({"partitionfrustumrevalidate",
                                 "early-out: process partitions crossing or entering the frustum every frame. default true"},
                                &m_frameConfig.partitionFrustumRevalidate);
  m_info.parameterRegistry->add({"partitionfrustumdistance",
                                 "early-out: the frustum revalidation stops at this fraction of the scene size. "
                                 "default 0.25"},
                                &m_frameConfig.partitionFrustumDistance);
  m_info.parameterRegistry->add({"blascacheretire", "early-out: step a pinned cached blas down to a coarser level rather than holding its groups. default true"},
                                &m_frameConfig.blasCacheRetire);
  m_info.parameterRegistry->add({"instancesorting", "sort instances before traversal. default false"}, &m_rendererConfig.useSorting);
  m_info.parameterRegistry->add({"renderclusterbits", "clusters enqueued per frame in bits (1 << N), BLAS input for ray tracing. 8 to 25, default 20"},
                                &m_rendererConfig.numRenderClusterBits);
  m_info.parameterRegistry->add({"rendertraversalbits", "intermediate traversal tasks in bits (1 << N). 8 to 25, default 20"},
                                &m_rendererConfig.numTraversalTaskBits);
  m_info.parameterRegistry->add({"visualize", "0 shaded, 1 grey, 2 visibility buffer, 3 material, 4 clusters, 5 groups, 6 lod levels, 7 triangles, 8 blas, 9 blas reuse, 10 depth only, 11 discrete lod, 12 partition, 13 partition active. default 6"},
                                &m_frameConfig.visualize);
  m_info.parameterRegistry->add({"instancebboxes", "draw the instance bounding boxes as an overlay. default false"},
                                &m_frameConfig.showInstanceBboxes);
  m_info.parameterRegistry->add({"clusterbboxes", "draw the rendered clusters' bounding boxes as an overlay. default false"},
                                &m_frameConfig.showClusterBboxes);
  m_info.parameterRegistry->add({"swraster", "allow compute-shader rasterization, needs visualize 2 or 10 or the hybrid renderer. default true"},
                                &m_rendererConfig.useComputeRaster);
  m_info.parameterRegistry->add({"swrasterthreshold", "use SW raster when a cluster's longest edge covers fewer projected pixels. default 8"},
                                &m_frameConfig.swRasterThreshold);
  m_info.parameterRegistry->add({"renderstats", "extra atomic counters for statistics, costs performance. default false"},
                                &m_rendererConfig.useRenderStats);
  m_info.parameterRegistry->add({"extmeshshader", "use EXT instead of NV mesh shader, forced on without NV mesh shader support"},
                                &m_rendererConfig.useEXTmeshShader);
  m_info.parameterRegistry->add({"forcepreprocessmegabytes", "always preprocess into a cache file above this geometry size in MiB. default 2048"},
                                (uint32_t*)&m_sceneLoaderConfig.forcePreprocessMiB);
  m_info.parameterRegistry->add({"facetshading", "shade from the geometric (faceted) normal. default true"}, &m_tweak.facetShading);
  m_info.parameterRegistry->add({"flipwinding", "flip the triangle winding. default false"}, &m_rendererConfig.flipWinding);
  m_info.parameterRegistry->add({"forcetwosided", "disable back-face culling for all materials. default false"},
                                &m_rendererConfig.forceTwoSided);
  m_info.parameterRegistry->add({"autosharing", "automatically set blas sharing based on scene's instancing usage. default true"},
                                &m_tweak.autoSharing);
  m_info.parameterRegistry->add({"autoloadcache", "automatically load cache file if found. default true"},
                                &m_sceneLoaderConfig.autoLoadCache);
  m_info.parameterRegistry->add({"mappedcache", "work from memory mapped cache file, otherwise load to sysmem. default false"},
                                &m_sceneLoaderConfig.memoryMappedCache);
  m_info.parameterRegistry->add({"processingonly", "directly terminate app once cache file was saved. default false"},
                                &m_sceneLoaderConfig.processingOnly);
  m_info.parameterRegistry->add({"processingpartial", "in processingonly mode also allow partial/resuming processing. default false"},
                                &m_sceneLoaderConfig.processingAllowPartial);
  m_info.parameterRegistry->add({"processingmemorygigabytes",
                                 "upper budget in GiB for the estimated memory of geometries processed in parallel; "
                                 "0 is automatic (60 % of installed memory), negative is percentage of installed "
                                 "memory (-50 == 50 %), clamped to what is available. default 0"},
                                &m_sceneLoaderConfig.processingMemoryGiB);
  m_info.parameterRegistry->add({"processingthreadpct", "float percentage of threads during initial file load and processing into lod clusters, default 0.5 == 50 %"},
                                &m_sceneLoaderConfig.processingThreadsPct);
  m_info.parameterRegistry->add({"compressed", "store groups compressed, smaller cache file and can speed up streaming. default true"},
                                &m_sceneConfig.useCompressedData);
  m_info.parameterRegistry->add({"compressedpositionbits", "position mantissa bits to drop for better compression, 0 to 22. default 7"},
                                &m_sceneConfig.compressionPosDropBits);
  m_info.parameterRegistry->add({"compressedtexcoordbits", "texcoord mantissa bits to drop for better compression, 0 to 22. default 7"},
                                &m_sceneConfig.compressionTexDropBits);
  m_info.parameterRegistry->add({"cachesuffix", "default is .nvsngeo"}, &m_sceneCacheSuffix);
  m_info.parameterRegistry->add({"simplifyoverrides",
                                 "json file with per-mesh overrides of the simplification settings, each entry is a "
                                 "\"mesh\" name regular expression plus the settings it replaces"},
                                &m_sceneLoaderConfig.simplifyOverridesFile);
  m_info.parameterRegistry->add({"skipnodes", "c++ regular expression string to skip adding instances whose name matches"},
                                &m_sceneLoaderConfig.skipNodeNames);
  m_info.parameterRegistry->add({"partitionnodes", "c++ regular expression string, all instances below a node whose name matches share a partition, a nested match starts a new one, unmatched instances share one more. Used by partitionlayout 5"},
                                &m_sceneLoaderConfig.partitionNodeNames);
  m_info.parameterRegistry->add({"skipmeshes", "c++ regular expression string to skip adding instances whose mesh name matches"},
                                &m_sceneLoaderConfig.skipMeshNames);
  m_info.parameterRegistry->add({"skipmaterials", "c++ regular expression string to skip adding instances whose first mesh material name matches"},
                                &m_sceneLoaderConfig.skipMaterialNames);
  m_info.parameterRegistry->add({"skipalphablended", "skip instances with alpha blended materials. default true"},
                                &m_sceneLoaderConfig.skipAlphaBlended);
  m_info.parameterRegistry->add({"skipalphamasked", "skip instances with alpha masked materials. default false"},
                                &m_sceneLoaderConfig.skipAlphaMasked);
  m_info.parameterRegistry->add({"persistenttraversal", "use the persistent traversal kernel. default true"},
                                &m_rendererConfig.usePersistentTraversal);
  m_info.parameterRegistry->add({"traversalvulkanmemorymodel",
                                 "compile the persistent traversal kernel with the Vulkan memory model, ignored without persistenttraversal. default false"},
                                &m_rendererConfig.useVulkanMemoryModel);
  m_info.parameterRegistry->add({"traversalatomics", "persistent traversal kernel polls with atomic load/store instead of plain coherent access. default true"},
                                &m_rendererConfig.useAtomicLoadStore);
  m_info.parameterRegistry->add({"persistenthitdemand", "hit-streaming demand descent as one persistent dispatch (resident subtree only). default true"},
                                &m_rendererConfig.usePersistentHitDemand);
  m_info.parameterRegistry->add({"texlodmode", "ray/path tracer texture LOD: 0 gradient, 1 explicit lod, 2 mip0"},
                                &m_rendererConfig.textureLodMode);
  m_info.parameterRegistry->add({"texturegradientscale", "scale the texture gradient in ray tracing and compute rasterization, 0 to 1. default 1"},
                                &m_frameConfig.frameConstants.texGradScale);
  m_info.parameterRegistry->add({"texlodbias", "bias added to all material texture LOD selection, on top of the automatic DLSS bias. default 0"},
                                &m_tweak.texLodBias);
  m_info.parameterRegistry->add({"streampropagation", "hit-streaming demand propagation: 0 resident levels, 1 resident subtree (default), 2 all levels, 3 all levels rolling"},
                                &m_rendererConfig.streamPropagationMode);
  m_info.parameterRegistry->add({"streamrollingcoarse", "all levels rolling: coarsest lod levels swept in full per frame, >= 1. default 4"},
                                &m_frameConfig.streamRollingCoarseLevels);
  m_info.parameterRegistry->add({"opticalmode", "ray and path tracing optical effect: 0 none (default), 1 mirror box, 2 mirror sphere, 3 magnifier lens"},
                                (int*)&m_frameConfig.frameConstants.opticalMode);
  m_info.parameterRegistry->add({"lenszoom", "lens magnification, for opticalmode 3. default 4"},
                                &m_frameConfig.frameConstants.lensZoom);
  m_info.parameterRegistry->add({"lensdistortion", "lens radial distortion at its rim, < 0 barrel, > 0 pincushion, for opticalmode 3. default 0"},
                                &m_frameConfig.frameConstants.lensDistortion);
  m_info.parameterRegistry->add({"lenssize", "lens aperture radius in world units, set with lensdistance when that is 0"},
                                &m_frameConfig.frameConstants.lensSize);
  m_info.parameterRegistry->add({"lensdistance", "distance ahead of the camera the screen-centered lens is kept, 0 derives it from the near plane. default 0"},
                                &m_frameConfig.frameConstants.lensDist);
  m_info.parameterRegistry->add({"texturedmaterials", "enable textured materials"}, &m_sceneLoaderConfig.enableTexturedMaterials);
  m_info.parameterRegistry->add({"skipnormalmaps", "skip loading normal maps when textured materials are enabled"},
                                &m_sceneLoaderConfig.skipNormalMaps);
  m_info.parameterRegistry->add({"maxtexturemegabytes",
                                 "upper VRAM budget for material textures in MiB; 0 disables limit, negative is "
                                 "percentage of device local heap (-25 == 25 %). default 4096"},
                                &m_memoryBudgetArgs.maxTextureMegaBytes);
  {
    // HACK as zorah.cfg ships with some deprecated settings
    static bool dummy;
    m_info.parameterRegistry->add({"twosided", "deprecated - now detecting doubleSided materials - there is a new forcetwosided"},
                                  &dummy);
    m_info.parameterRegistry->add({"hbaofullres", "deprecated - now always using full resolution"}, &dummy);
    m_info.parameterRegistry->add({"pathtraceexposurebias", "deprecated - use exposure"}, &dummy);
    static glm::vec3 dummyVec;
    m_info.parameterRegistry->addVector({"suncolor", "deprecated - the simple sky was removed"}, &dummyVec);
  }

  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants.wireThickness           = 2.f;
  m_frameConfig.frameConstants.wireSmoothing           = 1.f;
  m_frameConfig.frameConstants.wireColor               = {118.f / 255.f, 185.f / 255.f, 0.f};
  m_frameConfig.frameConstants.wireStipple             = 0;
  m_frameConfig.frameConstants.wireBackfaceColor       = {0.5f, 0.5f, 0.5f};
  m_frameConfig.frameConstants.wireStippleRepeats      = 5;
  m_frameConfig.frameConstants.wireStippleLength       = 0.5f;
  m_frameConfig.frameConstants.doShadow                = 1;
  m_frameConfig.frameConstants.shadowRayMinT           = 0.001f;
  m_frameConfig.frameConstants.shadowRayDistanceBias   = 1.0f;
  m_frameConfig.frameConstants.doWireframe             = 0;
  m_frameConfig.frameConstants.ambientOcclusionSamples = 2;
  m_frameConfig.frameConstants.visualize               = VISUALIZE_LOD;
  m_frameConfig.frameConstants.facetShading            = 1;
  m_frameConfig.frameConstants.cullMinPixelSize        = 1.0f;
  m_frameConfig.frameConstants.wMirrorBox              = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  m_frameConfig.frameConstants.opticalMode             = OPTICAL_MODE_NONE;
  m_frameConfig.frameConstants.lensZoom                = 4.0f;
  m_frameConfig.frameConstants.lensDistortion          = 0.0f;
  m_frameConfig.frameConstants.lensSize                = 1.0f;
  m_frameConfig.frameConstants.lensDist                = 0.0f;
  m_frameConfig.frameConstants.texGradScale            = 1.0f;

  m_frameConfig.frameConstants.lightMixer = 0.85f;

  m_frameConfig.frameConstants.skyPhysical             = {};
  m_frameConfig.frameConstants.skyPhysical.groundColor = {0.62f, 0.59f, 0.55f};
  // basic path tracer defaults
  m_frameConfig.frameConstants.pathtraceNumBounces   = 1;
  m_frameConfig.frameConstants.pathtraceFireflyClamp = 20.0f;
  m_frameConfig.frameConstants.pathtraceAmbientTail  = 1;
  m_frameConfig.frameConstants.pathtraceAmbientFill  = 0.15f;
  // nvpro_core2 tonemapper defaults, auto-exposure on and mean averaging (no histogram pass)
  m_frameConfig.frameConstants.tonemapper              = {};
  m_frameConfig.frameConstants.tonemapper.autoExposure = 1;
  m_frameConfig.frameConstants.tonemapper.averageMode  = 0;

  m_lastAmbientOcclusionSamples = m_frameConfig.frameConstants.ambientOcclusionSamples;

  m_sceneLoaderConfig.progressInfo.completedCount = &m_sceneCompletedCount;
  m_sceneLoaderConfig.progressInfo.totalCount     = &m_sceneTotalCount;
  m_sceneLoaderConfig.progressInfo.progressPhase  = &m_sceneProgressPhase;
}

void LodClusters::initScene(std::filesystem::path filePath, std::string cacheSuffix, bool configChange)
{
  deinitScene();

  std::string fileName = nvutils::utf8FromPath(filePath);

  if(!fileName.empty())
  {
    LOGI("Loading scene %s\n", fileName.c_str());

    m_scene        = nullptr;
    m_sceneLoading = true;
    // picking results of the old scene must not survive into the new one, their instance and
    // cluster ids do not mean anything there
    m_resources.clearReadbackData();
    m_renderSceneInitAllowed = true;
    m_sceneCompletedCount    = 0;
    m_sceneTotalCount        = 0;
    m_sceneProgressPhase     = uint32_t(LoadPhase::ProcessingScene);
    m_sceneLoaderConfigLast  = m_sceneLoaderConfig;

#if USE_DLSS
    // disable when inactive
    m_resources.setFramebufferDlss(false, m_rendererConfig.dlssQuality);
#endif

    m_sceneLoadingThread = std::thread([=, this]() {
      auto scene = std::make_unique<Scene>();
      if(scene->init(filePath, m_sceneConfig, m_sceneLoaderConfig, cacheSuffix, configChange) != Scene::SCENE_RESULT_SUCCESS)
      {
        scene = nullptr;
        LOGW("Loading scene failed\n");
      }
      else
      {
        // The scene stays private to this thread until it is done, m_scene is only assigned by
        // the main thread in handleChanges. The ui keeps drawing while we load and tests m_scene
        // for null, so a scene whose grid or textures are still being set up must not be in there.
        m_sceneFilePath       = filePath;
        m_tweak.clusterConfig = findSceneClusterConfig(scene->m_config);

        scene->updateSceneGrid(m_sceneGridConfig);
        m_sceneGridConfigLast = m_sceneGridConfig;
        updatedSceneGrid(*scene);

        m_renderSceneCanPreload = ScenePreloaded::canPreload(m_resources.getDeviceLocalHeapSize(), scene.get());

        if(!configChange)
        {
          m_sceneConfig = scene->m_config;
          postInitNewScene(*scene);
          m_tweakLast       = m_tweak;
          m_sceneConfigLast = m_sceneConfig;
          m_sceneConfigEdit = m_sceneConfig;
        }

        // Load textures on this loader thread so the busy popup also reports their progress
        // (Probing/Loading Textures). Uploads run on the transfer queue. Build into m_renderScenePending
        // (not m_renderScene) so the concurrently-running UI never touches a half-constructed scene; the
        // main thread promotes it and finishes the GPU geometry setup via initRenderSceneGeometry().
        auto renderScene = std::make_unique<RenderScene>();
        if(renderScene->initTextures(&m_resources, scene.get(), m_texturesConfig, m_sceneLoaderConfig.progressInfo))
        {
          m_renderScenePending         = std::move(renderScene);
          m_renderSceneGeometryPending = true;
        }
        else
        {
          LOGW("Loading scene textures failed\n");
        }

        // hand over to the main thread, the m_sceneLoading store below publishes it
        m_scenePending = std::move(scene);
      }
      m_sceneLoading = false;
    });

    // a capture should not record the loading screen
    if(m_app->isHeadless() || m_sceneLoadBlocking || m_frameCapture.config.enabled)
    {
      m_sceneLoadingThread.join();
    }

    return;
  }

  return;
}

void LodClusters::initRenderScene()
{
  assert(m_scene);

  m_renderSceneInitAllowed = true;
  m_renderScene            = std::make_unique<RenderScene>();

  // Synchronous (config-change) path: textures load on the main thread here (no busy popup, but
  // the progress atomics are still updated), then the GPU geometry setup follows immediately.
  if(!m_renderScene->initTextures(&m_resources, m_scene.get(), m_texturesConfig, m_sceneLoaderConfig.progressInfo))
  {
    LOGW("Loading scene textures failed\n");
    deinitRenderScene();
    m_renderSceneInitAllowed = false;

    return;
  }
  m_renderSceneGeometryPending = false;
  initRenderSceneGeometry();
}

void LodClusters::initRenderSceneGeometry()
{
  assert(m_renderScene);

  m_renderSceneInitAllowed = true;

  // Finalize the async texture uploads on the graphics queue (this always runs on the main thread,
  // for both the async-load and config-change paths) before the geometry GPU setup.
  m_renderScene->sceneTextures.acquireOwnership(&m_resources);

  updateDerivedStreamingConfig();
  bool success = m_renderScene->initGeometry(&m_resources, m_streamingConfig, m_tweak.useStreaming);

  // if preload fails, try streaming
  if(!m_tweak.useStreaming && !success)
  {
    // override to use streaming
    m_tweak.useStreaming     = true;
    m_tweakLast.useStreaming = true;

    if(!m_renderScene->initGeometry(&m_resources, m_streamingConfig, true))
    {
      LOGW("Init renderscene failed\n");
      deinitRenderScene();
      m_renderSceneInitAllowed = false;
    }
  }
  else if(!success && m_tweak.useStreaming)
  {
    LOGW("Init renderscene failed\n");
    deinitRenderScene();
    m_renderSceneInitAllowed = false;
  }

  m_streamingConfigLast = m_streamingConfig;
  m_texturesConfigLast  = m_texturesConfig;
}

void LodClusters::deinitRenderScene()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
  if(m_renderScene)
  {
    m_renderScene->deinit();
    m_renderScene = nullptr;
  }
  // a loader thread may have built a pending render scene that was never promoted (e.g. reload
  // before the main thread finished the geometry setup). Its textures were uploaded but not yet
  // acquired, so acquire ownership before destroying the images to avoid leaving stale async-upload
  // barriers behind for the next load.
  if(m_renderScenePending)
  {
    m_renderScenePending->sceneTextures.acquireOwnership(&m_resources);
    m_renderScenePending->deinit();
    m_renderScenePending = nullptr;
  }
  m_renderSceneGeometryPending = false;
}

void LodClusters::deinitScene()
{
  // async loading must complete before teardown
  if(m_sceneLoadingThread.joinable())
  {
    m_sceneLoadingThread.join();
  }

  deinitRenderScene();

  // a load may have finished without the main thread getting to install its scene
  if(m_scenePending)
  {
    m_scenePending->deinit();
    m_scenePending = nullptr;
  }

  if(m_scene)
  {
    m_scene->deinit();
    m_scene = nullptr;
  }
}

void LodClusters::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_windowSize = size;
  m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoQuarter);
  updateImguiImage();
  if(m_renderer)
  {
    m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
    m_rendererFboChangeID = m_resources.m_fboChangeID;
  }
}

void LodClusters::updateImguiImage()
{
  if(m_imguiTexture)
  {
    ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);
    m_imguiTexture = nullptr;
  }

  VkImageView imageView = m_resources.m_frameBuffer.useResolved ?
                              m_resources.m_frameBuffer.imgColorResolved.descriptor.imageView :
                              m_resources.m_frameBuffer.imgColorLdr.descriptor.imageView;

  assert(imageView);

  m_imguiTexture = ImGui_ImplVulkan_AddTexture(m_imguiSampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void LodClusters::onPreRender()
{
  m_profilerTimeline->frameAdvance();

  if(m_screenshotFrameInterval > 0 && m_frames > 0 && (m_frames % m_screenshotFrameInterval) == 0)
  {
    saveScreenshot(m_screenshotMode, fmt::format("screenshot_{}_{:08}.jpg", m_screenshotTimeStamp, m_frames - 1));
  }
}

void LodClusters::onPostRender()
{
  m_frameCapture.postSubmit();
}

void LodClusters::onLastHeadlessFrame()
{
  m_frameCapture.finish();

  // the interval never covers the final frame, which is typically the one of interest
  if(m_screenshotFrameInterval > 0 && m_frames > 0)
  {
    saveScreenshot(m_screenshotMode, fmt::format("screenshot_{}_{:08}.jpg", m_screenshotTimeStamp, m_frames - 1));
  }

  if(m_headlessPickingWait >= 0)
  {
    LOGW("headlesspicking: no result yet, needs more headlessframes\n");
  }
}


void LodClusters::deinitRenderer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_renderer)
  {
    m_renderer->deinit(m_resources);
    m_renderer = nullptr;
  }
}

void LodClusters::initRenderer(RendererType rtype)
{
  deinitRenderer();
  if(!m_renderScene)
    return;

  LOGI("Initializing renderer and compiling shaders\n");

  printf("init renderer %d\n", rtype);

  if(m_renderScene->useStreaming)
  {
    if(!m_renderScene->sceneStreaming.reloadShaders())
    {
      LOGE("RenderScene shaders failed\n");
      return;
    }
  }

  // the partition early-out is independent of the ptlas, the tlas path skips the same kernels
  if(m_rendererConfig.usePartitions && rtype != RENDERER_RASTER_CLUSTERS_LOD)
  {
    ScenePartitionConfig partitionConfig;
    partitionConfig.layout                 = m_rendererConfig.partitionLayout;
    partitionConfig.gridCellSize           = m_rendererConfig.partitionGridCellSize;
    partitionConfig.gridOrigin             = m_rendererConfig.partitionGridOrigin;
    partitionConfig.gridCellsPerAxis       = std::min(m_rendererConfig.partitionGridCellsPerAxis, 4096u);
    partitionConfig.derivedTargetInstances = m_rendererConfig.partitionDerivedTarget;
    partitionConfig.derivedMaxInstances    = m_rendererConfig.partitionDerivedMaxInstances;
    partitionConfig.derivedMaxCount        = m_rendererConfig.partitionDerivedMaxCount;
    if(partitionConfig.layout >= ScenePartitionConfig::LAYOUT_COUNT)
    {
      LOGW("--partitionlayout %u is unknown, using the derived grid\n", partitionConfig.layout);
      partitionConfig.layout = ScenePartitionConfig::LAYOUT_DERIVED_GRID;
    }
    // the device limit only binds the ptlas; the early-out on its own has no such cap
    partitionConfig.maxPartitions = m_resources.m_supportsPartitionedTlas && m_rendererConfig.usePtlas ?
                                        PartitionedTlas::getProperties(m_resources.m_physicalDevice).maxPartitionCount :
                                        0;

    m_scene->updatePartitions(partitionConfig);
  }

  switch(rtype)
  {
    case RENDERER_RASTER_CLUSTERS_LOD:
      m_renderer = makeRendererRasterClustersLod();
      break;
    case RENDERER_RAYTRACE_CLUSTERS_LOD:
    case RENDERER_PATHTRACE_CLUSTERS_LOD:
      m_renderer = makeRendererRayTraceClustersLod();
      break;
    case RENDERER_HYBRID_CLUSTERS_LOD:
      m_renderer = makeRendererHybridClustersLod();
      break;
  }

  RendererConfig config = m_rendererConfig;
  config.usePathtrace   = rtype == RENDERER_PATHTRACE_CLUSTERS_LOD || rtype == RENDERER_HYBRID_CLUSTERS_LOD;
  if(m_renderer && !m_renderer->init(m_resources, *m_renderScene, config))
  {
    m_renderer = nullptr;
    LOGE("Renderer init failed\n");
  }

  m_rendererFboChangeID = m_resources.m_fboChangeID;
}

// `scene` rather than m_scene: this also runs on the loader thread, where the scene is not
// installed yet
void LodClusters::postInitNewScene(Scene& scene)
{
  glm::vec3 extent         = scene.m_bbox.hi - scene.m_bbox.lo;
  glm::vec3 center         = (scene.m_bbox.hi + scene.m_bbox.lo) * 0.5f;
  float     sceneDimension = glm::length(extent);

  m_frameConfig.frameConstants.wLightPos = center + sceneDimension;
  m_frameConfig.frameConstants.sceneSize = glm::length(scene.m_bbox.hi - scene.m_bbox.lo);

  setSceneCamera(m_sceneFilePath, scene);

  if(m_frameConfig.frameConstants.lensDist == 0.0f)
  {
    // unset, keep the lens just past the near plane, covering about half the screen height
    m_frameConfig.frameConstants.lensDist = 4.0f * float(m_info.cameraManipulator->getClipPlanes().x);
    m_frameConfig.frameConstants.lensSize = 0.3f * m_frameConfig.frameConstants.lensDist;
  }

  m_frames = 0;
  clampStreamingConfig(scene);

  if(!scene.m_hasVertexNormals)
    m_tweak.facetShading = true;

  // the azimuth/elevation UI expects it normalized
  m_frameConfig.frameConstants.skyPhysical.sunDirection = glm::normalize(m_frameConfig.frameConstants.skyPhysical.sunDirection);
}


void LodClusters::onAttach(nvapp::Application* app)
{
  m_app = app;

  m_tweak.supersample = std::max(1, m_tweak.supersample);
  m_info.cameraManipulator->setMode(nvutils::CameraManipulator::Fly);
  m_renderer = nullptr;

  if(m_resources.m_supportsSmBuiltinsNV)
  {
    VkPhysicalDeviceProperties2 physicalProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
    physicalProperties.pNext = &smProperties;
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &physicalProperties);
    // pseudo heuristic
    // larger GPUs seem better off with lower values
    if(smProperties.shaderSMCount * smProperties.shaderWarpsPerSM > 4096)
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 2;
    else if(smProperties.shaderSMCount * smProperties.shaderWarpsPerSM > 2048 + 1024)
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 4;
    else
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 8;
  }
  else
  {
    m_rendererConfig.usePersistentTraversal = false;
    m_rendererConfig.usePersistentHitDemand = false;
  }

  {
    m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_CLUSTERS_LOD, "Rasterization");

    m_ui.enumAdd(GUI_PARTITIONLAYOUT, int(ScenePartitionConfig::LAYOUT_FIXED_GRID), "fixed grid");
    m_ui.enumAdd(GUI_PARTITIONLAYOUT, int(ScenePartitionConfig::LAYOUT_DERIVED_GRID), "derived grid");
    m_ui.enumAdd(GUI_PARTITIONLAYOUT, int(ScenePartitionConfig::LAYOUT_DERIVED_BVH_MEDIAN), "derived bvh median");
    m_ui.enumAdd(GUI_PARTITIONLAYOUT, int(ScenePartitionConfig::LAYOUT_DERIVED_BVH_SAH), "derived bvh sah");
    m_ui.enumAdd(GUI_PARTITIONLAYOUT, int(ScenePartitionConfig::LAYOUT_DERIVED_QUADTREE), "derived quadtree");
    m_ui.enumAdd(GUI_PARTITIONLAYOUT, int(ScenePartitionConfig::LAYOUT_NODE_REGEX), "node regex");
    m_ui.enumAdd(GUI_BUILDMODE, 0, "default");
    m_ui.enumAdd(GUI_BUILDMODE, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR, "fast build");
    m_ui.enumAdd(GUI_BUILDMODE, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, "fast trace");

    if(!m_resources.m_supportsClusterRaytracing)
    {
      LOGW("WARNING: Cluster raytracing not supported\n");
      if(usesRayTracing())
      {
        m_tweak.renderer = RENDERER_RASTER_CLUSTERS_LOD;
      }
    }
    else
    {
      m_ui.enumAdd(GUI_RENDERER, RENDERER_RAYTRACE_CLUSTERS_LOD, "Ray tracing");
      m_ui.enumAdd(GUI_RENDERER, RENDERER_PATHTRACE_CLUSTERS_LOD, "Path tracing");
      // the raster pass' alpha test interpolates texture coordinates with fragment barycentrics
      if(m_resources.m_supportsBarycentrics)
      {
        m_ui.enumAdd(GUI_RENDERER, RENDERER_HYBRID_CLUSTERS_LOD, "Hybrid");
      }
    }

    {
      for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
      {
        std::string enumStr = fmt::format("{}T_{}V", s_clusterInfos[i].tris, s_clusterInfos[i].verts);
        m_ui.enumAdd(GUI_MESHLET, s_clusterInfos[i].cfg, enumStr.c_str());
      }
    }

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 540, "540p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 720, "720p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1080, "1080p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1440, "1440p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2160, "2160p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1024, "1024 sq");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2048, "2048 sq");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 4096, "4096 sq");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_SHADED, "shaded");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GREY, "grey");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_VIS_BUFFER, "visibility buffer");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_MATERIAL, "material");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GROUP, "cluster groups");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_LOD, "lod levels");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLE, "triangles");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_BLAS, "blas");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_BLAS_REUSE, "blas reuse");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_PARTITION, "partition");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_PARTITION_ACTIVE, "partition active");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DEPTH_ONLY, "depth only (black)");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DISCRETE_LOD, "discrete lod");

    m_ui.enumAdd(GUI_OPTICAL, OPTICAL_MODE_NONE, "None");
    m_ui.enumAdd(GUI_OPTICAL, OPTICAL_MODE_BOX, "Mirror Box");
    m_ui.enumAdd(GUI_OPTICAL, OPTICAL_MODE_SPHERE, "Mirror Sphere");
    m_ui.enumAdd(GUI_OPTICAL, OPTICAL_MODE_LENS, "Lens");
  }

  // Initialize core components

  m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  m_resources.init(app->getDevice(), app->getPhysicalDevice(), app->getInstance(), app->getQueue(0), app->getQueue(1));
  m_frameCapture.init(m_resources, app->getFrameCycleSize(), m_supportsVideoEncode ? &app->getQueue(2) : nullptr, app->isHeadless());
  if(!app->isHeadless())
  {
    // nvapp's hook for DLSS frame generation sees the swapchain image once the UI is drawn
    app->setFrameGenerationCallback(
        [this](VkCommandBuffer cmd, VkImage image, VkImageView view, VkFormat format, VkExtent2D size, uint64_t) {
          if(m_frameCapture.capturesWindow() && (m_frameCapture.config.enabled || m_frameCapture.isActive()))
          {
            m_frameCapture.cmdFrame(cmd, m_app->getFrameCycleIndex(), image, view, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                    format, size, m_app->getFrameSignalSemaphore());
          }
        });
  }

  m_texturesConfig.immutableSampler = m_resources.m_samplerTriLinear;

#if USE_DLSS
  m_resources.m_dlssWaitForInit = m_resources.m_dlssWaitForInit || !m_dlssDelayLoad;

  // NGX init takes seconds, overlap it with scene loading
  if(m_rendererConfig.useDlss)
  {
    m_resources.initDlssAsync(usesRayTracing() ? Resources::DlssMode::eRayReconstruction : Resources::DlssMode::eSuperResolution,
                              m_rendererConfig.dlssQuality);
  }
#endif

  {
    NVVK_CHECK(m_resources.m_samplerPool.acquireSampler(m_imguiSampler));
    NVVK_DBG_NAME(m_imguiSampler);
  }

  m_resources.initFramebuffer({128, 128}, m_tweak.supersample, m_tweak.hbaoQuarter);
  updateImguiImage();

  if(!m_clasAllocatorVis.init(m_resources))
  {
    LOGW("CLAS allocator visualization unavailable\n");
  }

  setFromClusterConfig(m_sceneConfig, m_tweak.clusterConfig);

  if(!m_resources.m_supportsMeshShaderNV)
  {
    m_rendererConfig.useEXTmeshShader = true;
  }

  // Search for default scene if none was provided on the command line
  if(m_sceneFilePathDropNew.empty())
  {
    const std::filesystem::path              exeDirectoryPath   = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> defaultSearchPaths = {
        // regular build
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_DOWNLOAD_DIRECTORY),
        // install build
        std::filesystem::absolute(exeDirectoryPath / "resources"),
    };

    m_sceneFilePathDefault = m_sceneFilePathDropNew = nvutils::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);

    // enforce unique geometries in the sample scene
    m_sceneGridConfig.uniqueGeometries = 1.0f;

    if(m_sceneGridConfig.numCopies == 1)
    {
      if(m_resources.getDeviceLocalHeapSize() >= 8ull * 1024 * 1024 * 1024)
      {
        m_sceneGridConfig.numCopies = 1024;  // 32x32 grid
      }
      else
      {
        m_sceneGridConfig.numCopies = 64;
      }
    }
  }

  if(m_resources.getDeviceLocalHeapSize() >= 8ull * 1024 * 1024 * 1024)
  {
    m_streamingConfig.maxClasMegaBytes     = 2 * 1024;
    m_streamingConfig.maxGeometryMegaBytes = 2 * 1024;
  }
  else
  {
    m_streamingConfig.maxClasMegaBytes     = 1 * 1024;
    m_streamingConfig.maxGeometryMegaBytes = 1 * 1024;
  }

  // command line / config overrides win over the heuristic above
  applyMemoryBudgetArgs();

  m_cameraStringCommandLine = m_cameraString;
  // camera-path provenance is tracked by m_cameraPathsExternal, which the
  // --addcamerapath/--loadcamerapaths handlers set (also for config/sequence
  // parsing that happens after this point), so no snapshot is needed here

  // baseline for onFileDrop, see m_sceneConfigBaseline
  m_sceneConfigBaseline       = m_sceneConfig;
  m_sceneLoaderConfigBaseline = m_sceneLoaderConfig;

  std::filesystem::path newFileDrop = m_sceneFilePathDropNew;
  onFileDrop(newFileDrop);

  resolvePathtraceRenderer();
  m_tweakLast          = m_tweak;
  m_sceneConfigLast    = m_sceneConfig;
  m_sceneConfigEdit    = m_sceneConfig;
  m_rendererConfigLast = m_rendererConfig;
}

void LodClusters::onDetach()
{
  // don't block the close on a running background load; let the OS reclaim everything
  if(m_sceneLoading.load())
  {
    LOGI("Scene load still in progress on window close, exiting immediately\n");
    std::_Exit(0);
  }
#if USE_DLSS
  // NGX calls can't be interrupted, the first feature creation alone takes seconds
  if(m_resources.isDlssLoading())
  {
    LOGI("DLSS load still in progress on window close, exiting immediately\n");
    std::_Exit(0);
  }
#endif

  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(!m_app->isHeadless())
    m_app->setFrameGenerationCallback({});
  m_frameCapture.deinit();
  deinitRenderer();
  deinitScene();

  m_clasAllocatorVis.deinit(m_resources);

  m_resources.m_samplerPool.releaseSampler(m_imguiSampler);
  ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);

  m_resources.deinit();

  m_profilerGpuTimer.deinit();
}

// The streamer needs a resident slot for every geometry's lowest detail group, so fewer groups
// than geometries is not a legal setting. Enforced here rather than only at scene load, because
// the ui, a config or a parameter sequence can lower it at any time.
// `scene` rather than m_scene: this also runs on the loader thread, where the scene is not
// installed yet
void LodClusters::clampStreamingConfig(const Scene& scene)
{
  m_streamingConfig.maxGroups = std::max(m_streamingConfig.maxGroups, uint32_t(scene.getActiveGeometryCount()));
}

void LodClusters::applyMemoryBudgetArgs()
{
  if(memcmp(&m_memoryBudgetArgs, &m_memoryBudgetArgsLast, sizeof(MemoryBudgetArgs)) == 0)
    return;
  m_memoryBudgetArgsLast = m_memoryBudgetArgs;

  const size_t heapMegaBytes = size_t(m_resources.getDeviceLocalHeapSize() / (1024 * 1024));

  // > 0 absolute MiB, < 0 percentage of the device local heap, 0 keeps the current value
  auto resolve = [&heapMegaBytes](int32_t arg, size_t& target) {
    if(arg > 0)
      target = size_t(arg);
    else if(arg < 0 && heapMegaBytes > 0)
      target = std::max(size_t(1), (heapMegaBytes * size_t(-arg)) / 100);
  };

  resolve(m_memoryBudgetArgs.maxTransferMegaBytes, m_streamingConfig.maxTransferMegaBytes);
  resolve(m_memoryBudgetArgs.maxGeometryMegaBytes, m_streamingConfig.maxGeometryMegaBytes);
  resolve(m_memoryBudgetArgs.maxClasMegaBytes, m_streamingConfig.maxClasMegaBytes);
  resolve(m_memoryBudgetArgs.startClasMegaBytes, m_streamingConfig.startClasMegaBytes);
  resolve(m_memoryBudgetArgs.clasGrowMegaBytes, m_streamingConfig.clasGrowMegaBytes);
  resolve(m_memoryBudgetArgs.maxBlasCachingMegaBytes, m_streamingConfig.maxBlasCachingMegaBytes);

  // textures: 0 is a valid value that disables the limit
  size_t textureMegaBytes = size_t(std::max(0, m_memoryBudgetArgs.maxTextureMegaBytes));
  resolve(m_memoryBudgetArgs.maxTextureMegaBytes, textureMegaBytes);
  m_texturesConfig.maxBudgetMiB = uint32_t(textureMegaBytes);

  // the allocator requires the initial clas allocation to fit the budget
  m_streamingConfig.startClasMegaBytes = std::min(m_streamingConfig.startClasMegaBytes, m_streamingConfig.maxClasMegaBytes);

  const MemoryBudgetArgs defaultArgs;
  if(memcmp(&m_memoryBudgetArgs, &defaultArgs, sizeof(MemoryBudgetArgs)) != 0)
  {
    LOGI(
        "Memory budgets (device local heap %zu MiB):\n"
        "  geometry    %zu MiB\n"
        "  clas        %zu MiB (start %zu, grow %zu)\n"
        "  blas cache  %zu MiB\n"
        "  transfer    %zu MiB\n"
        "  textures    %u MiB\n",
        heapMegaBytes, m_streamingConfig.maxGeometryMegaBytes, m_streamingConfig.maxClasMegaBytes,
        m_streamingConfig.startClasMegaBytes, m_streamingConfig.clasGrowMegaBytes,
        m_streamingConfig.maxBlasCachingMegaBytes, m_streamingConfig.maxTransferMegaBytes, m_texturesConfig.maxBudgetMiB);
  }
}

// Settings that come with the app rather than with the model: after the scene's own
// config file (if any) was parsed, we also run "scene_overrides/<scene name>.cfg" when
// present. Must happen before the actual scene load, as it can alter loading behavior.
void LodClusters::applySceneOverrides(const std::filesystem::path& filePath)
{
  if(filePath.empty())
    return;

  const std::filesystem::path              exeDirectoryPath = nvutils::getExecutablePath().parent_path();
  const std::vector<std::filesystem::path> searchPaths      = {
      // regular build
      std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "scene_overrides"),
      // install build
      std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "scene_overrides"),
      std::filesystem::absolute(exeDirectoryPath / "scene_overrides"),
  };

  const std::filesystem::path overrideFileName = std::filesystem::path(filePath).stem().concat(".cfg");

  std::filesystem::path overrideFilePath;
  for(const std::filesystem::path& searchPath : searchPaths)
  {
    std::filesystem::path candidate = searchPath / overrideFileName;
    if(std::filesystem::exists(candidate))
    {
      overrideFilePath = candidate;
      break;
    }
  }

  if(overrideFilePath.empty())
    return;

  std::string overrideFilePathString = nvutils::utf8FromPath(overrideFilePath);

  LOGI("Loading scene override config: %s\n", overrideFilePathString.c_str());

  std::vector<const char*> args;
  args.push_back("--configfile");
  args.push_back(overrideFilePathString.c_str());

  // an override must not change which scene is loaded
  std::filesystem::path oldFilePath = m_sceneFilePathDropNew;

  m_info.parameterParser->parse(std::span(args), false, {}, {}, true);

  m_sceneFilePathDropNew = oldFilePath;

  // budgets from the override may be relative to the device local heap
  applyMemoryBudgetArgs();

  if(!m_cameraStringCommandLine.empty())
  {
    // override from command-line
    m_cameraString = m_cameraStringCommandLine;
  }
}

void LodClusters::saveCacheFile()
{
  if(m_scene)
  {
    m_scene->saveCache();
  }
}

void LodClusters::onFileDrop(const std::filesystem::path& filePath)
{
  if(filePath.empty())
    return;

  if(filePath.extension() == ".hdr")
  {
    m_envHdriFile      = filePath;
    m_tweak.useEnvHdri = true;
    return;
  }

  if(!m_sceneLoadFromConfig)
  {
    // avoid certain state to affect the new scene
    if(!m_sceneFilePathDropLast.empty() && filePath != m_sceneFilePathDropLast)
    {
      // reset grid parameter (in case scene is too large to be replicated)
      m_sceneGridConfig.numCopies        = 1;
      m_sceneGridConfig.uniqueGeometries = 0.0f;

      m_cameraSpeed             = 0;
      m_cameraClipNear          = 0;
      m_cameraClipFar           = 0;
      m_cameraGltf              = 0;
      m_cameraString            = {};
      m_cameraStringLast        = {};
      m_cameraStringCommandLine = {};

      // drop what the previous scene's config file and scene_overrides had set
      m_sceneConfig       = m_sceneConfigBaseline;
      m_sceneLoaderConfig = m_sceneLoaderConfigBaseline;
    }
    m_sceneFilePathDropLast = filePath;
    m_sceneFilePathDropNew  = filePath;

    addToRecentFiles(filePath);
  }

  if(filePath.extension() == ".cfg")
  {
    std::string filePathString = nvutils::utf8FromPath(filePath);

    LOGI("Loading config: %s\n", filePathString.c_str());

    std::vector<const char*> args;
    args.push_back("--configfile");
    args.push_back(filePathString.c_str());

    std::filesystem::path oldFilePath = m_sceneFilePathDropNew;

    // config parsing might change m_sceneFilePathDropNew
    // and m_cameraString
    m_info.parameterParser->parse(std::span(args), false, {}, {}, true);

    applySceneOverrides(filePath);

    // budgets from the config may be relative to the device local heap
    applyMemoryBudgetArgs();

    if(!m_cameraStringCommandLine.empty())
    {
      // override from command-line
      m_cameraString = m_cameraStringCommandLine;
    }

    if(m_sceneFilePathDropNew != m_sceneFilePathDropLast)
    {
      bool oldState = m_sceneLoadFromConfig;

      m_sceneLoadFromConfig             = true;
      std::filesystem::path cfgFilePath = m_sceneFilePathDropNew;
      onFileDrop(cfgFilePath);

      m_sceneLoadFromConfig  = oldState;
      m_sceneFilePathDropNew = oldFilePath;
    }

    return;
  }

  if(!m_sceneLoadFromConfig)
  {
    // coming from a config file, the overrides were already applied for that config's name
    applySceneOverrides(filePath);
  }

  LOGI("Loading model: %s\n", nvutils::utf8FromPath(filePath).c_str());
  deinitRenderer();

  initScene(filePath, m_sceneCacheSuffix, false);
}

void LodClusters::doProcessingOnly()
{
  setFromClusterConfig(m_sceneConfig, m_tweak.clusterConfig);
  assert(m_app == nullptr);

  if(m_sceneFilePathDropNew.extension() == ".cfg")
  {
    std::string filePathString = nvutils::utf8FromPath(m_sceneFilePathDropNew);

    LOGI("Loading config: %s\n", filePathString.c_str());

    std::vector<const char*> args;
    args.push_back("--configfile");
    args.push_back(filePathString.c_str());

    std::filesystem::path oldFilePath = m_sceneFilePathDropNew;

    // config parsing might change m_sceneFilePathDropNew
    // and m_cameraString
    m_info.parameterParser->parse(std::span(args), false, {}, {}, true);

    applySceneOverrides(oldFilePath);
  }
  else
  {
    applySceneOverrides(m_sceneFilePathDropNew);
  }

  m_scene = std::make_unique<Scene>();
  m_scene->init(m_sceneFilePathDropNew, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, false);
}

// The readback counters are single frame samples, so a benchmark step needs their mean over
// exactly the frames the reported timers cover: the last `numAveraged` frames since the
// profiler's last reset. Once `sequenceaverages` is reached that is a sliding window, the
// count stops growing but every frame still adds a sample.
void LodClusters::accumulateFrameStats()
{
  if(!m_renderer)
    return;

  nvutils::ProfilerTimeline::TimerInfo info;
  std::string                          apiName;
  bool valid = m_profilerTimeline->getFrameTimerInfo("Trace", info, apiName) && info.numAveraged > 0;
  if(!valid || info.numAveraged < m_statsNumAveragedLast)
  {
    // profiler averages were thrown away, so ours must restart too. A step that resets mid way
    // is a step whose numbers are worth less, so that is counted.
    if(valid)
      m_statsProfilerResets++;
    m_statsSampleCount = 0;
    m_statsTotal       = {};
  }
  m_statsNumAveragedLast = valid ? info.numAveraged : 0;
  if(!valid)
    return;

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  if(m_rendererConfig.useHybridMatchCheck)
  {
    // edge pixels round differently in raster and ray tracing, a mismatch shows as a patch
    uint32_t pixels = 0;
    for(uint32_t i = 20; i <= 30; i++)
      pixels += readback.debugA[i];
    uint32_t off = readback.debugA[21] + readback.debugA[22] + readback.debugA[25] + readback.debugA[26]
                   + readback.debugA[27] + readback.debugA[28];
    if(pixels && off > 8 + pixels / 10000)
    {
      m_hybridMatchFailedFrames++;
      m_hybridMatchWorst = std::max(m_hybridMatchWorst, off);
    }
  }

  Renderer::FrameStats fs;
  fs.frames                      = 1;
  fs.traversalTasks              = readback.numTraversalTasks;
  fs.renderClusters              = readback.numRenderClusters;
  fs.blasBuilds                  = readback.numBlasBuilds;
  fs.ptlasUpdates                = readback.numPtlasUpdates;
  fs.ptlasUpdatesPeak            = readback.numPtlasUpdates;
  fs.ptlasDirtyPartitions        = readback.numPtlasDirtyPartitions;
  fs.ptlasDirtyInstances         = readback.numPtlasDirtyInstances;
  fs.activePartitions            = readback.numActivePartitions;
  fs.activeInstances             = readback.numActiveInstances;
  fs.partitionInvalidateAllCount = readback.partitionInvalidateAllCount ? 1 : 0;
  fs.instanceHitScaleChanges     = readback.numInstanceHitScaleChanges;

  if(m_renderScene && m_renderScene->useStreaming)
  {
    const SceneStreaming::CachedBlasPatchCounts& pc = m_renderScene->sceneStreaming.getFramePatchCounts();
    fs.patchLowerDetail                             = pc.lowerDetail;
    fs.patchHigherDetail                            = pc.higherDetail;
    fs.patchInvalidateOnly                          = pc.invalidateOnly;
    fs.patchDeferred                                = pc.deferred;
    fs.patchUnsafeFreed                             = pc.unsafeFreed;
    fs.patchRetired                                 = pc.retired;
    fs.pinnedUnloads                                = m_renderScene->sceneStreaming.getFramePinnedUnloads();
    fs.pinnedDeferredUnloads                        = m_renderScene->sceneStreaming.getFramePinnedDeferredUnloads();
  }

  m_statsSamples[m_statsSampleCount % m_statsSamples.size()] = fs;
  m_statsSampleCount++;
  m_statsTotal += fs;

  // averaging all frames outgrows the samples, but then covers everything since the reset
  Renderer::FrameStats window;
  if(info.numAveraged > m_statsSamples.size())
  {
    window = m_statsTotal;
  }
  else
  {
    uint32_t count = std::min(info.numAveraged, m_statsSampleCount);
    for(uint32_t i = 0; i < count; i++)
    {
      window += m_statsSamples[(m_statsSampleCount - 1 - i) % m_statsSamples.size()];
    }
  }
  window.profilerResets         = m_statsProfilerResets;
  m_renderer->getFrameStatsRW() = window;
}

void LodClusters::appendMemoryReport(std::string& message)
{
  Renderer::ResourceUsageInfo resourceActual   = m_renderer->getResourceUsage(false);
  Renderer::ResourceUsageInfo resourceReserved = m_renderer->getResourceUsage(true);
  bool                        hasTextures      = m_renderScene && m_renderScene->sceneTextures.hasTextures();
  size_t                      textureMemBytes  = hasTextures ? m_renderScene->sceneTextures.getTextureMemBytes() : 0;
  VkDeviceSize                renderTargetMemBytes = m_resources.getFramebufferMemBytes();
  message += fmt::format("Memory; Actual; Reserved;\n");
  if(hasTextures)
  {
    message += fmt::format("Textures; {}; {};\n", textureMemBytes, textureMemBytes);
  }
  message += fmt::format("Geometry; {}; {};\n", resourceActual.geometryMemBytes, resourceReserved.geometryMemBytes);
  if(usesRayTracing())
  {
    message += fmt::format("CLAS; {}; {};\n", resourceActual.rtClasMemBytes, resourceReserved.rtClasMemBytes);
    message += fmt::format("BLAS; {}; {};\n", resourceActual.rtBlasMemBytes, resourceReserved.rtBlasMemBytes);
    message += fmt::format("TLAS; {}; {};\n", resourceActual.rtTlasMemBytes, resourceReserved.rtTlasMemBytes);
  }

  message += fmt::format("Operations; {}; {};\n", resourceActual.operationsMemBytes, resourceReserved.operationsMemBytes);
  message += fmt::format("Render Targets; {}; {};\n", renderTargetMemBytes, renderTargetMemBytes);
  message += fmt::format("Total; {}; {};\n", resourceActual.getTotalSum() + textureMemBytes + renderTargetMemBytes,
                         resourceReserved.getTotalSum() + textureMemBytes + renderTargetMemBytes);
  if(m_renderScene->useStreaming)
  {
    StreamingStats stats;
    m_renderScene->sceneStreaming.getStats(stats);
    message += fmt::format("Resident; Actual; Reserved;\n");
    message += fmt::format("Groups; {}; {};\n", stats.residentGroups, stats.maxGroups);
    message += fmt::format("Clusters; {}; {};\n", stats.residentClusters, stats.maxClusters);
    message += fmt::format("Triangles; {};\n", stats.residentTriangles);
    message += fmt::format("Last Loads; {};\n", stats.loadCount);
    message += fmt::format("Last Unloads; {};\n", stats.unloadCount);
    message += fmt::format("Last Load Requests; {};\n", stats.loadRequestCount);
    message += fmt::format("Last Load Fails; {}; {}; {}; {}; {};\n", stats.couldNotTransfer, stats.couldNotAllocateClas,
                           stats.couldNotAllocateGroup, stats.couldNotStore, stats.uncompletedLoadCount);
    message += fmt::format("Frame Loads; {}; {}; {}; {};\n", stats.frameLoadCount, stats.frameLoadClusters,
                           stats.frameLoadDeviceBytes, stats.frameLoadStoredBytes);
    if(m_rendererConfig.useBlasCaching)
    {
      message += fmt::format("Cached BLAS; {};\n", stats.cachedBlasCount);
      message += fmt::format("Cached BLAS memory; {}; {};\n", stats.usedCachedBlasBytes, stats.reservedCachedBlasBytes);
    }
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);
  message += fmt::format("Traversal; Actual; Reserved;\n");
  message += fmt::format("Traversal Tasks; {}; {};\n", readback.numTraversalTasks, m_renderer->getMaxTraversalTasks());
  if(m_rendererConfig.useHitStreaming)
  {
    // the demand BFS shares the traversalNodeInfos array, so the reserve is the same
    message += fmt::format("Hit Demand Tasks; {}; {};\n", readback.numHitDemandTasks, m_renderer->getMaxTraversalTasks());
  }
  message += fmt::format("Traversal Clusters; {}; {};\n", readback.numRenderClusters, m_renderer->getMaxRenderClusters());
  if(m_tweak.renderer == RENDERER_HYBRID_CLUSTERS_LOD)
  {
    // first pass, rejected by last frame's hiz, second pass
    message += fmt::format("Hybrid Clusters; {}; {}; {}; {};\n", readback.numHybridClusters, readback.numHybridRejectClusters,
                           readback.numHybridClusters2, m_renderer->getMaxRenderClusters());
    if(m_rendererConfig.useComputeRaster)
    {
      // both passes
      message += fmt::format("Hybrid SW Clusters; {};\n", readback.numHybridClustersSW);
    }
  }
  message += fmt::format("BLAS builds; {}; {};\n", readback.numBlasBuilds, m_renderer->getMaxBlasBuilds());
  if(usesBlasDedup())
  {
    message += fmt::format("BLAS candidates; {};\n", readback.numBlasCandidates);
    message += fmt::format("BLAS deduped; {};\n", readback.numBlasDeduped);
    message += fmt::format("BLAS dedup probes; {};\n", readback.numBlasDedupProbes);
    message += fmt::format("BLAS dedup missed; {};\n", readback.numBlasDedupMissed);
  }
  // dev only, written while STREAM_DEBUG_RESIDENCY is set
  if(readback.debugA[0] + readback.debugA[1] + readback.debugA[2] + readback.debugA[3] + readback.debugA[4])
  {
    message += fmt::format("Residency Clusters; {}; {}; {}; {}; {};\n", readback.debugA[0], readback.debugA[1],
                           readback.debugA[2], readback.debugA[3], readback.debugA[4]);
  }
  // dev only, written while DEBUG_CULLED_GROUP_LOD is set
  if(readback.debugA[5] + readback.debugA[6])
  {
    message += fmt::format("Occlusion Inherit Mismatch; {}; {};\n", readback.debugA[5], readback.debugA[6]);
  }
  // written with --hybridmatchcheck, failures are reported once per report
  if(m_hybridMatchFailedFrames)
  {
    LOGE("hybridmatchcheck error: %u frames with up to %u mismatching pixels\n", m_hybridMatchFailedFrames, m_hybridMatchWorst);
    m_hybridMatchFailedFrames = 0;
    m_hybridMatchWorst        = 0;
  }
  if(readback.debugA[23])
  {
    message += fmt::format("Hybrid Match; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {};\n", readback.debugA[20],
                           readback.debugA[21], readback.debugA[22], readback.debugA[23], readback.debugA[24],
                           readback.debugA[25], readback.debugA[26], readback.debugA[27], readback.debugA[28],
                           readback.debugA[29], readback.debugA[30]);
  }
  // what the renderer applies, the raster renderer has no partitions
  const RendererConfig& activeConfig  = m_renderer->getConfig();
  const bool            hasPartitions = activeConfig.usePartitions && m_scene && !m_scene->m_partitions.empty();
  const bool            hasPtlas      = hasPartitions && activeConfig.usePtlas;
  const bool            hasEarlyOut   = hasPartitions && activeConfig.usePartitionEarlyOut;

  if(hasPartitions)
  {
    if(hasPtlas)
    {
      message += fmt::format("Ptlas Updates; {}; {};\n", readback.numPtlasUpdates, m_scene->m_instances.size());
      message +=
          fmt::format("Ptlas Dirty Partitions; {}; {};\n", readback.numPtlasDirtyPartitions, m_scene->m_partitions.size());
      message += fmt::format("Ptlas Dirty Instances; {}; {};\n", readback.numPtlasDirtyInstances, m_scene->m_instances.size());
    }
    if(hasEarlyOut)
    {
      // what the early-out left for the per-instance kernels to do
      message += fmt::format("Active Partitions; {}; {};\n", readback.numActivePartitions, m_scene->m_partitions.size());
      message += fmt::format("Active Instances; {}; {};\n", readback.numActiveInstances, m_scene->m_instances.size());
    }

    // block means; the single frame samples above swing hard on a forced revalidation frame
    const Renderer::FrameStats& fs = m_renderer->getFrameStats();
    message += fmt::format("Mean; Value; Frames;\n");
    message += fmt::format("Profiler Resets; {};\n", fs.profilerResets);
    message += fmt::format("Frames InvalidateAll; {}; {};\n", fs.partitionInvalidateAllCount, fs.frames);
    message += fmt::format("Patches LowerDetail; {};\n", fs.patchLowerDetail);
    message += fmt::format("Patches HigherDetail; {};\n", fs.patchHigherDetail);
    message += fmt::format("Patches InvalidateOnly; {};\n", fs.patchInvalidateOnly);
    message += fmt::format("Patches Deferred; {};\n", fs.patchDeferred);
    message += fmt::format("Patches UnsafeFreed; {};\n", fs.patchUnsafeFreed);
    message += fmt::format("Patches Retired; {};\n", fs.patchRetired);
    message += fmt::format("Pinned Unloads; {}; {};\n", fs.pinnedUnloads, fs.pinnedDeferredUnloads);
    message += fmt::format("Mean Traversal Tasks; {}; {};\n", fs.mean(fs.traversalTasks), fs.frames);
    message += fmt::format("Mean BLAS builds; {}; {};\n", fs.mean(fs.blasBuilds), fs.frames);
    if(hasPtlas)
    {
      message += fmt::format("Mean Ptlas Updates; {}; {};\n", fs.mean(fs.ptlasUpdates), fs.frames);
      message += fmt::format("Mean Ptlas Dirty Partitions; {}; {};\n", fs.mean(fs.ptlasDirtyPartitions), fs.frames);
      message += fmt::format("Mean Ptlas Dirty Instances; {}; {};\n", fs.mean(fs.ptlasDirtyInstances), fs.frames);
    }
    if(hasEarlyOut)
    {
      message += fmt::format("Mean Active Partitions; {}; {};\n", fs.mean(fs.activePartitions), fs.frames);
      message += fmt::format("Mean Active Instances; {}; {};\n", fs.mean(fs.activeInstances), fs.frames);
    }
  }
  if(m_rendererConfig.useHitStreaming && m_rendererConfig.useHitStreamingLod)
  {
    const Renderer::FrameStats& fs = m_renderer->getFrameStats();
    message += fmt::format("Hit Scale Changes; {}; {};\n", readback.numInstanceHitScaleChanges, m_scene->m_instances.size());
    message += fmt::format("Mean Hit Scale Changes; {}; {};\n", fs.mean(fs.instanceHitScaleChanges), fs.frames);
  }
  if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD && m_rendererConfig.useCulling
     && m_rendererConfig.useTwoPassCulling && m_rendererConfig.useTwoPassRejectLists)
  {
    // what the first cull pass handed to the second one, the node and group lists
    // are capped by the traversal task limit, the instance list by the scene
    message += fmt::format("Reject Nodes; {}; {};\n", readback.numRejectNodes, m_renderer->getMaxTraversalTasks());
    message += fmt::format("Reject Groups; {}; {};\n", readback.numRejectGroups, m_renderer->getMaxTraversalTasks());
    message += fmt::format("Reject Instances; {};\n", readback.numRejectInstances);
  }

  if(m_rendererConfig.useRenderStats)
  {
    message += fmt::format("Enqueued; Actual;\n");
    message += fmt::format("Enqueued Tasks; {};\n", readback.numTraversedTasks);
    message += fmt::format("Enqueued Clusters; {};\n", readback.numRenderedClusters);
    if(m_rendererConfig.useDiscreteLod)
    {
      message += fmt::format("Discrete Instances; {};\n", readback.numDiscreteInstances);
    }
    if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)
    {
      message += fmt::format("Enqueued Triangles; {};\n", readback.numRenderedTriangles);
      if(m_rendererConfig.useCulling && m_rendererConfig.usePrimitiveCulling)
      {
        message += fmt::format("Rastered Triangles; {};\n", readback.numRasteredTriangles);
        message += fmt::format("Rastered TrianglesSW; {};\n", readback.numRasteredTrianglesSW);
      }
    }
  }
}

void LodClusters::logMemoryReport(const std::string& label)
{
  std::string message;
  message += fmt::format("MemoryReport {} = {{ \n", label);
  if(m_renderer)
  {
    appendMemoryReport(message);
  }
  message += fmt::format("}}\n");

  nvutils::Logger::getInstance().log(nvutils::Logger::eSTATS, "%s", message.c_str());
}

void LodClusters::parameterSequenceCallback(const nvutils::ParameterSequencer::State& state)
{
  logMemoryReport(fmt::format("{} \"{}\"", state.index, state.description));

  // The samples restart with the profiler's own reset, only the reset count is per step.
  m_statsProfilerResets = 0;
  if(m_renderer)
  {
    m_renderer->resetFrameStats();
  }

  saveScreenshot(m_sequenceScreenshotMode, fmt::format("screenshot_{}_{}.jpg", state.index, state.description));
}

void LodClusters::saveScreenshot(ScreenshotMode mode, const std::string& filename, int quality)
{
  if(mode == SCREENSHOT_OFF)
    return;

  // headless has no window content
  if(m_app->isHeadless())
  {
    mode = SCREENSHOT_VIEWPORT;
  }

  if(mode == SCREENSHOT_WINDOW)
  {
    m_app->saveScreenShot(std::filesystem::path(filename), quality);
  }
  else if(mode == SCREENSHOT_VIEWPORT)
  {
    // the displayed image: resolved at window size, otherwise the target (fixed resolution)
    const Resources::FrameBuffer& fb = m_resources.m_frameBuffer;
    m_app->saveImageToFile(fb.useResolved ? fb.imgColorResolved.image : fb.imgColorLdr.image,
                           fb.useResolved ? fb.windowSize : fb.targetSize, filename, quality,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
}

const LodClusters::ClusterInfo LodClusters::s_clusterInfos[NUM_CLUSTER_CONFIGS] = {
    {64, 64, CLUSTER_64T_64V},     {64, 128, CLUSTER_64T_128V},   {64, 192, CLUSTER_64T_192V},
    {96, 96, CLUSTER_96T_96V},     {128, 128, CLUSTER_128T_128V}, {128, 256, CLUSTER_128T_256V},
    {256, 256, CLUSTER_256T_256V},
};

LodClusters::ClusterConfig LodClusters::findSceneClusterConfig(const SceneConfig& sceneConfig)
{
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    const ClusterInfo& entry = s_clusterInfos[i];
    if(sceneConfig.clusterTriangles <= entry.tris && sceneConfig.clusterVertices <= entry.verts)
    {
      return entry.cfg;
    }
  }

  return CLUSTER_256T_256V;
}

void LodClusters::setFromClusterConfig(SceneConfig& sceneConfig, ClusterConfig clusterConfig)
{
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    if(s_clusterInfos[i].cfg == clusterConfig)
    {
      sceneConfig.clusterTriangles = s_clusterInfos[i].tris;
      sceneConfig.clusterVertices  = s_clusterInfos[i].verts;
      return;
    }
  }
}

// `scene` rather than m_scene: this also runs on the loader thread, where the scene is not
// installed yet
void LodClusters::updatedSceneGrid(const Scene& scene)
{
  {
    glm::vec3 gridExtent = scene.m_gridBbox.hi - scene.m_gridBbox.lo;
    float     gridRadius = glm::length(gridExtent) * 0.5f;

    glm::vec3 modelExtent = scene.m_bbox.hi - scene.m_bbox.lo;
    float     modelRadius = glm::length(modelExtent) * 0.5f;

    bool bigScene = scene.m_isBig;

    if(!m_cameraSpeed)
      m_info.cameraManipulator->setSpeed(modelRadius * (bigScene ? 0.0025f : 0.25f));

    if(m_cameraString.empty())
    {
      glm::vec2 clip((bigScene ? 0.0001f : 0.01F) * modelRadius,
                     bigScene ? gridRadius * 1.2f : std::max(50.0f * modelRadius, gridRadius * 1.2f));
      if(m_cameraClipNear)
        clip.x = m_cameraClipNear < 0 ? -m_cameraClipNear : m_cameraClipNear * gridRadius * 2.0f;
      if(m_cameraClipFar)
        clip.y = m_cameraClipFar < 0 ? -m_cameraClipFar : m_cameraClipFar * gridRadius * 2.0f;
      m_info.cameraManipulator->setClipPlanes(clip);
    }
  }

  if(m_tweak.autoSharing)
  {
    m_rendererConfig.useBlasSharing = (scene.m_instances.size() > scene.getActiveGeometryCount() * 3);
  }
}

void LodClusters::updateDerivedStreamingConfig()
{
  // the hybrid renderer rasterizes, ray tracing alone keeps positions only for the clas build
  m_streamingConfig.keepPersistentPositions = m_streamingKeepPositions || m_tweak.renderer == RENDERER_HYBRID_CLUSTERS_LOD;
}

// --pathtrace predates the path tracing renderer type. Given on a command line or in a sequence
// it still switches ray tracing to path tracing and back, a renderer switch only updates it.
void LodClusters::resolvePathtraceRenderer()
{
  if(uint32_t(m_tweak.renderer) > RENDERER_HYBRID_CLUSTERS_LOD)
  {
    LOGW("--renderer %d is unknown, using ray tracing\n", int(m_tweak.renderer));
    m_tweak.renderer = RENDERER_RAYTRACE_CLUSTERS_LOD;
  }
  if(m_tweak.renderer == RENDERER_HYBRID_CLUSTERS_LOD && !m_resources.m_supportsBarycentrics)
  {
    LOGW("the hybrid renderer needs fragment shader barycentrics, using path tracing\n");
    m_tweak.renderer = RENDERER_PATHTRACE_CLUSTERS_LOD;
  }
  if(m_pathtraceParsed)
  {
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD || m_tweak.renderer == RENDERER_PATHTRACE_CLUSTERS_LOD)
    {
      m_tweak.renderer = m_rendererConfig.usePathtrace ? RENDERER_PATHTRACE_CLUSTERS_LOD : RENDERER_RAYTRACE_CLUSTERS_LOD;
    }
    m_pathtraceParsed = false;
  }
  else if(m_tweak.renderer != m_tweakLast.renderer)
  {
    m_rendererConfig.usePathtrace = usesPathTracing();
  }
}

void LodClusters::handleChanges()
{
  if(m_sceneLoading)
    return;

  resolvePathtraceRenderer();

  updateDerivedStreamingConfig();

  // install what the loader thread completed, everything else about the scene was already set up
  // on that thread, only the pointer handover is deferred to here
  if(m_scenePending)
  {
    m_scene = std::move(m_scenePending);
  }

  // budgets arrive as absolute or heap-relative arguments, so they need resolving before the
  // streaming config below is compared. Without this a budget that changes after the scene was
  // loaded (a parameter sequence, most of all) would be parsed and then silently ignored.
  applyMemoryBudgetArgs();
  if(m_scene)
  {
    clampStreamingConfig(*m_scene);
  }

  if(m_scene && m_sceneLoaderConfig != m_sceneLoaderConfigLast)
  {
    std::filesystem::path filePath = m_sceneFilePathDropLast;
    onFileDrop(filePath);
  }

  if(m_sceneFilePathDropLast != m_sceneFilePathDropNew)
  {
    std::filesystem::path newFilePath = m_sceneFilePathDropNew;
    onFileDrop(newFilePath);
  }

  // Load the per-scene camera paths file once a (new) scene has finished loading.
  // Done here on the main thread (not in the loader thread's setSceneCamera) so
  // it does not race the UI / playback reads of m_cameraPaths. Explicitly provided
  // paths (--addcamerapath/--loadcamerapaths, via command line, config, or sequence)
  // are global and take precedence, so the per-scene file is only used without them.
  if(m_scene && m_sceneFilePath != m_cameraPathsLoadedScene)
  {
    m_cameraPathsLoadedScene = m_sceneFilePath;

    if(!m_cameraPathsExternal)
    {
      m_cameraPaths.clear();
      m_cameraPathActive   = -1;
      m_cameraPathEditKey  = -1;
      m_cameraPathPlayback = CAMERA_PATH_STOP;

      std::filesystem::path pathsFile = sceneCameraPathsFile(m_sceneFilePath);
      if(std::filesystem::exists(pathsFile) && loadCameraPaths(pathsFile, m_cameraPaths))
      {
        m_cameraPathActive = m_cameraPaths.empty() ? -1 : 0;
        LOGI("Loaded %zu camera path(s) from %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(pathsFile).c_str());
      }
    }
  }

  if(!m_resources.m_supportsMeshShaderNV)
  {
    m_rendererConfig.useEXTmeshShader = true;
  }

  if(m_rendererConfig.useBlasSharing && m_scene && m_scene->m_instances.size() > (1 << 27))
  {
    m_rendererConfig.useBlasSharing = false;
  }

  if((m_frameConfig.visualize == VISUALIZE_VIS_BUFFER || m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY)
     && m_rendererConfig.useShading)
  {
    m_rendererConfig.useShading = false;
  }
  if(!(m_frameConfig.visualize == VISUALIZE_VIS_BUFFER || m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY)
     && !m_rendererConfig.useShading)
  {
    m_rendererConfig.useShading = true;
  }
  m_rendererConfig.useDepthOnly = m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY;


  // the hybrid renderer's raster passes only write ids, shading is the path tracer's
  if(m_rendererConfig.useComputeRaster
     && (!m_rendererConfig.useCulling || (m_rendererConfig.useShading && m_tweak.renderer != RENDERER_HYBRID_CLUSTERS_LOD)))
  {
    m_rendererConfig.useComputeRaster = false;
  }


  bool dlssLoaded = false;
#if USE_DLSS
  // switch to DLSS once the async init of the renderer's mode finished
  Resources::DlssMode dlssWanted = usesRayTracing() ? Resources::DlssMode::eRayReconstruction : Resources::DlssMode::eSuperResolution;
  dlssLoaded = m_resources.pollDlssInit() == dlssWanted && m_rendererConfig.useDlss
               && m_resources.m_frameBuffer.dlssMode == Resources::DlssMode::eNone;

  if(m_resources.m_dlssEvaluateFailed)
  {
    // re-initializes the renderer natively
    m_rendererConfig.useDlss         = false;
    m_resources.m_dlssEvaluateFailed = false;
  }
#endif

  bool frameBufferChanged = false;
  // a change of only the HBAO resolution leaves the change id alone
  const bool supersampleChanged = tweakChanged(m_tweak.supersample);
  const bool hbaoQuarterChanged = tweakChanged(m_tweak.hbaoQuarter);
  if(supersampleChanged || hbaoQuarterChanged)
  {
    const size_t fboChangeID = m_resources.m_fboChangeID;
    m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoQuarter);
    if(fboChangeID != m_resources.m_fboChangeID)
    {
      updateImguiImage();
      frameBufferChanged = true;
    }
  }

  bool shaderChanged = false;
  if(m_reloadShaders)
  {
    shaderChanged   = true;
    m_reloadShaders = false;
    // shader compile errors are a common cause of a failed render-scene init, so a reload is the
    // way to retry after fixing them
    m_renderSceneInitAllowed = true;
  }

  bool envHdriChanged = false;
  if(m_envHdriFile != m_envHdriFileLast)
  {
    envHdriChanged = true;

    deinitRenderer();
    m_resources.loadEnvHdri(m_envHdriFile);
    m_envHdriFileLast = m_envHdriFile;
  }

  bool sceneChanged = false;
  if(memcmp(&m_sceneConfig, &m_sceneConfigLast, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;

    deinitRenderer();
    initScene(m_sceneFilePath, m_scene->m_cacheSuffix, true);
  }

  if(!m_cameraString.empty() && m_cameraString != m_cameraStringLast)
  {
    applyCameraString();
  }

  bool sceneGridChanged   = false;
  bool renderSceneChanged = false;
  if(m_scene)
  {
    if(m_renderScenePending && m_renderSceneGeometryPending)
    {
      // The async loader thread built m_renderScenePending and uploaded its textures (reported in the
      // busy popup). Publish it now and finish the GPU geometry setup here on the main thread; the
      // textures stay as loaded.
      m_renderSceneGeometryPending = false;
      m_renderScene                = std::move(m_renderScenePending);

      deinitRenderer();
      initRenderSceneGeometry();
      renderSceneChanged = true;
    }
    else if(!m_renderScene && m_renderSceneInitAllowed)
    {
      // async loading might us get into this state
      // pretend scene grid changed to re-init renderscene
      sceneGridChanged = true;
    }

    if(!sceneChanged && memcmp(&m_sceneGridConfig, &m_sceneGridConfigLast, sizeof(m_sceneGridConfig)))
    {
      sceneGridChanged = true;

      deinitRenderer();
      m_scene->updateSceneGrid(m_sceneGridConfig);
      updatedSceneGrid(*m_scene);
      m_resources.clearReadbackData();
    }

    bool streamingChanged = tweakChanged(m_tweak.useStreaming)
                            || (memcmp(&m_streamingConfig, &m_streamingConfigLast, sizeof(m_streamingConfig)))
                            || (memcmp(&m_texturesConfig, &m_texturesConfigLast, sizeof(m_texturesConfig)));
    if(sceneGridChanged || streamingChanged)
    {
      if(!sceneChanged || !sceneGridChanged)
      {
        deinitRenderer();
      }

      renderSceneChanged = true;
      deinitRenderScene();
      initRenderScene();

      if(streamingChanged)
      {
        m_streamHistoryMax = 0;
      }
    }

    if(sceneChanged || shaderChanged || renderSceneChanged || envHdriChanged || tweakChanged(m_tweak.renderer)
       || tweakChanged(m_tweak.supersample)
#if USE_DLSS
       || rendererCfgChanged(m_rendererConfig.useDlss) || rendererCfgChanged(m_rendererConfig.dlssQuality) || dlssLoaded
#endif
       || rendererCfgChanged(m_rendererConfig.flipWinding) || rendererCfgChanged(m_rendererConfig.useDebugVisualization)
       || rendererCfgChanged(m_rendererConfig.useCulling) || rendererCfgChanged(m_rendererConfig.forceTwoSided)
       || rendererCfgChanged(m_rendererConfig.useSorting) || rendererCfgChanged(m_rendererConfig.numRenderClusterBits)
       || rendererCfgChanged(m_rendererConfig.numTraversalTaskBits) || rendererCfgChanged(m_rendererConfig.useShading)
       || rendererCfgChanged(m_rendererConfig.useBlasSharing) || rendererCfgChanged(m_rendererConfig.useRenderStats)
       || rendererCfgChanged(m_rendererConfig.useBlasMerging) || rendererCfgChanged(m_rendererConfig.useBlasCaching)
       || rendererCfgChanged(m_rendererConfig.useBlasDedup) || rendererCfgChanged(m_rendererConfig.useEXTmeshShader)
       || rendererCfgChanged(m_rendererConfig.useComputeRaster) || rendererCfgChanged(m_rendererConfig.usePrimitiveCulling)
       || (m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD
           && (rendererCfgChanged(m_rendererConfig.useTwoPassCulling) || rendererCfgChanged(m_rendererConfig.useTwoPassRejectLists)))
       || rendererCfgChanged(m_rendererConfig.useDepthOnly) || rendererCfgChanged(m_rendererConfig.useCulledGroupLod)
       || rendererCfgChanged(m_rendererConfig.useHybridMatchCheck) || rendererCfgChanged(m_rendererConfig.usePersistentTraversal)
       || rendererCfgChanged(m_rendererConfig.useVulkanMemoryModel) || rendererCfgChanged(m_rendererConfig.useAtomicLoadStore)
       || rendererCfgChanged(m_rendererConfig.useDiscreteLod) || rendererCfgChanged(m_rendererConfig.usePartitions)
       || rendererCfgChanged(m_rendererConfig.usePtlas) || rendererCfgChanged(m_rendererConfig.usePtlasExplicitAabb)
       || rendererCfgChanged(m_rendererConfig.partitionLayout) || rendererCfgChanged(m_rendererConfig.partitionGridCellSize)
       || rendererCfgChanged(m_rendererConfig.partitionGridOrigin) || rendererCfgChanged(m_rendererConfig.partitionGridCellsPerAxis)
       || rendererCfgChanged(m_rendererConfig.partitionDerivedTarget) || rendererCfgChanged(m_rendererConfig.partitionDerivedMaxInstances)
       || rendererCfgChanged(m_rendererConfig.partitionDerivedMaxCount) || rendererCfgChanged(m_rendererConfig.usePartitionEarlyOut)
       || rendererCfgChanged(m_rendererConfig.textureLodMode) || rendererCfgChanged(m_rendererConfig.usePathtrace)
       || rendererCfgChanged(m_rendererConfig.useRayQuery) || rendererCfgChanged(m_rendererConfig.streamPropagationMode)
       || rendererCfgChanged(m_rendererConfig.usePersistentHitDemand) || rendererCfgChanged(m_rendererConfig.useHitStreaming)
       || rendererCfgChanged(m_rendererConfig.useHitStreamingLod) || rendererCfgChanged(m_rendererConfig.hitStreamingShadowDepth)
       || rendererCfgChanged(m_rendererConfig.cachingLevels) || rendererCfgChanged(m_rendererConfig.useHybridDiscreteLod)
       || rendererCfgChanged(m_rendererConfig.hybridCachingLevels) || rendererCfgChanged(m_rendererConfig.useHybridBlasDedup))
    {
      // groups loaded without hit tracking carry no hit frames, the first age filter would unload them
      bool hitStreamingOn = m_rendererConfig.useHitStreaming && m_tweak.renderer != RENDERER_RASTER_CLUSTERS_LOD
                            && !(m_rendererConfigLast.useHitStreaming && m_tweakLast.renderer != RENDERER_RASTER_CLUSTERS_LOD);

      if(m_renderScene
         && (rendererCfgChanged(m_rendererConfig.useBlasCaching) || rendererCfgChanged(m_rendererConfig.cachingLevels)
             || rendererCfgChanged(m_rendererConfig.useDiscreteLod) || rendererCfgChanged(m_rendererConfig.hybridCachingLevels)
             || hitStreamingOn
             // the hybrid renderer caches its own chain of levels
             || (m_tweak.renderer == RENDERER_HYBRID_CLUSTERS_LOD) != (m_tweakLast.renderer == RENDERER_HYBRID_CLUSTERS_LOD)))
      {
        m_renderScene->streamingReset();
      }

      // the renderer checks for these
      if(shaderChanged)
      {
        m_resources.reloadShadingShaders();
      }

      initRenderer(m_tweak.renderer);

      if(shaderChanged)
      {
        m_resources.m_hbaoPass.reloadShaders();
        m_clasAllocatorVis.reloadShaders(m_resources);
        //m_resources.m_hiz.initPipelines();
      }
    }
    else if(m_renderer && frameBufferChanged)
    {
      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }

    if(m_streamingResetRequest)
    {
      if(m_renderScene)
      {
        m_renderScene->streamingReset();
      }
      m_streamingResetRequest = false;
    }
  }


  bool hadChange = shaderChanged || memcmp(&m_tweakLast, &m_tweak, sizeof(m_tweak))
                   || memcmp(&m_rendererConfigLast, &m_rendererConfig, sizeof(m_rendererConfig))
                   || memcmp(&m_sceneConfigLast, &m_sceneConfig, sizeof(m_sceneConfig))
                   || memcmp(&m_streamingConfigLast, &m_streamingConfig, sizeof(m_streamingConfig))
                   || memcmp(&m_texturesConfigLast, &m_texturesConfig, sizeof(m_texturesConfig))
                   || memcmp(&m_sceneGridConfigLast, &m_sceneGridConfig, sizeof(m_sceneGridConfig));
  m_tweakLast           = m_tweak;
  m_rendererConfigLast  = m_rendererConfig;
  m_streamingConfigLast = m_streamingConfig;
  m_texturesConfigLast  = m_texturesConfig;
  m_sceneConfigLast     = m_sceneConfig;
  m_sceneGridConfigLast = m_sceneGridConfig;

  if(hadChange)
  {
    m_equalFrames = 0;
    if(m_tweak.autoResetTimers)
    {
      // runs after the sequencer's own resetFrameSections in the same frame, so this
      // sample-side delay overrides the sequencer's clamped `sequenceresetframes` (max 8)
      m_info.profilerManager->resetFrameSections(m_tweak.autoResetTimersDelay);
    }
  }
}

void LodClusters::applyCameraString()
{
  nvutils::CameraManipulator::Camera cam = m_info.cameraManipulator->getCamera();
  if(cam.setFromString(m_cameraString))
  {
    m_info.cameraManipulator->setCamera(cam);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }
  m_cameraStringLast = m_cameraString;
}

void LodClusters::registerCameraPathParameters()
{
  // `--addcamerapath "<string>"` appends a path definition; the index is the
  // order in which the paths were added. Repeatable.
  m_info.parameterRegistry->addCustom(
      {"addcamerapath", "append a camera path definition string (repeatable); the path index equals the order added"}, 1,
      [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path&) -> bool {
        CameraPath path;
        if(!path.setFromString(args[0]))
        {
          LOGE("addcamerapath: could not parse camera path string\n");
          return false;
        }
        m_cameraPaths.push_back(std::move(path));
        m_cameraPathsExternal = true;
        if(m_cameraPathActive < 0)
          m_cameraPathActive = 0;
        LOGI("addcamerapath: added path %d with %zu keyframes\n", int(m_cameraPaths.size()) - 1, m_cameraPaths.back().size());
        return true;
      });

  // `--loadcamerapaths "<filename>"` replaces all path definitions with the
  // paths from a simple text file (one path per line). Full overwrite.
  m_info.parameterRegistry->addCustom(
      {"loadcamerapaths", "load all camera path definitions from a text file (full overwrite)"}, 1,
      [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path& basePath) -> bool {
        std::filesystem::path file = std::filesystem::path(args[0]);
        if(file.is_relative() && !basePath.empty())
          file = basePath / file;
        if(!loadCameraPaths(file, m_cameraPaths))
        {
          LOGE("loadcamerapaths: could not open %s\n", nvutils::utf8FromPath(file).c_str());
          return false;
        }
        m_cameraPathActive    = m_cameraPaths.empty() ? -1 : 0;
        m_cameraPathPlayback  = CAMERA_PATH_STOP;
        m_cameraPathsExternal = true;
        LOGI("loadcamerapaths: loaded %zu path(s) from %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(file).c_str());
        return true;
      });

  // `--runcamerapath <index> <framecount>` plays back a path over a fixed number
  // of frames (deterministic), one keyframe-step per rendered frame. Meant to be
  // used within benchmark sequences.
  m_info.parameterRegistry->add({"camerapathloop",
                                 "make --runcamerapath wrap instead of holding at the end, so <framecount> sets one "
                                 "lap and the step length sets the duration. needs a closed path. default false"},
                                &m_cameraPathLoop);
  m_info.parameterRegistry->addCustom(
      {"runcamerapath", "deterministic fixed-step playback: <index> <framecount>, spreads the given path across framecount frames"},
      2, [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path&) -> bool {
        int index      = atoi(args[0]);
        int frameCount = atoi(args[1]);
        if(index < 0 || index >= int(m_cameraPaths.size()))
        {
          LOGE("runcamerapath: path index %d out of range (%d paths defined)\n", index, int(m_cameraPaths.size()));
          return false;
        }
        m_cameraPathActive   = index;
        m_cameraPathFrames   = std::max(frameCount, 1);
        m_cameraPathFrame    = 0;
        m_cameraPathPlayback = CAMERA_PATH_FIXED;
        return true;
      });
}

std::filesystem::path LodClusters::sceneCameraPathsFile(const std::filesystem::path& scenePath) const
{
  if(scenePath.empty())
    return {};

  // stored next to the model file, named after it, e.g. ".../bunny.camerapaths.txt"
  std::filesystem::path file = scenePath;
  file.replace_extension(".camerapaths.txt");
  return file;
}

bool LodClusters::applyCameraPathSample(double u)
{
  if(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()))
    return false;

  CameraPathKey key;
  if(!m_cameraPaths[m_cameraPathActive].sample(u, key))
    return false;

  // keep the manipulator's other camera state (clip planes, projection, ...)
  nvutils::CameraManipulator::Camera cam = m_info.cameraManipulator->getCamera();
  cam.eye                                = key.eye;
  cam.ctr                                = key.ctr;
  cam.up                                 = key.up;
  cam.fov                                = key.fov;
  m_info.cameraManipulator->setCamera(cam, true);
  return true;
}

void LodClusters::updateCameraPath(double time)
{
  if(m_cameraPathPlayback == CAMERA_PATH_STOP)
    return;

  if(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()) || m_cameraPaths[m_cameraPathActive].empty())
  {
    m_cameraPathPlayback = CAMERA_PATH_STOP;
    return;
  }

  const CameraPath& path = m_cameraPaths[m_cameraPathActive];

  if(m_cameraPathPlayback == CAMERA_PATH_FIXED)
  {
    double u;
    if(m_cameraPathLoop)
    {
      // one lap per `framecount` frames, so the speed a step runs at and how long it runs
      // are independent. u goes 0 .. (n-1)/n and wraps, keeping the seam the same step size
      // as every other. Assumes a closed path - a path whose last key is not its first will
      // jump there, which is why this is opt-in rather than driven off the file's loop flag.
      int n = std::max(m_cameraPathFrames, 1);
      u     = double(m_cameraPathFrame % n) / double(n);
      // a capture that waits for the viewport starts with this frame
      if(!m_frameCapture.isWaiting())
        m_cameraPathFrame++;
    }
    else
    {
      // deterministic: frame f maps to u = f / (frames - 1)
      int denom = std::max(m_cameraPathFrames - 1, 1);
      u         = double(std::min(m_cameraPathFrame, denom)) / double(denom);

      // advance for the next frame, hold at the end (the sequence/headless run
      // stops on its own frame count)
      if(m_cameraPathFrame < m_cameraPathFrames - 1 && !m_frameCapture.isWaiting())
        m_cameraPathFrame++;
    }
    applyCameraPathSample(u);
    m_cameraPathU = u;
  }
  else  // CAMERA_PATH_REALTIME
  {
    applyCameraPathSample(m_cameraPathU);

    // a capture plays at the path's speed however long its frames take
    double dt           = m_frameCapture.getTimeStep(std::max(time - m_lastTime, 0.0));
    dt                  = m_cameraPathStarted ? dt : 0.0;
    m_cameraPathStarted = true;
    if(path.duration > 0.0)
      m_cameraPathU += dt / path.duration;

    if(m_cameraPathU >= 1.0)
    {
      if(path.loop)
        m_cameraPathU = std::fmod(m_cameraPathU, 1.0);
      else
      {
        m_cameraPathU        = 1.0;
        m_cameraPathPlayback = CAMERA_PATH_STOP;
      }
    }
  }
}

void LodClusters::onRender(VkCommandBuffer cmd)
{
  double time = m_clock.getSeconds();

  {
    const Resources::FrameBuffer& fb = m_resources.m_frameBuffer;
    m_frameCapture.beginFrame(m_frameCapture.capturesWindow() ? m_app->getWindowSize() :
                              fb.useResolved                  ? fb.windowSize :
                                                                fb.targetSize);
  }

  // drive the camera along the active path (overrides the manipulator for the
  // rendered frame); runs after nvapp::ElementCamera's per-frame input update
  updateCameraPath(time);

  m_resources.beginFrame(m_app->getFrameCycleIndex());

  m_frameConfig.windowSize = m_windowSize;
  m_frameConfig.hbaoActive = false;

  if(m_renderer)
  {
    if(m_rendererFboChangeID != m_resources.m_fboChangeID)
    {
      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }

    bool rasterDlssSrActive = false;
#if USE_DLSS
    rasterDlssSrActive = m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD && m_rendererConfig.useDlss
                         && m_resources.m_frameBuffer.dlssUpscaler.isAvailable();
    // the hybrid renderer rasterizes primary visibility with the jitter of the path tracer rays,
    // which only use it while the denoiser is active
    rasterDlssSrActive = rasterDlssSrActive
                         || (m_tweak.renderer == RENDERER_HYBRID_CLUSTERS_LOD && m_renderer && m_renderer->getConfig().useDlss);
#endif
    m_frameConfig.hbaoActive = m_rendererConfig.useShading && m_tweak.hbaoActive && m_resources.m_frameBuffer.useRasterization;

    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;

    // for motion always use last
    frameConstants.viewProjMatrixPrev = frameConstants.viewProjMatrix;

    if(m_frames)
    {
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    int supersample = m_tweak.supersample;

    uint32_t renderWidth  = m_resources.m_frameBuffer.renderSize.width;
    uint32_t renderHeight = m_resources.m_frameBuffer.renderSize.height;

    uint32_t targetWidth  = m_resources.m_frameBuffer.targetSize.width;
    uint32_t targetHeight = m_resources.m_frameBuffer.targetSize.height;

    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;
    frameConstants.visualize    = m_frameConfig.visualize;
    frameConstants.frame        = m_frames;
    frameConstants.timeSec      = float(time);

    // Latch last frame's picking result into the UBO so debug passes
    // (e.g. instance / cluster bbox highlight) don't have to race the
    // current frame's readback SSBO.
    {
      shaderio::Readback lastReadback;
      m_resources.getReadbackData(lastReadback);
      bool valid                      = isPickingValid(lastReadback);
      frameConstants.pickedInstanceID = valid ? lastReadback.instanceId : ~0u;
      // clusterTriangleId low 32 bits = (clusterID << 8) | triangleID,
      // matching packedClusterTriangleId in the picking shaders.
      frameConstants.pickedClusterID = valid ? (lastReadback.clusterTriangleId >> 8) : ~0u;

      updateExposure(lastReadback);
    }
    if(isRawVisualize())
      frameConstants.tonemapMode = TONEMAP_MODE_BYPASS;
    else if(m_frameConfig.visualize != VISUALIZE_SHADED && m_frameConfig.visualize != VISUALIZE_GREY)
      frameConstants.tonemapMode = TONEMAP_MODE_CLIP;
    else
      frameConstants.tonemapMode = TONEMAP_MODE_FULL;

    frameConstants.visFilterInstanceID = m_tweak.filterInstanceID < 0 ? ~0u : uint32_t(m_tweak.filterInstanceID);
    frameConstants.visFilterClusterID  = m_tweak.filterClusterID < 0 ? ~0u : uint32_t(m_tweak.filterClusterID);

    frameConstants.bgColor   = m_resources.m_bgColor;
    frameConstants.viewport  = glm::ivec2(renderWidth, renderHeight);
    frameConstants.viewportf = glm::vec2(renderWidth, renderHeight);
    frameConstants.nearPlane = float(m_info.cameraManipulator->getClipPlanes().x);
    frameConstants.farPlane  = float(m_info.cameraManipulator->getClipPlanes().y);
    frameConstants.wUpDir    = m_info.cameraManipulator->getUp();
#if USE_DLSS
    frameConstants.jitter = shaderio::dlssJitter(m_frames);
#endif
    frameConstants.fov = float(glm::radians(m_info.cameraManipulator->getFov()));

    glm::mat4 projection = glm::perspectiveRH_ZO(frameConstants.fov, float(targetWidth) / float(targetHeight),
                                                 frameConstants.farPlane, frameConstants.nearPlane);
    projection[1][1] *= -1;
    glm::mat4 projectionRender = projection;
    if(rasterDlssSrActive)
    {
      glm::mat4 jitterMatrix(1.0f);
      jitterMatrix[3][0] = -frameConstants.jitter.x * 2.0f / std::max(frameConstants.viewportf.x, 1.0f);
      jitterMatrix[3][1] = -frameConstants.jitter.y * 2.0f / std::max(frameConstants.viewportf.y, 1.0f);
      projectionRender   = jitterMatrix * projection;
    }

    glm::mat4 view  = m_info.cameraManipulator->getViewMatrix();
    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrix       = projection * view;
    frameConstants.viewProjMatrixI      = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewProjMatrixRender = projectionRender * view;
    frameConstants.viewMatrix           = view;
    frameConstants.viewMatrixI          = viewI;
    frameConstants.projMatrix           = projection;
    frameConstants.projMatrixI          = glm::inverse(projection);
    frameConstants.pixelAngle = 2.0f * glm::abs(frameConstants.projMatrixI[1][1]) / glm::max(frameConstants.viewportf.y, 1.0f);

    // ray cones and hardware derivatives are taken at render resolution, so when DLSS upscales,
    // cancel the extra blur with the resolution ratio
    frameConstants.texLodBias =
        m_tweak.texLodBias + 0.5f * std::log2(float(renderWidth * renderHeight) / float(targetWidth * targetHeight));
    frameConstants.texLodBiasScale = std::exp2(frameConstants.texLodBias);

    frameConstants.viewPos = frameConstants.viewMatrixI[3];  // position of eye in the world
    frameConstants.viewDir = -viewI[2];

    frameConstants.viewPlane   = frameConstants.viewDir;
    frameConstants.viewPlane.w = -glm::dot(glm::vec3(frameConstants.viewPos), glm::vec3(frameConstants.viewDir));

    frameConstants.wLightPos = frameConstants.viewMatrixI[3];  // place light at position of eye in the world
    frameConstants.skyPhysical.sunDirection = glm::normalize(frameConstants.skyPhysical.sunDirection);

    // the lens is camera-anchored; the shader derives its center from viewPos +
    // viewDir * lensDist, so it no longer needs a world position in wMirrorBox
    // the lens divides by zoom and size, 0 gives NaN directions: the UI's minimums
    frameConstants.lensZoom = std::max(frameConstants.lensZoom, 1.0f);
    frameConstants.lensSize = std::max(frameConstants.lensSize, std::max(frameConstants.sceneSize, 1.0f) * 0.00001f);

    {
      // hiz
      m_resources.m_hizUpdate[0].farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);
      m_resources.m_hizUpdate[0].farInfo.getSize((float*)&frameConstants.hizSize);
    }

    {
      // hbao setup
      auto& hbaoView            = m_frameConfig.hbaoSettings.view;
      hbaoView.farPlane         = frameConstants.farPlane;
      hbaoView.nearPlane        = frameConstants.nearPlane;
      hbaoView.projectionMatrix = projection;
      hbaoView.viewMatrix       = view;
      hbaoView.tanFovy          = tanf(frameConstants.fov * 0.5f);

      // negative radius is absolute world-space
      float sceneSize                   = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);
      m_frameConfig.hbaoSettings.radius = m_tweak.hbaoRadius < 0 ? -m_tweak.hbaoRadius : sceneSize * m_tweak.hbaoRadius;
      frameConstants.ambientOcclusionRadius = m_tweak.aoRadius < 0 ? -m_tweak.aoRadius : sceneSize * m_tweak.aoRadius;
    }

    VkExtent2D hdriSize              = m_resources.getEnvHdriSize();
    frameConstants.envHdri.enabled   = m_tweak.useEnvHdri && m_resources.hasEnvHdri() ? 1 : 0;
    frameConstants.envHdri.intensity = m_tweak.envHdriIntensity;
    frameConstants.envHdri.rotation  = glm::radians(m_tweak.envHdriRotationDeg);
    frameConstants.envHdri.elevation = glm::radians(m_tweak.envHdriElevationDeg);
    frameConstants.envHdri.width     = hdriSize.width;
    frameConstants.envHdri.height    = hdriSize.height;

    frameConstants.fogDensity = 0;
    if(m_tweak.fogDistance != 0 && m_frameConfig.visualize == VISUALIZE_SHADED)
    {
      // relative to the default haze of the physical sky, 2 + haze is its turbidity
      float gridSize            = glm::length(m_scene->m_gridBbox.hi - m_scene->m_gridBbox.lo);
      float distance            = m_tweak.fogDistance < 0 ? -m_tweak.fogDistance : gridSize * m_tweak.fogDistance;
      float hazeScale           = std::max(2.0f, 2.0f + frameConstants.skyPhysical.haze) / 2.1f;
      frameConstants.fogDensity = hazeScale * glm::ln_two<float>() / distance;
    }

    if(!m_frames)
    {
      // on first frame replicate last
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    if(!m_frameConfig.freezeLoD)
    {
      m_frameConfig.traversalViewMatrix = m_frameConfig.frameConstants.viewMatrix;
      m_frameConfig.traversalFov        = m_frameConfig.frameConstants.fov;
      m_frameConfig.traversalViewHeight = m_frameConfig.frameConstants.viewportf.y;
    }
    if(!m_frameConfig.freezeCulling)
    {
      m_frameConfig.cullViewProjMatrix     = m_frameConfig.frameConstants.viewProjMatrix;
      m_frameConfig.cullViewProjMatrixLast = m_frameConfig.frameConstantsLast.viewProjMatrix;
    }

    if(m_frames)
    {
      shaderio::FrameConstants frameCurrent = m_frameConfig.frameConstants;
      frameCurrent.frame                    = m_frameConfig.frameConstantsLast.frame;
      frameCurrent.jitter                   = m_frameConfig.frameConstantsLast.jitter;
      frameCurrent.viewProjMatrixRender     = m_frameConfig.frameConstantsLast.viewProjMatrixRender;
      if(memcmp(&frameCurrent, &m_frameConfig.frameConstantsLast, sizeof(shaderio::FrameConstants)))
        m_equalFrames = 0;
      else
        m_equalFrames++;
    }

    m_resources.cmdUpdateSkyLighting(cmd, m_frameConfig, m_profilerGpuTimer);
    m_renderer->render(cmd, m_resources, *m_renderScene, m_frameConfig, m_profilerGpuTimer);
    accumulateFrameStats();

    // after all streaming operations of this frame, so it reflects the state the
    // next frame allocates against
    if(m_clasAllocatorVisVisible && m_renderScene->useStreaming)
    {
      auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Clas Allocator Vis");

      const shaderio::SceneStreaming& streamingData = m_renderScene->sceneStreaming.getShaderStreamingData();

      StreamingAllocatorVis::FrameInfo visInfo;
      visInfo.streamingAddress  = m_renderScene->sceneStreaming.getShaderStreamingBuffer().address;
      visInfo.activeGroupsCount = streamingData.resident.activeGroupsCount;
      visInfo.colorXor          = m_frameConfig.frameConstants.colorXor;
      visInfo.showGroups        = m_clasAllocatorVisShowGroups;
      visInfo.displayWidth      = uint32_t(m_clasAllocatorVisDisplaySize.x);
      visInfo.displayHeight     = uint32_t(m_clasAllocatorVisDisplaySize.y);
      visInfo.wantHistogram     = m_clasAllocatorVisHistogram;
      visInfo.cycleIndex        = m_resources.m_cycleIndex;

      m_clasAllocatorVis.cmdUpdate(cmd, streamingData.clasAllocator, visInfo);
    }
  }
  else
  {
    m_resources.emptyFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  {
    m_resources.postProcessFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  // the whole window is captured once the UI is drawn
  if((m_frameCapture.config.enabled || m_frameCapture.isActive()) && !m_frameCapture.capturesWindow())
  {
    nvvk::ProfilerGpuTimer::FrameSection sec = m_profilerGpuTimer.cmdFrameSection(cmd, "Frame Capture");
    const Resources::FrameBuffer&        fb  = m_resources.m_frameBuffer;
    const nvvk::Image&                   img = fb.useResolved ? fb.imgColorResolved : fb.imgColorLdr;
    m_frameCapture.cmdFrame(cmd, m_app->getFrameCycleIndex(), img.image, img.descriptor.imageView,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, img.format,
                            fb.useResolved ? fb.windowSize : fb.targetSize, m_app->getFrameSignalSemaphore());
  }

  m_resources.endFrame();

  // signal new semaphore state with this command buffer's submit
  VkSemaphoreSubmitInfo semSubmit = m_resources.m_queueStates.primary.advanceSignalSubmit(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
  m_app->addSignalSemaphore(semSubmit);
  // but also enqueue waits if there are any
  while(!m_resources.m_queueStates.primary.m_pendingWaits.empty())
  {
    m_app->addWaitSemaphore(m_resources.m_queueStates.primary.m_pendingWaits.back());
    m_resources.m_queueStates.primary.m_pendingWaits.pop_back();
  }

  if(m_renderer && m_headlessPickingWait > 0)
  {
    m_headlessPickingWait--;
  }

  m_lastTime = time;
  m_frames++;

  // sample-side periodic memory report (independent of the sequencer's per-sequence report)
  if(m_memoryReportInterval && m_renderer && (uint32_t(m_frames) % m_memoryReportInterval) == 0)
  {
    logMemoryReport(fmt::format("Frame_{}", m_frames));
  }
}

// `scene` rather than m_scene: this runs on the loader thread, where the scene is not installed
// yet (and its cameras are fixed up here)
void LodClusters::setSceneCamera(const std::filesystem::path& filePath, Scene& scene)
{
  nvgui::SetCameraJsonFile(filePath);

  glm::vec3 modelExtent = scene.m_bbox.hi - scene.m_bbox.lo;
  float     modelRadius = glm::length(modelExtent) * 0.5f;
  glm::vec3 modelCenter = (scene.m_bbox.hi + scene.m_bbox.lo) * 0.5f;

  bool bigScene = scene.m_isBig;

  if(!scene.m_cameras.empty())
  {
    // keep the clip planes updatedSceneGrid derived, the camera's defaults would replace them
    glm::dvec2 clipPlanes = m_info.cameraManipulator->getClipPlanes();

    auto& c = scene.m_cameras[std::clamp(m_cameraGltf, 0, int(scene.m_cameras.size()) - 1)];
    m_info.cameraManipulator->setFov(c.fovy);


    c.eye              = glm::vec3(c.worldMatrix[3]);
    float     distance = glm::length(modelCenter - c.eye);
    glm::mat3 rotMat   = glm::mat3(c.worldMatrix);
    c.center           = {0, 0, -distance};
    c.center           = c.eye + (rotMat * c.center);
    c.up               = {0, 1, 0};

    m_info.cameraManipulator->setCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy)), clipPlanes});

    nvgui::SetHomeCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy)), clipPlanes});
    for(auto& cam : scene.m_cameras)
    {
      cam.eye            = glm::vec3(cam.worldMatrix[3]);
      float     distance = glm::length(modelCenter - cam.eye);
      glm::mat3 rotMat   = glm::mat3(cam.worldMatrix);
      cam.center         = {0, 0, -distance};
      cam.center         = cam.eye + (rotMat * cam.center);
      cam.up             = {0, 1, 0};


      nvgui::AddCamera({cam.eye, cam.center, cam.up, static_cast<float>(glm::degrees(cam.fovy)), clipPlanes});
    }
  }
  else
  {
    glm::vec3 up  = {0, 1, 0};
    glm::vec3 dir = {1.0f, bigScene ? 0.33f : 0.75f, 1.0f};

    m_info.cameraManipulator->setLookat(modelCenter + dir * (modelRadius * (bigScene ? 0.5f : 1.f)), modelCenter, up);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }

  if(m_cameraSpeed)
  {
    m_info.cameraManipulator->setSpeed(m_cameraSpeed);
  }

  if(!m_cameraString.empty())
  {
    applyCameraString();
  }
}

float LodClusters::decodePickingDepth(const shaderio::Readback& readback)
{
  if(!isPickingValid(readback))
  {
    return 0.f;
  }
  uint32_t bits = readback._packedDepth0;
  bits ^= ~(int(bits) >> 31) | 0x80000000u;
  float res = *(float*)&bits;
  // reversed-Z depth as written by the shaders (1 = near plane, 0 = far plane),
  // matches frameConstants.projMatrix
  return res;
}

bool LodClusters::isPickingValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}

void LodClusters::updateExposure(const shaderio::Readback& readback)
{
  shaderio::TonemapperData& tm = m_frameConfig.frameConstants.tonemapper;

  if(readback.autoExposureSampleCount > 0)
  {
    // geometric-mean (log-average) luminance -> exposure that maps it to the middle-grey key,
    // with the measured luminance restricted to the tonemapper's EV100 window
    float avgLogLuma = readback.autoExposureLumaSum / float(readback.autoExposureSampleCount);
    float ev100      = std::clamp(shaderio::luminanceEv100(std::exp2(avgLogLuma)), tm.evMinValue, tm.evMaxValue);
    float avgLuma    = shaderio::ev100Luminance(ev100);

    const float key    = 0.18f;
    float       target = std::clamp(key / std::max(avgLuma, 1e-4f), 0.01f, 100.0f);
    // a capture adapts at its frame rate, so it looks the same however long its frames take
    double dt        = m_frameCapture.getTimeStep(m_exposureTimer.getSeconds());
    float  adaptRate = std::clamp(tm.autoExposureSpeed * float(dt), 0.0f, 1.0f);
    m_exposure += (target - m_exposure) * adaptRate;
  }
  m_exposureTimer.reset();

  bool  debugLighting = m_frameConfig.visualize != VISUALIZE_SHADED;
  float autoScale     = tm.autoExposure != 0 ? m_exposure : 1.0f;
  float debugScale    = debugLighting ? std::exp2(m_tweak.debugExposure) : 1.0f;
  tm.inputMatrix = shaderio::getColorCorrectionMatrix(tm.exposure * autoScale * debugScale, tm.temperature, tm.tint);

  m_frameConfig.frameConstants.skyAmbientScale  = debugLighting ? m_tweak.debugSkyScale : 1.0f;
  m_frameConfig.frameConstants.rasterSkyScale   = m_tweak.rasterSkyScale;
  m_frameConfig.frameConstants.debugRoughness   = m_tweak.debugRoughness;
  m_frameConfig.frameConstants.debugSpecular    = m_tweak.debugSpecular;
  m_frameConfig.frameConstants.debugPaletteGain = m_tweak.debugPaletteGain;
  m_frameConfig.frameConstants.debugContrast    = m_tweak.debugContrast;
  m_frameConfig.frameConstants.debugSaturation  = m_tweak.debugSaturation;
}

}  // namespace lodclusters
