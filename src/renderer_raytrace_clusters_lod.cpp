/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <volk.h>
#include <nvvk/sbt_generator.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/alignment.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>

#include "renderer.hpp"
#include "partitioned_acceleration_structures.hpp"

#define USE_LARGE_BUFFER_BLAS 1

//////////////////////////////////////////////////////////////////////////

namespace lodclusters {

class RendererRayTraceClustersLod : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene);

private:
  // sharing and caching are independent, but both need the instance lod classification
  bool useBlasReuse() const { return m_config.useBlasSharing || m_config.useBlasCaching; }

  bool initShaders(Resources& res, RenderScene& scene);

  void initRayTracingPipeline(Resources& res);
  bool initRayTracingBlas(Resources& res, RenderScene& scene, VkDeviceSize& scratchSize);
  void initRayTracingTlas(Resources& res, RenderScene& rscene, VkDeviceSize& scratchSize);
  bool initRayTracingPtlas(Resources& res, RenderScene& rscene, VkDeviceSize& scratchSize);
  // the initial build writing all instances, needs the scratch and the build data
  void initRayTracingPtlasInstances(Resources& res, RenderScene& rscene);
  // the partition arrays the early-out reads, independent of the ptlas
  void initInstancePartitions(Resources& res, RenderScene& rscene);

  void updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, bool update = false);
  void updateRayTracingPtlas(VkCommandBuffer cmd);


  struct Shaders
  {
    Shader rayGen;
    Shader rayClosestHit;
    Shader rayAnyHit;
    Shader rayClosestHitAO;
    Shader rayMiss;
    Shader rayMissAO;

    Shader computeTraversalPresort;
    Shader computeTraversalInit;
    Shader computeTraversalRun;
    Shader computeTraversalGroups;
    Shader computeTraversalMerge;
    Shader computeBuildSetup;

    Shader computeBlasInsertClusters;
    Shader computeBlasSetupInsertion;
    Shader computeBlasCachingSetupCopy;
    Shader computeBlasCachingSetupBuild;

    Shader computeInstanceAssignBlas;
    Shader computeInstanceClassifyLod;
    Shader computeGeometryBlasSharing;
    Shader computeGeometryCachedRetain;
    Shader computePartitionSetup;
    Shader computePartitionBlockCompact;
    Shader computeInstanceHitScale;

    Shader computeHitPropagation;
    Shader computeHitDemand;
  };

  struct Pipelines
  {
    VkPipeline rayTracing = nullptr;

    VkPipeline computeTraversalPresort = nullptr;
    VkPipeline computeTraversalInit    = nullptr;
    VkPipeline computeTraversalRun     = nullptr;
    VkPipeline computeTraversalGroups  = nullptr;
    VkPipeline computeTraversalMerge   = nullptr;
    VkPipeline computeBuildSetup       = nullptr;

    VkPipeline computeBlasInsertClusters    = nullptr;
    VkPipeline computeBlasSetupInsertion    = nullptr;
    VkPipeline computeBlasCachingSetupCopy  = nullptr;
    VkPipeline computeBlasCachingSetupBuild = nullptr;
    VkPipeline computeInstanceAssignBlas    = nullptr;
    VkPipeline computeInstanceClassifyLod   = nullptr;
    VkPipeline computeGeometryBlasSharing   = nullptr;
    VkPipeline computeGeometryCachedRetain  = nullptr;
    VkPipeline computePartitionSetup        = nullptr;
    VkPipeline computePartitionBlockCompact = nullptr;
    VkPipeline computeInstanceHitScale      = nullptr;

    VkPipeline computeHitPropagation = nullptr;
    VkPipeline computeHitDemand      = nullptr;
  };

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV m_rtClasProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};

  nvvk::SBTGenerator::Regions m_sbtRegions;  // Shading binding table wrapper
  nvvk::Buffer                m_sbtBuffer;

  Shaders            m_shaders;
  Pipelines          m_pipelines;
  VkShaderStageFlags m_stageFlags{};
  bool               m_hasAlphaMask = false;  // scene has alpha-masked geometry -> pipeline needs the any-hit group
  VkPipelineLayout   m_pipelineLayout{};

  nvvk::DescriptorPack m_dsetPack;

  nvvk::Buffer m_sceneBuildBuffer;
  nvvk::Buffer m_sceneDataBuffer;

#if USE_LARGE_BUFFER_BLAS
  nvvk::LargeBuffer m_sceneBlasDataBuffer;
#else
  nvvk::Buffer m_sceneBlasDataBuffer;
#endif
  nvvk::Buffer            m_sceneTraversalBuffer;
  nvvk::Buffer            m_sceneGeometryHistogramBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio;

  VkClusterAccelerationStructureClustersBottomLevelInputNV m_blasInput{};
  VkDeviceSize                                             m_blasDataSize = 0;

  VkClusterAccelerationStructureMoveObjectsInputNV m_blasMoveInput{};

  nvvk::Buffer                                m_tlasInstancesBuffer;
  VkAccelerationStructureGeometryKHR          m_tlasGeometry{};
  VkAccelerationStructureBuildGeometryInfoKHR m_tlasBuildInfo{};
  nvvk::AccelerationStructure                 m_tlas;

  // for m_config.usePtlas
  PartitionedTlas m_ptlas;
  // holds the operation count, followed by the operation of the frame
  nvvk::Buffer m_ptlasOperationsBuffer;
  uint32_t     m_partitionCount = 0;

  nvvk::Buffer m_scratchBuffer;
};

bool RendererRayTraceClustersLod::initShaders(Resources& res, RenderScene& rscene)
{
  if(!initBasicShaders(res, rscene, false))
  {
    return false;
  }

  shaderc::CompileOptions options = res.makeCompilerOptions();

  // persistent producer/consumer demand descent only makes sense for the multi-level BFS
  // (resident subtree); resident-levels is a deliberate single hop per frame, and the sweep
  // (all levels) uses a different shader. Gate the macro so build_setup and the demand
  // shader agree, and so the host dispatch below matches.
  const bool usePersistentDemandBFS =
      m_config.usePersistentHitDemand && m_config.streamPropagationMode == STREAM_PROPAGATION_RESIDENT_SUBTREE;
#if USE_DLSS
  bool supportsDLSS = true;
#else
  bool supportsDLSS = false;
#endif

  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));
  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", rscene.scene->m_maxClusterVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", rscene.scene->m_maxClusterTriangles));
  options.AddMacroDefinition("GROUP_CLUSTER_COUNT", fmt::format("{}", rscene.scene->m_config.clusterGroupSize));
  options.AddMacroDefinition("TARGETS_RASTERIZATION", "0");
  options.AddMacroDefinition("USE_STREAMING", rscene.useStreaming ? "1" : "0");
  options.AddMacroDefinition("USE_SORTING", m_config.useSorting ? "1" : "0");
  options.AddMacroDefinition("USE_CULLING", m_config.useCulling ? "1" : "0");
  options.AddMacroDefinition("USE_TWO_PASS_CULLING", "0");
  options.AddMacroDefinition("USE_TWO_PASS_REJECT_LISTS", "0");
  options.AddMacroDefinition("USE_BLAS_REUSE", useBlasReuse() ? "1" : "0");
  options.AddMacroDefinition("USE_BLAS_SHARING", m_config.useBlasSharing ? "1" : "0");
  options.AddMacroDefinition("USE_BLAS_MERGING", m_config.useBlasSharing && m_config.useBlasMerging ? "1" : "0");
  options.AddMacroDefinition("USE_BLAS_CACHING", m_config.useBlasCaching ? "1" : "0");
  options.AddMacroDefinition("USE_PTLAS", m_config.usePtlas ? "1" : "0");
  options.AddMacroDefinition("USE_PARTITIONS", (m_config.usePtlas || m_config.usePartitionEarlyOut) ? "1" : "0");
  options.AddMacroDefinition("USE_PARTITION_EARLY_OUT", m_config.usePartitionEarlyOut ? "1" : "0");
  options.AddMacroDefinition("USE_RENDER_STATS", m_config.useRenderStats ? "1" : "0");
  options.AddMacroDefinition("USE_DLSS", supportsDLSS && m_config.useDlss ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_NORMALS", rscene.scene->m_hasVertexNormals ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TANGENTS", rscene.scene->m_hasVertexTangents ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TEXCOORDS",
                             rscene.scene->m_hasVertexTexCoord0 || rscene.scene->m_hasVertexTexCoord1 ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TEXCOORD_0", rscene.scene->m_hasVertexTexCoord0 ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TEXCOORD_1", rscene.scene->m_hasVertexTexCoord1 ? "1" : "0");
  options.AddMacroDefinition("ALLOW_SHADING", m_config.useShading ? "1" : "0");
  options.AddMacroDefinition("USE_DEPTH_ONLY", !m_config.useShading && m_config.useDepthOnly ? "1" : "0");
  options.AddMacroDefinition("DEBUG_VISUALIZATION", m_config.useDebugVisualization ? "1" : "0");
  options.AddMacroDefinition("USE_EXT_MESH_SHADER", fmt::format("{}", m_config.useEXTmeshShader ? 1 : 0));
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", fmt::format("{}", m_meshShaderWorkgroupSize));
  options.AddMacroDefinition("MESHSHADER_BBOX_COUNT", fmt::format("{}", m_meshShaderBoxes));
  options.AddMacroDefinition("USE_SW_RASTER", "0");
  options.AddMacroDefinition("USE_TWO_SIDED", rscene.scene->m_hasTwoSided && !m_config.forceTwoSided ? "1" : "0");
  options.AddMacroDefinition("USE_FORCED_TWO_SIDED", m_config.forceTwoSided ? "1" : "0");
  options.AddMacroDefinition("USE_FORCED_INVISIBLE_CULLING", m_config.useForcedInvisibleCulling ? "1" : "0");
  options.AddMacroDefinition("USE_PERSISTENT_TRAVERSAL_KERNEL", m_config.usePersistentTraversal ? "1" : "0");
  options.AddMacroDefinition("TRAVERSAL_VULKAN_MEMORY_MODEL",
                             m_config.usePersistentTraversal && m_config.useVulkanMemoryModel ? "1" : "0");
  options.AddMacroDefinition("TRAVERSAL_ATOMIC_LOAD_STORE", m_config.useAtomicLoadStore ? "1" : "0");
  options.AddMacroDefinition("USE_PERSISTENT_HIT_DEMAND_KERNEL", usePersistentDemandBFS ? "1" : "0");
  // Ray/path tracer texture LOD mode (overrides the TARGETS_RASTERIZATION-derived default in shaderio.h).
  options.AddMacroDefinition("TEXTURE_LOD_MODE", m_config.textureLodMode == 1 ? "TEXLODMODE_LOD" :
                                                 m_config.textureLodMode == 2 ? "TEXLODMODE_IMPLICIT" :
                                                                                "TEXLODMODE_GRAD");
  options.AddMacroDefinition("HAS_ALPHA_TEST", rscene.scene->m_hasAlphaMask ? "1" : "0");
  options.AddMacroDefinition("HAS_TEXTURED_MATERIALS", rscene.scene->m_hasTexturedMaterials ? "1" : "0");
  options.AddMacroDefinition("USE_PATHTRACING", m_config.usePathtrace ? "1" : "0");
  // path depth up to which sun shadow rays register hits, the ray tracer only has primary hits
  uint32_t shadowHitDepth = m_config.useHitStreaming ? uint32_t(std::max(m_config.hitStreamingShadowDepth, 1)) : 0;
  options.AddMacroDefinition("HIT_STREAMING_SHADOW_DEPTH", fmt::format("{}", shadowHitDepth));
  options.AddMacroDefinition("USE_AO_HITGROUP", shadowHitDepth > 0 ? "1" : "0");
  options.AddMacroDefinition("USE_HIT_STREAMING", m_config.useHitStreaming ? "1" : "0");
  // implies USE_STREAMING and USE_HIT_STREAMING
  options.AddMacroDefinition("USE_HIT_STREAMING_LOD", m_config.useHitStreamingLod ? "1" : "0");

  shaderc::CompileOptions optionsAO = options;
  options.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "0");
  optionsAO.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "1");

  // The basic path tracer uses a separate shader set: the closest-hit only reports the hit
  // and all shading happens in the ray-generation shader.
  const bool  pt        = m_config.usePathtrace;
  const char* rgenFile  = pt ? "render_pathtrace.rgen.glsl" : "render_raytrace.rgen.glsl";
  const char* rchitFile = pt ? "render_pathtrace_clusters.rchit.glsl" : "render_raytrace_clusters.rchit.glsl";
  const char* rahitFile = pt ? "render_pathtrace_clusters.rahit.glsl" : "render_raytrace_clusters.rahit.glsl";
  const char* rmissFile = pt ? "render_pathtrace.rmiss.glsl" : "render_raytrace.rmiss.glsl";

  m_hasAlphaMask = rscene.scene->m_hasAlphaMask;
  res.compileShader(m_shaders.rayGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgenFile, &options);
  res.compileShader(m_shaders.rayClosestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rchitFile, &options);
  // the any-hit only alpha-tests cutout geometry; compile it (and its hit groups) only when the scene has any
  if(m_hasAlphaMask)
    res.compileShader(m_shaders.rayAnyHit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, rahitFile, &options);
  // shadow ray closest-hit that records hits for hit-based streaming, traced via the AO hit
  // group (AO rays skip it). The path tracer's shadow rays run its own closest-hit, so this
  // stage is unused when path tracing.
  res.compileShader(m_shaders.rayClosestHitAO, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                    "render_raytrace_clusters_ao.rchit.glsl", &optionsAO);
  res.compileShader(m_shaders.rayMiss, VK_SHADER_STAGE_MISS_BIT_KHR, rmissFile, &options, "primary");
  res.compileShader(m_shaders.rayMissAO, VK_SHADER_STAGE_MISS_BIT_KHR, rmissFile, &optionsAO, "ao");

  if(m_config.useSorting)
  {
    res.compileShader(m_shaders.computeTraversalPresort, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_presort.comp.glsl", &options);
  }

  if(useBlasReuse())
  {
    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init_blas_reuse.comp.glsl", &options);
    if(m_config.useBlasSharing && m_config.useBlasMerging)
    {
      res.compileShader(m_shaders.computeTraversalMerge, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_blas_merging.comp.glsl", &options);
    }
  }
  else
  {
    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", &options);
  }

  res.compileShader(m_shaders.computeTraversalGroups, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run_groups.comp.glsl", &options);
  res.compileShader(m_shaders.computeTraversalRun, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", &options);
  res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasInsertClusters, VK_SHADER_STAGE_COMPUTE_BIT, "blas_clusters_insert.comp.glsl", &options);
  res.compileShader(m_shaders.computeBlasSetupInsertion, VK_SHADER_STAGE_COMPUTE_BIT, "blas_setup_insertion.comp.glsl", &options);
  res.compileShader(m_shaders.computeInstanceAssignBlas, VK_SHADER_STAGE_COMPUTE_BIT, "instance_assign_blas.comp.glsl", &options);

  if(useBlasReuse())
  {
    res.compileShader(m_shaders.computeInstanceClassifyLod, VK_SHADER_STAGE_COMPUTE_BIT, "instance_classify_lod.comp.glsl", &options);
  }
  if(m_config.useBlasSharing)
  {
    res.compileShader(m_shaders.computeGeometryBlasSharing, VK_SHADER_STAGE_COMPUTE_BIT, "geometry_blas_sharing.comp.glsl", &options);
  }
  if(m_config.useBlasCaching)
  {
    res.compileShader(m_shaders.computeGeometryCachedRetain, VK_SHADER_STAGE_COMPUTE_BIT,
                      "geometry_cached_retain.comp.glsl", &options);
  }
  if(m_config.usePartitionEarlyOut)
  {
    res.compileShader(m_shaders.computePartitionSetup, VK_SHADER_STAGE_COMPUTE_BIT, "partition_setup.comp.glsl", &options);
    res.compileShader(m_shaders.computePartitionBlockCompact, VK_SHADER_STAGE_COMPUTE_BIT,
                      "partition_block_compact.comp.glsl", &options);
  }
  if(m_config.useHitStreamingLod)
  {
    res.compileShader(m_shaders.computeInstanceHitScale, VK_SHADER_STAGE_COMPUTE_BIT, "instance_hit_scale.comp.glsl", &options);
  }
  if(m_config.useBlasCaching)
  {
    res.compileShader(m_shaders.computeBlasCachingSetupCopy, VK_SHADER_STAGE_COMPUTE_BIT,
                      "blas_caching_setup_copy.comp.glsl", &options);
    res.compileShader(m_shaders.computeBlasCachingSetupBuild, VK_SHADER_STAGE_COMPUTE_BIT,
                      "blas_caching_setup_build.comp.glsl", &options);
  }

  if(rscene.useStreaming && m_config.useHitStreaming)
  {
    // hit-streaming demand propagation strategy (see shaderio.h STREAM_PROPAGATION_*):
    // resident levels / subtree (0/1) share the BFS shader (differing only in host pass
    // count); all levels (2) and its rolling variant (3) use the level-wise sweep shader.
    // Rolling is a host-only distinction (one reverse level per frame), so it compiles
    // identically to ALL_LEVELS.
    const bool useLevelsSweep = m_config.streamPropagationMode == STREAM_PROPAGATION_ALL_LEVELS
                                || m_config.streamPropagationMode == STREAM_PROPAGATION_ALL_LEVELS_ROLLING;
    shaderc::CompileOptions streamOptions = res.makeCompilerOptions();
    // build_setup writes their indirect grids
    streamOptions.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
    streamOptions.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));
    streamOptions.AddMacroDefinition("STREAM_PROPAGATION_MODE", m_config.streamPropagationMode == STREAM_PROPAGATION_RESIDENT_LEVELS ?
                                                                    "STREAM_PROPAGATION_RESIDENT_LEVELS" :
                                                                useLevelsSweep ? "STREAM_PROPAGATION_ALL_LEVELS" :
                                                                                 "STREAM_PROPAGATION_RESIDENT_SUBTREE");
    // must match the value given to build_setup via `options` above (the two cooperate)
    streamOptions.AddMacroDefinition("USE_PERSISTENT_HIT_DEMAND_KERNEL", usePersistentDemandBFS ? "1" : "0");
    res.compileShader(m_shaders.computeHitPropagation, VK_SHADER_STAGE_COMPUTE_BIT, "stream_hit_propagation.comp.glsl", &streamOptions);
    res.compileShader(m_shaders.computeHitDemand, VK_SHADER_STAGE_COMPUTE_BIT,
                      useLevelsSweep ? "stream_hit_demand_levels.comp.glsl" : "stream_hit_demand_tree.comp.glsl", &streamOptions);
  }

  return res.verifyShaders(m_shaders);
}

bool RendererRayTraceClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_resourceReservedUsage = {};
  m_config                = config;
  m_maxRenderClusters     = 1u << m_config.numRenderClusterBits;
  m_maxTraversalTasks     = 1u << m_config.numTraversalTaskBits;
  res.setFramebufferUseRasterization(false);

  if(!rscene.useStreaming)
  {
    m_config.useBlasMerging = false;
    m_config.useBlasCaching = false;
  }
  else if(!rscene.sceneStreaming.getStreamingConfig().usePersistentClasAllocator)
  {
    // A cached blas stores the addresses of the CLAS it was built from and is kept across
    // frames, so those addresses have to stay valid for as long as the cache lives. Only the
    // persistent CLAS allocator guarantees that: the move based compaction relocates every
    // resident CLAS whenever any group is unloaded, which would leave cached blas of untouched
    // geometries pointing at other geometries' CLAS memory (device loss).
    // Only this local copy is cleared, so the requested setting comes back with the allocator.
    m_config.useBlasCaching = false;
  }

  // the persistent kernel's Vulkan memory model variant uses device scope atomics
  if(!res.m_physicalDeviceInfo.features12.vulkanMemoryModelDeviceScope)
  {
    m_config.useVulkanMemoryModel = false;
  }
  // both are consumers of the partitions, only this local copy is cleared so the requested
  // settings come back when partitions are re-enabled
  m_config.usePtlas = m_config.usePtlas && m_config.usePartitions && res.m_supportsPartitionedTlas;
  // without blas reuse `traversal_init` traverses every instance, the early-out would build
  // blas for skipped instances that are never assigned
  m_config.usePartitionEarlyOut = m_config.usePartitionEarlyOut && m_config.usePartitions && useBlasReuse();
  if(m_config.usePtlas)
  {
    // fixed grid and node regex layouts are not bounded by the device limit
    uint32_t maxPartitionCount = PartitionedTlas::getProperties(res.m_physicalDevice).maxPartitionCount;
    if(rscene.scene->m_partitions.size() > maxPartitionCount)
    {
      LOGW("ptlas disabled: %zu partitions exceed the device limit of %u\n", rscene.scene->m_partitions.size(), maxPartitionCount);
      m_config.usePtlas = false;
    }
  }

#if USE_DLSS
  res.setFramebufferDlss(m_config.useDlss, m_config.dlssQuality);
  m_config.useDlss = m_config.useDlss && res.m_frameBuffer.dlssMode == Resources::DlssMode::eRayReconstruction
                     && res.m_frameBuffer.dlssDenoiser.isActive();
#endif

  // lod clamping is built on top of hit-based streaming, so it only applies when that is on
  // (enabling lod alone does not implicitly turn hit streaming on)
  m_config.useHitStreamingLod = m_config.useHitStreamingLod && m_config.useHitStreaming && rscene.useStreaming;

  if(!initShaders(res, rscene))
  {
    LOGE("RendererRayTraceClustersLod shaders failed\n");
    return false;
  }

  if(!rscene.updateClasRequired(true))
  {
    LOGE("RendererRayTraceClustersLod rscene.updateClasRequired failed\n");
    return false;
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.resetCachedBlas();
  }

  initBasics(res, rscene);

  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.rtClasMemBytes     = rscene.getClasSize(true);
  m_resourceReservedUsage.operationsMemBytes = logMemoryUsage(rscene.getOperationsSize(), "operations", "rscene total");

  {
    // get ray tracing properties

    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_accProperties};
    m_accProperties.pNext = &m_rtProperties;
    m_rtProperties.pNext  = &m_rtClasProperties;
    vkGetPhysicalDeviceProperties2(res.m_physicalDevice, &prop2);

    VkDeviceSize scratchSize = 0;
    if(rscene.useStreaming)
    {
      scratchSize = rscene.sceneStreaming.getRequiredClasScratchSize();
      LOGI("raytracer: CLAS scratchsize %d KiB\n", uint32_t((scratchSize + 1023) / 1024));
    }

    if(!initRayTracingBlas(res, rscene, scratchSize))
    {
      LOGE("Resources exceeding max buffer allocation size\n");
      deinit(res);
      return false;
    }

    // the PTLAS instances are written later, they need the build data
    if(!m_config.usePtlas)
    {
      initRayTracingTlas(res, rscene, scratchSize);
    }
    else if(!initRayTracingPtlas(res, rscene, scratchSize))
    {
      LOGE("RendererRayTraceClustersLod PTLAS setup failed\n");
      deinit(res);
      return false;
    }

    // streaming also stores newly built clas in scratch, the ptlas build needs it 256 byte aligned
    res.createBuffer(m_scratchBuffer, scratchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0,
                     std::max<VkDeviceSize>(m_accProperties.minAccelerationStructureScratchOffsetAlignment,
                                            PartitionedTlas::ADDRESS_ALIGNMENT));
    NVVK_DBG_NAME(m_scratchBuffer.buffer);

    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_scratchBuffer.bufferSize, "operations", "rt scratch");

    // Update tlas build information
    m_tlasBuildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
    m_tlasBuildInfo.dstAccelerationStructure  = m_tlas.accel;
    m_tlasBuildInfo.scratchData.deviceAddress = m_scratchBuffer.address;
  }

  // scene building data

  {
    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    NVVK_DBG_NAME(m_sceneBuildBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneBuildBuffer.bufferSize, "operations", "build shaderio");

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.maxRenderClusters  = m_maxRenderClusters;
    m_sceneBuildShaderio.maxTraversalInfos  = uint32_t(1u << m_config.numTraversalTaskBits);
    m_sceneBuildShaderio.tlasInstances      = m_tlasInstancesBuffer.address;
    m_sceneBuildShaderio.numGeometries      = uint32_t(rscene.scene->getActiveGeometryCount());

    m_sceneBuildShaderio.indirectDispatchGroups.gridY        = 1;
    m_sceneBuildShaderio.indirectDispatchGroups.gridZ        = 1;
    m_sceneBuildShaderio.indirectDispatchBlasInsertion.gridY = 1;
    m_sceneBuildShaderio.indirectDispatchBlasInsertion.gridZ = 1;

    BufferRanges mem = {};
    m_sceneBuildShaderio.renderClusterInfos =
        mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);

    m_sceneBuildShaderio.instanceVisibility =
        mem.append(sizeof(uint8_t) * nvutils::align_up(m_renderInstances.size(), TRAVERSAL_INIT_WORKGROUP), 4);
    m_sceneBuildShaderio.instanceCachedSticky =
        mem.append(sizeof(uint8_t) * nvutils::align_up(m_renderInstances.size(), TRAVERSAL_INIT_WORKGROUP), 4);
    m_sceneBuildShaderio.blasBuildInfos = mem.append(sizeof(shaderio::BlasBuildInfo) * m_maxBlasBuilds, 16);
    if(m_config.usePtlas)
    {
      m_sceneBuildShaderio.ptlasInstanceUpdates =
          mem.append(sizeof(shaderio::PtlasInstanceUpdate) * m_renderInstances.size(), 16);
      m_sceneBuildShaderio.ptlasInstanceBlas = mem.append(sizeof(uint64_t) * m_renderInstances.size(), 8);
      m_sceneBuildShaderio.ptlasPartitionUpdates =
          mem.append(sizeof(uint32_t) * std::max(size_t(1), rscene.scene->m_partitions.size()), 4);
    }
    if(m_config.usePtlas || m_config.usePartitionEarlyOut)
    {
      m_sceneBuildShaderio.instancePartitions = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.partitionRanges =
          mem.append(sizeof(glm::uvec2) * std::max(size_t(1), rscene.scene->m_partitions.size()), 8);
    }
    if(m_config.usePartitionEarlyOut)
    {
      const size_t numPartitions                  = std::max(size_t(1), rscene.scene->m_partitions.size());
      m_sceneBuildShaderio.partitionInstanceOrder = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.partitionNonPersistent = mem.append(sizeof(uint32_t) * numPartitions, 4);
      m_sceneBuildShaderio.partitionLastFrame     = mem.append(sizeof(uint32_t) * numPartitions, 4);
      m_sceneBuildShaderio.partitionBboxes        = mem.append(sizeof(shaderio::BBox) * numPartitions, 16);
      m_sceneBuildShaderio.partitionVisible       = mem.append(sizeof(uint32_t) * numPartitions, 4);
      const uint32_t numBlocks =
          uint32_t((m_renderInstances.size() + PARTITION_INSTANCE_BLOCK_SIZE - 1) / PARTITION_INSTANCE_BLOCK_SIZE);
      m_sceneBuildShaderio.numInstanceBlocks  = numBlocks;
      m_sceneBuildShaderio.partitionBlockBits = mem.append(sizeof(uint32_t) * ((numBlocks + 31) / 32), 4);
      m_sceneBuildShaderio.activeBlocks       = mem.append(sizeof(uint32_t) * numBlocks, 4);
    }
    m_sceneBuildShaderio.instanceBuildInfos = mem.append(sizeof(shaderio::InstanceBuildInfo) * m_renderInstances.size(), 16);

    if(m_config.useHitStreamingLod)
    {
      m_sceneBuildShaderio.instanceHitDemand = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.instanceHitScale  = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.geometryHitDemandLevel = mem.append(sizeof(uint32_t) * m_sceneBuildShaderio.numGeometries, 4);
    }

    if(m_config.useBlasSharing)
    {
      m_sceneBuildShaderio.geometryBuildInfos =
          mem.append(sizeof(shaderio::GeometryBuildInfo) * m_sceneBuildShaderio.numGeometries, 16);
    }

    if(m_config.useBlasCaching)
    {
      m_sceneBuildShaderio.geometryCachedRetain = mem.append(sizeof(uint32_t) * m_sceneBuildShaderio.numGeometries, 4);
      m_sceneBuildShaderio.geometryCachedRetainNext = mem.append(sizeof(uint32_t) * m_sceneBuildShaderio.numGeometries, 4);
      m_sceneBuildShaderio.geometryCachedInfos =
          mem.append(sizeof(shaderio::GeometryCachedInfo) * m_sceneBuildShaderio.numGeometries, 8);

      m_sceneBuildShaderio.cachedBlasClusterAddressesDst =
          mem.append(sizeof(uint64_t) * rscene.sceneStreaming.getMaxCachedBlasBuilds(), 8);
      m_sceneBuildShaderio.cachedBlasClusterAddressesSrc =
          mem.append(sizeof(uint64_t) * rscene.sceneStreaming.getMaxCachedBlasBuilds(), 8);
    }

    if(m_config.useSorting)
    {
      // can alias some data required for sorting, with other data used at traversal/blas time.
      mem.beginOverlap();
      m_sceneBuildShaderio.instanceSortKeys   = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.instanceSortValues = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      mem.splitOverlap();
    }

    m_sceneBuildShaderio.traversalGroupInfos = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, 8);
    m_sceneBuildShaderio.blasBuildSizes      = mem.append(sizeof(uint32_t) * m_maxBlasBuilds, 4);
    m_sceneBuildShaderio.blasBuildAddresses  = mem.append(sizeof(uint64_t) * m_maxBlasBuilds, 8);

    m_sceneBuildShaderio.blasClusterAddresses = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxRenderClusters, 8);

    if(m_config.useSorting)
    {
      mem.endOverlap();
    }

    res.createBuffer(m_sceneDataBuffer, mem.getSize(),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    NVVK_DBG_NAME(m_sceneDataBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneDataBuffer.bufferSize, "operations", "build data");

    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceVisibility += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceCachedSticky += m_sceneDataBuffer.address;
    if(m_config.usePtlas || m_config.usePartitionEarlyOut)
    {
      m_sceneBuildShaderio.instancePartitions += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.partitionRanges += m_sceneDataBuffer.address;
    }
    if(m_config.usePartitionEarlyOut)
    {
      m_sceneBuildShaderio.partitionInstanceOrder += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.partitionNonPersistent += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.partitionLastFrame += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.partitionBboxes += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.partitionVisible += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.partitionBlockBits += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.activeBlocks += m_sceneDataBuffer.address;
    }
    m_sceneBuildShaderio.blasBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildSizes += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasClusterAddresses += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortKeys += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortValues += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceBuildInfos += m_sceneDataBuffer.address;
    if(m_config.usePtlas)
    {
      m_sceneBuildShaderio.ptlasInstanceUpdates += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.ptlasInstanceBlas += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.ptlasPartitionUpdates += m_sceneDataBuffer.address;
    }
    if(m_config.useHitStreamingLod)
    {
      m_sceneBuildShaderio.instanceHitDemand += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.instanceHitScale += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.geometryHitDemandLevel += m_sceneDataBuffer.address;
    }
    m_sceneBuildShaderio.traversalGroupInfos += m_sceneDataBuffer.address;
    if(m_config.useBlasSharing)
    {
      m_sceneBuildShaderio.geometryBuildInfos += m_sceneDataBuffer.address;
    }
    if(m_config.useBlasCaching)
    {
      m_sceneBuildShaderio.geometryCachedInfos += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.geometryCachedRetain += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.geometryCachedRetainNext += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.cachedBlasClusterAddressesDst += m_sceneDataBuffer.address;
      m_sceneBuildShaderio.cachedBlasClusterAddressesSrc += m_sceneDataBuffer.address;
    }

    // read before its first write
    {
      std::vector<uint8_t> zeros(m_renderInstances.size(), 0);
      res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.instanceCachedSticky - m_sceneDataBuffer.address,
                             zeros.size(), zeros.data());
    }

    res.createBuffer(m_sceneTraversalBuffer, sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(m_sceneTraversalBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneTraversalBuffer.bufferSize, "operations", "build traversal");

    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;

#if USE_LARGE_BUFFER_BLAS
    res.createLargeBuffer(m_sceneBlasDataBuffer, m_blasDataSize,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
#else
    res.createBuffer(m_sceneBlasDataBuffer, m_blasDataSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
#endif
    NVVK_DBG_NAME(m_sceneBlasDataBuffer.buffer);

    if(m_config.useBlasSharing)
    {
      res.createBuffer(m_sceneGeometryHistogramBuffer, sizeof(shaderio::GeometryBuildHistogram) * m_sceneBuildShaderio.numGeometries,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      NVVK_DBG_NAME(m_sceneGeometryHistogramBuffer.buffer);
      m_resourceReservedUsage.operationsMemBytes +=
          logMemoryUsage(m_sceneGeometryHistogramBuffer.bufferSize, "operations", "build geo");

      m_sceneBuildShaderio.geometryHistograms = m_sceneGeometryHistogramBuffer.address;
    }

    if(m_config.usePtlas || m_config.usePartitionEarlyOut)
    {
      initInstancePartitions(res, rscene);
    }

    // the shaders reach the scene by address, a ptlas has no object to bind
    if(!m_config.usePtlas)
    {
      m_sceneBuildShaderio.tlasAddress = m_tlas.address;
    }
    else
    {
      initRayTracingPtlasInstances(res, rscene);
    }
  }

  updateBasicDescriptors(res, rscene, &m_sceneBuildBuffer);

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.updateBindings(m_sceneBuildBuffer);
  }

  // use a single common descriptor set for all operations

  {
    m_stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                   | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDERMATERIALS_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    if(rscene.useStreaming)
    {
      bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
      bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    }
    bindings.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SKY_ENV_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SKY_IRRADIANCE_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_ENV_HDRI_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_ENV_HDRI_ACCEL_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDER_TARGET, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
#if USE_DLSS
    if(m_config.useDlss)
    {
      // skip first
      for(uint32_t i = 1; i < DlssDenoiser::eDlssCount; i++)
      {
        bindings.addBinding(BINDINGS_RENDER_TARGET + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
      }
    }
#endif

    m_dsetPack.init(bindings, res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout,
                               {m_dsetPack.getLayout(), rscene.sceneTextures.getDsetPack().getLayout()},
                               // build-setup passes push a single uint; the all-levels demand
                               // sweep pushes {reverseLevel, levelTotalGroups, geometryLevelOffsets}
                               {{m_stageFlags, 0, sizeof(uint32_t) * 2 + sizeof(uint64_t)}});

    nvvk::WriteSetContainer writeSets;
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), &res.m_commonBuffers.readBack);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERMATERIALS_SSBO), m_renderMaterialBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), &res.m_hizUpdate[0].farImageInfo);
    if(rscene.useStreaming)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_SSBO), rscene.sceneStreaming.getShaderStreamingBuffer());
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
    }
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SKY_ENV_TEX), res.getSkyEnvDescriptor());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SKY_IRRADIANCE_TEX), res.getSkyIrradianceDescriptor());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_ENV_HDRI_TEX), res.getEnvHdriDescriptor());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_ENV_HDRI_ACCEL_SSBO), res.getEnvHdriAccel());
#if USE_DLSS
    if(m_config.useDlss)
    {
      // apply all
      for(uint32_t i = 0; i < DlssDenoiser::eDlssCount; i++)
      {
        VkDescriptorImageInfo renderTargetInfo =
            res.m_frameBuffer.dlssDenoiser.getDescriptorImageInfo(DlssDenoiser::DlssBufferType(i));
        writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDER_TARGET + i), &renderTargetInfo);
      }
    }
    else
#endif
    {
      VkDescriptorImageInfo renderTargetInfo;
      renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      renderTargetInfo.imageView   = res.m_frameBuffer.imgColor.descriptor.imageView;
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDER_TARGET), &renderTargetInfo);
    }

    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  // initialize ray tracing pipeline

  initRayTracingPipeline(res);

  // initialize traversal pipeline

  {
    VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                 = "main";
    compInfo.layout                      = m_pipelineLayout;

    res.createComputePipeline(compInfo, m_shaders.computeBuildSetup, m_pipelines.computeBuildSetup);

    if(m_config.useSorting)
    {
      res.createComputePipeline(compInfo, m_shaders.computeTraversalPresort, m_pipelines.computeTraversalPresort);
    }

    res.createComputePipeline(compInfo, m_shaders.computeTraversalInit, m_pipelines.computeTraversalInit);

    res.createComputePipeline(compInfo, m_shaders.computeTraversalRun, m_pipelines.computeTraversalRun);

    res.createComputePipeline(compInfo, m_shaders.computeTraversalGroups, m_pipelines.computeTraversalGroups);

    res.createComputePipeline(compInfo, m_shaders.computeBlasInsertClusters, m_pipelines.computeBlasInsertClusters);

    res.createComputePipeline(compInfo, m_shaders.computeBlasSetupInsertion, m_pipelines.computeBlasSetupInsertion);

    res.createComputePipeline(compInfo, m_shaders.computeInstanceAssignBlas, m_pipelines.computeInstanceAssignBlas);

    if(useBlasReuse())
    {
      res.createComputePipeline(compInfo, m_shaders.computeInstanceClassifyLod, m_pipelines.computeInstanceClassifyLod);
    }

    if(m_config.useBlasSharing)
    {
      res.createComputePipeline(compInfo, m_shaders.computeGeometryBlasSharing, m_pipelines.computeGeometryBlasSharing);
      if(m_config.useBlasMerging)
      {
        res.createComputePipeline(compInfo, m_shaders.computeTraversalMerge, m_pipelines.computeTraversalMerge);
      }
    }
    if(m_config.useBlasCaching)
    {
      res.createComputePipeline(compInfo, m_shaders.computeGeometryCachedRetain, m_pipelines.computeGeometryCachedRetain);
    }
    if(m_config.usePartitionEarlyOut)
    {
      res.createComputePipeline(compInfo, m_shaders.computePartitionSetup, m_pipelines.computePartitionSetup);
      res.createComputePipeline(compInfo, m_shaders.computePartitionBlockCompact, m_pipelines.computePartitionBlockCompact);
    }
    if(m_config.useHitStreamingLod)
    {
      res.createComputePipeline(compInfo, m_shaders.computeInstanceHitScale, m_pipelines.computeInstanceHitScale);
    }

    if(m_config.useBlasCaching)
    {
      res.createComputePipeline(compInfo, m_shaders.computeBlasCachingSetupCopy, m_pipelines.computeBlasCachingSetupCopy);

      res.createComputePipeline(compInfo, m_shaders.computeBlasCachingSetupBuild, m_pipelines.computeBlasCachingSetupBuild);
    }
    // use m_config: it matches the normalized config the shaders were compiled with,
    // so the pipelines line up.
    if(rscene.useStreaming && m_config.useHitStreaming)
    {
      res.createComputePipeline(compInfo, m_shaders.computeHitPropagation, m_pipelines.computeHitPropagation);

      res.createComputePipeline(compInfo, m_shaders.computeHitDemand, m_pipelines.computeHitDemand);
    }
  }

  return true;
}

static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void RendererRayTraceClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  m_sceneBuildShaderio.errorOverDistanceThreshold = updateLodPixelError(res, rscene, frame);

  m_sceneBuildShaderio.traversalViewMatrix    = frame.traversalViewMatrix;
  m_sceneBuildShaderio.cullViewProjMatrix     = frame.cullViewProjMatrix;
  m_sceneBuildShaderio.cullViewProjMatrixLast = frame.cullViewProjMatrixLast;

  m_sceneBuildShaderio.frameIndex       = m_frameIndex;
  m_sceneBuildShaderio.culledErrorScale = std::max(1.0f, frame.culledErrorScale);
  m_sceneBuildShaderio.hitCameraErrorScale = m_config.useHitStreamingLod ? std::max(0.0f, frame.hitLodCameraScale) : 0.0f;
  m_sceneBuildShaderio.sharingPushCulled     = frame.sharingPushCulled;
  m_sceneBuildShaderio.sharingTolerantLevels = frame.sharingTolerantLevels;
  // the ui range, larger values would route instances without a cached blas to it
  m_sceneBuildShaderio.cachingPushLevels   = std::min(frame.cachingPushLevels, uint32_t(SHADERIO_MAX_LOD_LEVELS));
  m_sceneBuildShaderio.cachingStickyLevels = std::min(frame.cachingStickyLevels, uint32_t(SHADERIO_MAX_LOD_LEVELS));
  // The retention window is only honest while every partition is guaranteed to be visited
  // within it, so with the early-out on, forced revalidation is not optional.
  uint32_t revalidateFrames = frame.partitionRevalidateFrames;
  if(m_config.usePartitionEarlyOut && revalidateFrames == 0)
  {
    revalidateFrames = FrameConfig().partitionRevalidateFrames;
  }
  // restart the retention window once per revalidation cycle. Only the early-out skips
  // instances, without it the retention follows the current frame.
  uint32_t retainCycle                            = m_config.usePartitionEarlyOut ? std::max(1u, revalidateFrames) : 1u;
  m_sceneBuildShaderio.cachedRetainReset          = (m_frameIndex % retainCycle) == 0 ? 1 : 0;
  m_sceneBuildShaderio.partitionRevalidateFrames  = revalidateFrames;
  m_sceneBuildShaderio.partitionFrustumRevalidate = frame.partitionFrustumRevalidate ? 1 : 0;
  // the grid bounds cover all grid copies, a far plane is often set far beyond the scene
  const shaderio::BBox& sceneBbox = rscene.scene->m_gridBbox;
  m_sceneBuildShaderio.partitionFrustumDepth = frame.partitionFrustumDistance * glm::length(sceneBbox.hi - sceneBbox.lo);
  m_sceneBuildShaderio.partitionActiveCounter = 0;
  m_sceneBuildShaderio.activeBlockCounter     = 0;
  m_sceneBuildShaderio.numPartitions          = m_partitionCount;
  m_sceneBuildShaderio.sharingEnabledLevels   = frame.sharingEnabledLevels;

  // skyLighting is baked on the device
  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, offsetof(shaderio::FrameConstants, skyLighting),
                    (const uint32_t*)&frame.frameConstants);
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  if(m_config.usePtlas && m_partitionCount)
  {
    vkCmdFillBuffer(cmd, m_sceneDataBuffer.buffer, m_sceneBuildShaderio.ptlasPartitionUpdates - m_sceneDataBuffer.address,
                    sizeof(uint32_t) * m_partitionCount, 0);
  }
  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0);

  if(m_config.useBlasSharing)
  {
    vkCmdFillBuffer(cmd, m_sceneGeometryHistogramBuffer.buffer, 0, m_sceneGeometryHistogramBuffer.bufferSize, 0);
  }

  if(m_config.useBlasCaching && !m_config.useBlasSharing)
  {
    // required start value for the atomicMin on `cachedLevel` in `instance_classify_lod.comp.glsl`.
    // With sharing `geometry_blas_sharing.comp.glsl` writes the value outright.
    vkCmdFillBuffer(cmd, m_sceneDataBuffer.buffer, m_sceneBuildShaderio.geometryCachedInfos - m_sceneDataBuffer.address,
                    sizeof(shaderio::GeometryCachedInfo) * m_sceneBuildShaderio.numGeometries, ~0);
  }

  if(m_config.useHitStreamingLod && m_frameIndex == 0)
  {
    // one-time init of the per-instance finer-detail demand, afterwards `instance_hit_scale`
    // resets it every frame
    vkCmdFillBuffer(cmd, m_sceneDataBuffer.buffer, m_sceneBuildShaderio.instanceHitDemand - m_sceneDataBuffer.address,
                    sizeof(uint32_t) * m_renderInstances.size(), 0x7F7FFFFFu /* floatBitsToUint(FLT_MAX), == no demand */);
    vkCmdFillBuffer(cmd, m_sceneDataBuffer.buffer, m_sceneBuildShaderio.instanceHitScale - m_sceneDataBuffer.address,
                    sizeof(uint32_t) * m_renderInstances.size(), 0);
    vkCmdFillBuffer(cmd, m_sceneDataBuffer.buffer, m_sceneBuildShaderio.geometryHitDemandLevel - m_sceneDataBuffer.address,
                    sizeof(uint32_t) * m_sceneBuildShaderio.numGeometries, ~0u);
  }

  if(rscene.useStreaming)
  {
    SceneStreaming::FrameSettings settings;
    settings.freeze                = frame.freezeStreaming;
    settings.ageThreshold          = frame.streamingAgeThreshold;
    settings.unloadThreshold       = frame.streamingUnloadThreshold;
    settings.useBlasCaching        = m_config.useBlasCaching;
    settings.useHitStreaming       = m_config.useHitStreaming;
    settings.useHitStreamingLod    = m_config.useHitStreamingLod;
    settings.hitLodWidthFactor     = std::max(0.01f, frame.hitLodWidthFactor);
    settings.blasCacheFlags        = m_config.clusterBlasFlags;
    settings.blasCacheMaxClusters  = m_maxRenderClusters;
    settings.blasCacheMaxBuilds    = m_maxBlasBuilds;
    settings.blasCacheAgeThreshold = frame.cachingAgeThreshold;
    // deeper levels would share the last cached blas slot
    settings.blasCacheMinLevel = std::min(frame.cachingEnabledLevels, uint32_t(GEOMETRY_CACHED_BLAS_STRIDE));
    // an instance is only refreshed when its partition is processed, and forced revalidation
    // bounds that gap. Holding a replaced cached blas past it keeps a skipped instance from
    // ever meeting reused memory.
    settings.blasCacheDeferFrames = m_config.usePartitionEarlyOut ? revalidateFrames + 4 : 0;
    settings.blasCachePinGroups   = m_config.usePartitionEarlyOut;
    settings.blasCacheRetire      = m_config.usePartitionEarlyOut && frame.blasCacheRetire;

    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer, settings, profiler);

    if(m_config.useBlasCaching)
    {
      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();
      // can't add as much dynamic clusters as we reduce the budget with the once from the cached geometry builds
      m_sceneBuildShaderio.maxRenderClusters = m_maxRenderClusters - shaderData.update.patchCachedClustersCount;
    }
  }


  if(m_config.usePartitionEarlyOut)
  {
    // Only a cached blas freed without waiting out its hold can leave a skipped partition
    // pointing at reused memory. A patch on its own does not, see `deferCachedBlasFree`.
    bool unsafeFree = rscene.useStreaming && rscene.sceneStreaming.consumeUnsafeCachedBlasFree() != 0;
    m_sceneBuildShaderio.partitionInvalidateAll = (m_frameIndex == 0 || unsafeFree) ? 1 : 0;
  }

  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);


  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  res.cmdImageTransition(cmd, res.m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);


  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, m_scratchBuffer.address, profiler);
  }


  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Preparation");
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, 1,
                            rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);

    if(m_config.useSorting)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalPresort);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_PRESORT_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vrdxCmdSortKeyValue(cmd, res.m_vrdxSorter, m_sceneBuildShaderio.numRenderInstances, m_sceneDataBuffer.buffer,
                          m_sceneBuildShaderio.instanceSortKeys - m_sceneDataBuffer.address, m_sceneDataBuffer.buffer,
                          m_sceneBuildShaderio.instanceSortValues - m_sceneDataBuffer.address,
                          m_sortingAuxBuffer.buffer, 0, nullptr, 0);
    }

    if(m_config.useHitStreamingLod)
    {
      // must run before the partition setup, a changed scale wakes its partition
      auto timerSection = profiler.cmdFrameSection(cmd, "Instance Hit Scale");

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstanceHitScale);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, INSTANCE_HIT_SCALE_WORKGROUP));

      // the ray hit shaders atomicMin into the demand this reset
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(m_config.usePartitionEarlyOut)
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Partition Setup");

      vkCmdFillBuffer(cmd, m_sceneDataBuffer.buffer, m_sceneBuildShaderio.partitionBlockBits - m_sceneDataBuffer.address,
                      sizeof(uint32_t) * ((m_sceneBuildShaderio.numInstanceBlocks + 31) / 32), 0);

      memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier,
                           0, nullptr, 0, nullptr);

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computePartitionSetup);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_partitionCount, PARTITION_SETUP_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computePartitionBlockCompact);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numInstanceBlocks, PARTITION_BLOCK_COMPACT_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      // turn the compacted count into the dispatch the per-instance kernels use
      uint32_t buildSetupID = BUILD_SETUP_PARTITION_ACTIVE;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(useBlasReuse())
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Instance Classify");

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, 1,
                              rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstanceClassifyLod);
      if(m_config.usePartitionEarlyOut)
      {
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchActiveInstances));
      }
      else
      {
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, INSTANCES_CLASSIFY_LOD_WORKGROUP));
      }

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(m_config.useBlasSharing)
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Geometry Blas Sharing");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeGeometryBlasSharing);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numGeometries, GEOMETRY_BLAS_SHARING_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(m_config.useBlasCaching)
    {
      // `cachedLevel` is final for this frame, fold it into the retention value the
      // streaming age filter reads. See geometry_cached_retain.comp.glsl.
      auto timerSection = profiler.cmdFrameSection(cmd, "Geometry Cached Retain");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeGeometryCachedRetain);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numGeometries, GEOMETRY_CACHED_RETAIN_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(useBlasReuse() || m_config.useSorting)
    {
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    // we prepare traversal by filling in instance root nodes into the traversal queue
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, 1,
                            rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
    // only the blas reuse variant resolves its instance through `partitionInstanceOrder`,
    // `traversal_init.comp.glsl` indexes flat and must keep the linear dispatch
    if(m_config.usePartitionEarlyOut && useBlasReuse())
    {
      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchActiveInstances));
    }
    else
    {
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP));
    }

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in traversal queue

    uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Run");

    // this does the main traversal
    // it returns a list of render clusters

    if(m_config.usePersistentTraversal)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
    }
    else
    {
      // this is typically faster
      constexpr bool batchGroupsAtEnd = true;

      for(uint32_t p = 0; p < rscene.scene->m_maxNodeTreeDepth; p++)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchNodes));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        // first time we hit hiNodes == 1, is when we built the tree over the groups of highest detail lod
        // second time is when we link that node to the root node
        bool isLast = rscene.scene->m_maxNodeTreeDepth - 1 == p;

        uint32_t buildSetupID = !isLast && batchGroupsAtEnd ? BUILD_SETUP_TRAVERSAL_RUN_PASS_NODES_ONLY :
                                                              BUILD_SETUP_TRAVERSAL_RUN_PASS_COMBINED;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
        vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
        vkCmdDispatch(cmd, 1, 1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        if(buildSetupID == BUILD_SETUP_TRAVERSAL_RUN_PASS_COMBINED)
        {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
          vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
        }
      }
    }

    bool useBlasMerging = m_config.useBlasSharing && m_config.useBlasMerging;
    if(useBlasMerging)
    {
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(useBlasMerging)
    {
      // This kernel builds all clusters for merged instances based on residency of cluster groups.
      // It also does the age update for resident groups.

      assert(rscene.useStreaming);
      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalMerge);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(shaderData.resident.loGroupsCount + shaderData.resident.activeGroupsCount,
                                                   TRAVERSAL_BLAS_MERGING_WORKGROUP));
    }

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in render list

    uint32_t buildSetupID = BUILD_SETUP_BLAS_INSERTION;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);
  }

  if(rscene.useStreaming)
  {
    // This operation gives us new CLAS addresses

    // The age filter is skipped as it was handled during traversal already
    bool runAgeFilter = !(m_config.useBlasSharing && m_config.useBlasMerging);

    rscene.sceneStreaming.cmdPostTraversal(cmd, m_scratchBuffer.address, runAgeFilter, profiler);

    if(m_config.useHitStreaming)
    {
      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

      // faster stream-in demand propagation (STREAM_PROPAGATION_MODE, see docs/hitbased_streaming.md).
      // All modes fill the persistent per-group demand `groupDemand` that gates stream-in.
      //  - resident levels / subtree (BFS): the propagation pass seeds and enqueues the
      //    resident hit frontier; the demand pass descends the finer-overlap links. Resident
      //    levels stops after one hop, resident subtree loops to the max lod depth.
      //  - all levels (sweep): no queue; sweeps one reverse lod level per dispatch (coarsest
      //    first), so every group is visited exactly once per frame. The rolling variant
      //    dispatches only a single reverse level per frame instead of all of them.
      const bool useDemandBFS = m_config.streamPropagationMode != STREAM_PROPAGATION_ALL_LEVELS
                                && m_config.streamPropagationMode != STREAM_PROPAGATION_ALL_LEVELS_ROLLING;
      // persistent drain only for the resident-subtree BFS; must match the shader macro
      // gating in compileShaders (usePersistentDemandBFS).
      const bool usePersistentDemand =
          m_config.usePersistentHitDemand && m_config.streamPropagationMode == STREAM_PROPAGATION_RESIDENT_SUBTREE;
      // after traversal read it, before propagation and the descent refill it. See
      // `cmdClearGroupDemandSlice`, rides along with a transfer phase below when there is one.
      bool clearDemand = useDemandBFS;

      // reset the demand queue counters before the propagation pass seeds into them
      if(useDemandBFS)
      {
        uint32_t buildSetupID = BUILD_SETUP_HIT_DEMAND_INIT;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
        vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
        vkCmdDispatch(cmd, 1, 1, 1);

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }

      if(usePersistentDemand)
      {
        // The persistent demand kernel races consumers ahead of producers, so the queue's
        // produce region must read the ~0 "empty" sentinel until a producer writes it (same
        // mechanism as the frame-start clear the persistent traversal relies on). Traversal
        // already dirtied traversalNodeInfos, so re-clear it before the propagation seeds land.
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0u);
        if(clearDemand)
        {
          rscene.sceneStreaming.cmdClearGroupDemandSlice(cmd);
          clearDemand = false;
        }

        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }
      if(m_config.useHitStreamingLod)
      {
        // classify read last frame's value, propagation accumulates the next one
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        vkCmdFillBuffer(cmd, m_sceneDataBuffer.buffer, m_sceneBuildShaderio.geometryHitDemandLevel - m_sceneDataBuffer.address,
                        sizeof(uint32_t) * m_sceneBuildShaderio.numGeometries, ~0u);
        if(clearDemand)
        {
          rscene.sceneStreaming.cmdClearGroupDemandSlice(cmd);
          clearDemand = false;
        }

        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }
      if(clearDemand)
      {
        // neither transfer phase above ran, traversal read the demand this frame
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        rscene.sceneStreaming.cmdClearGroupDemandSlice(cmd);

        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }
      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Hit Propagation");

        // besides the hit-frame inheritance, this pass (for the BFS modes) also seeds
        // groupDemand and enqueues the resident hit frontier into the demand BFS queue.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeHitPropagation);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(shaderData.resident.activeGroupsCount + shaderData.resident.loGroupsCount,
                                                     STREAM_AGEFILTER_GROUPS_WORKGROUP));
      }
      {
        auto timerSection = profiler.cmdFrameSection(cmd, "Hit Demand");

        if(useDemandBFS)
        {
          // barrier: the propagation pass' seed writes (queue entries + counter and the
          // seeded groupDemand) must be visible to the setup and the first descent pass.
          memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                               &memBarrier, 0, nullptr, 0, nullptr);

          // set up the first frontier range (the seeds) and its indirect dispatch
          {
            uint32_t buildSetupID = BUILD_SETUP_HIT_DEMAND_RUN;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
            vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
            vkCmdDispatch(cmd, 1, 1, 1);

            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT
                                       | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                                 &memBarrier, 0, nullptr, 0, nullptr);
          }

          if(usePersistentDemand)
          {
            // one persistent dispatch drains the entire queue (producer/consumer), instead
            // of a host-synchronised dispatch per BFS level. BUILD_SETUP_HIT_DEMAND_RUN
            // handed the seeds to the traversal counters above.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeHitDemand);
            res.cmdLinearDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, STREAM_HIT_DEMAND_TREE_WORKGROUP));

            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &memBarrier, 0, nullptr, 0, nullptr);
          }
          else
          {
            // descend the frontier one BFS level per pass. The frontier empties once the
            // demand front stops advancing (freshness prune), so late passes dispatch 0
            // groups. Resident levels: one hop; resident subtree: up to the max lod depth.
            uint32_t passes = m_config.streamPropagationMode == STREAM_PROPAGATION_RESIDENT_LEVELS ?
                                  1u :
                                  std::max(1u, rscene.scene->m_maxLodLevelsCount);
            for(uint32_t pass = 0; pass < passes; pass++)
            {
              vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeHitDemand);
              vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchHitDemand));

              memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
              memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
              vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                   1, &memBarrier, 0, nullptr, 0, nullptr);

              uint32_t buildSetupID = BUILD_SETUP_HIT_DEMAND_RUN_PASS;
              vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
              vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
              vkCmdDispatch(cmd, 1, 1, 1);

              memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
              memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT
                                         | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
              vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                                   &memBarrier, 0, nullptr, 0, nullptr);
            }
          }
        }
        else
        {
          // all levels: sweep reverse lod levels coarsest->finest. A group's coarser overlaps
          // are one reverse level up; each group gathers their groupDemand at delta<=1.
          // backLodCount[k] = group count at reverse level k; the flat prefix offsets live in
          // geometryLevelOffsets.
          const auto& residentStats = rscene.sceneStreaming.getResidentStats();
          uint32_t    passes        = std::max(1u, rscene.scene->m_maxLodLevelsCount);

          struct
          {
            uint32_t        reverseLevel;
            uint32_t        levelTotalGroups;
            VkDeviceAddress geometryLevelOffsets;
          } pushConstants;
          pushConstants.geometryLevelOffsets = rscene.sceneStreaming.getGeometryLevelOffsetsAddress();

          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeHitDemand);

          // sweep the coarsest `coarseSwept` reverse levels in full every frame; ALL_LEVELS does
          // all of them, the rolling variant does only a UI-set count (cheap - coarse levels hold
          // few groups - giving coarse data single-frame stream-in reach) and advances one of the
          // remaining finer levels per frame below.
          uint32_t coarseSwept = passes;
          if(m_config.streamPropagationMode == STREAM_PROPAGATION_ALL_LEVELS_ROLLING)
            coarseSwept = std::max(1u, std::min(passes, frame.streamRollingCoarseLevels));

          for(uint32_t reverseLevel = 0; reverseLevel < coarseSwept; reverseLevel++)
          {
            // barrier: reverse level 0 reads the resident hit widths propagation wrote;
            // level k reads the groupDemand its coarser level k-1 just wrote.
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &memBarrier, 0, nullptr, 0, nullptr);

            pushConstants.reverseLevel     = reverseLevel;
            pushConstants.levelTotalGroups = residentStats.backLodCount[reverseLevel];
            vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(pushConstants), &pushConstants);
            res.cmdLinearDispatch(cmd, getWorkGroupCount(pushConstants.levelTotalGroups, STREAM_AGEFILTER_GROUPS_WORKGROUP));
          }

          // rolling: advance one finer level per frame (cycling coarseSwept..passes-1). Consecutive
          // frames advance consecutive levels, so a rolled level's coarser overlap was written last
          // frame (delta 1), or this frame when it is the first finer level reading the last swept
          // one. Phased by the streaming frameIndex, the clock the delta<=1 gather compares against.
          if(coarseSwept < passes)
          {
            uint32_t reverseLevel = coarseSwept + (shaderData.frameIndex % (passes - coarseSwept));

            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &memBarrier, 0, nullptr, 0, nullptr);

            pushConstants.reverseLevel     = reverseLevel;
            pushConstants.levelTotalGroups = residentStats.backLodCount[reverseLevel];
            vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(pushConstants), &pushConstants);
            res.cmdLinearDispatch(cmd, getWorkGroupCount(pushConstants.levelTotalGroups, STREAM_AGEFILTER_GROUPS_WORKGROUP));
          }
        }
      }
    }

    // no barrier needed here, given the critical barrier prior using these addresses
    // is directly prior running `m_pipelines.computeBlasInsertCluster`
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Blas Build Preparation");
    // this kernel prepares the per-blas clas reference list starting position.
    // it also resets the per-blas clas counters.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, 1,
                            rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasSetupInsertion);
    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, BLAS_SETUP_INSERTION_WORKGROUP));

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);

    // let's fill in the clusters from the unsorted render list, into the per-blas clas reference lists.

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasInsertClusters);
    vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchBlasInsertion));


    if(m_config.useBlasCaching)
    {
      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();
      if(shaderData.update.patchCachedBlasCount)
      {
        // seed blas builds
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, 1,
                                rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasCachingSetupBuild);
        // one work group per geometry
        res.cmdLinearDispatch(cmd, shaderData.update.patchCachedBlasCount);
      }
    }

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT
                               | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                             | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  if(rscene.useStreaming)
  {
    // initialize download work here so it can overlap with next
    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }

  // what is this? nah we never had any bugs in building and allocating the cluster data, totally not needed
#if !STREAMING_DEBUG_WITHOUT_RT
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Blas Build");

    // after we prepared the build information for the blas we can run it.

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    VkClusterAccelerationStructureInputInfoNV& inputs = cmdInfo.input;
    inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    // setup blas inputs
    inputs.maxAccelerationStructureCount = m_maxBlasBuilds;
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel  = &m_blasInput;
    inputs.flags                         = m_config.clusterBlasFlags;

    // input
    // we may actually build less BLAS than instances, due pre-build low detail blas or recycling
    cmdInfo.srcInfosCount = m_sceneBuildBuffer.address + offsetof(shaderio::SceneBuilding, blasBuildCounter);

    cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.blasBuildInfos;
    cmdInfo.srcInfosArray.size =
        sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * m_renderInstances.size();
    cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

    // output
    // the blas addresses are later provided to the ray instances
    cmdInfo.dstAddressesArray.deviceAddress = m_sceneBuildShaderio.blasBuildAddresses;
    cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_renderInstances.size();
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    // for statistics we keep track of blas sizes
    cmdInfo.dstSizesArray.deviceAddress = m_sceneBuildShaderio.blasBuildSizes;
    cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * m_renderInstances.size();
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    // in implicit mode we provide one big chunk from which outputs are sub-allocated
    cmdInfo.dstImplicitData = m_sceneBlasDataBuffer.address;

    cmdInfo.scratchData = m_scratchBuffer.address;

    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    if(m_config.useBlasCaching)
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Blas Copy");

      const shaderio::SceneStreaming& shaderData = rscene.sceneStreaming.getShaderStreamingData();
      if(shaderData.update.patchCachedBlasCount)
      {
        // prepare copy and then execute copy
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasCachingSetupCopy);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(shaderData.update.patchCachedBlasCount, BLAS_CACHING_SETUP_COPY_WORKGROUP));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &memBarrier, 0, nullptr, 0, nullptr);

        cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
        inputs  = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

        // setup move inputs
        inputs.maxAccelerationStructureCount = shaderData.update.patchCachedBlasCount;
        inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
        inputs.opInput.pMoveObjects          = &m_blasMoveInput;
        inputs.flags                         = 0;

        // input
        cmdInfo.srcInfosCount = m_sceneBuildBuffer.address + offsetof(shaderio::SceneBuilding, cachedBlasCopyCounter);

        cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.cachedBlasClusterAddressesSrc;
        cmdInfo.srcInfosArray.size = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV) * m_renderInstances.size();
        cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV);

        // output
        cmdInfo.dstAddressesArray.deviceAddress = m_sceneBuildShaderio.cachedBlasClusterAddressesDst;
        cmdInfo.dstAddressesArray.size          = sizeof(uint64_t) * m_renderInstances.size();
        cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

        cmdInfo.scratchData = m_scratchBuffer.address;

        vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

        memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }
    }
  }

  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Tlas Preparation");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeInstanceAssignBlas);
    if(m_config.usePartitionEarlyOut)
    {
      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchActiveInstances));
    }
    else
    {
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, INSTANCES_ASSIGN_BLAS_WORKGROUP));
    }

    // compute as well: the ptlas operation setup reads the update counter
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    // deliberately the same name for both, so a ptlas on/off comparison reads one timer
    auto timerSection = profiler.cmdFrameSection(cmd, "Tlas Build");

    if(m_config.usePtlas)
    {
      // the operation depends on how many instances changed their blas
      uint32_t buildSetupID = BUILD_SETUP_PTLAS_OPS;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);

      updateRayTracingPtlas(cmd);
    }
    else
    {
      updateRayTracingTlas(cmd, res, m_frameIndex != 0);
    }

    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                               | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);
  }
#endif

  // Ray trace
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "Render");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelines.rayTracing);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 1, 1,
                            rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);
#if !STREAMING_DEBUG_WITHOUT_RT
    vkCmdTraceRaysKHR(cmd, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                      frame.frameConstants.viewport.x, frame.frameConstants.viewport.y, 1);
#endif
    res.cmdBeginRendering(cmd, false, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    res.cmdImageTransition(cmd, res.m_frameBuffer.imgRaytracingDepth, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, true);

    writeRayTracingDepthBuffer(cmd);
    if(frame.showInstanceBboxes)
    {
      renderInstanceBboxes(cmd);
    }
    if(frame.showClusterBboxes)
    {
      renderClusterBboxes(cmd, m_sceneBuildBuffer, false);
    }
    vkCmdEndRendering(cmd);
  }


  if(!frame.freezeCulling)
  {
    res.cmdBuildHiz(cmd, frame, profiler, 0);
  }

#if USE_DLSS
  if(m_config.useDlss && res.m_frameBuffer.dlssMode == Resources::DlssMode::eRayReconstruction
     && res.m_frameBuffer.dlssDenoiser.isActive())
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "DLSS");

    // on failure the app disables DLSS before the next frame
    if(!res.m_frameBuffer.dlssDenoiser.denoise(cmd, frame.frameConstants.jitter, frame.frameConstants.viewMatrix,
                                               frame.frameConstants.projMatrix))
    {
      res.m_dlssEvaluateFailed = true;
    }
  }
#endif

  {
    // reservation for geometry may change
    m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);
    // reservation for blas may change
    m_resourceReservedUsage.rtBlasMemBytes = m_blasDataSize + rscene.getBlasSize(true);
    // reservation for clas may grow via sparse buffer resize
    m_resourceReservedUsage.rtClasMemBytes = rscene.getClasSize(true);

    m_resourceActualUsage                  = m_resourceReservedUsage;
    m_resourceActualUsage.geometryMemBytes = rscene.getGeometrySize(false);
    m_resourceActualUsage.rtClasMemBytes   = rscene.getClasSize(false);

    shaderio::Readback readback;
    res.getReadbackData(readback);
    m_resourceActualUsage.rtBlasMemBytes = readback.blasActualSizes + rscene.getBlasSize(false);
  }

  m_frameIndex++;
}

void RendererRayTraceClustersLod::deinit(Resources& res)
{
  deinitBasics(res);

  res.m_allocator.destroyBuffer(m_tlasInstancesBuffer);
  res.m_allocator.destroyBuffer(m_ptlasOperationsBuffer);
  m_ptlas.deinit(res);
  res.m_allocator.destroyBuffer(m_scratchBuffer);
  res.m_allocator.destroyAcceleration(m_tlas);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);
#if USE_LARGE_BUFFER_BLAS
  res.m_allocator.destroyLargeBuffer(m_sceneBlasDataBuffer);
#else
  res.m_allocator.destroyBuffer(m_sceneBlasDataBuffer);
#endif
  res.m_allocator.destroyBuffer(m_sceneGeometryHistogramBuffer);

  res.m_allocator.destroyBuffer(m_sbtBuffer);

  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);

  m_dsetPack.deinit();
  m_resourceReservedUsage = {};
}


bool RendererRayTraceClustersLod::initRayTracingBlas(Resources& res, RenderScene& rscene, VkDeviceSize& scratchSize)
{
  // BLAS space requirement (implicit)
  // the size of the generated blas is dynamic, need to query prebuild info.

  m_blasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
  // Just using m_hiPerGeometryClusters here is problematic, as the intermediate state
  // of a continuous lod can yield higher numbers (especially when streaming may temporarily cause overlapping of different levels).
  // Therefore, we use the highest sum of all clusters across all lod levels.
  m_blasInput.maxClusterCountPerAccelerationStructure = std::min(rscene.scene->m_maxPerGeometryClusters, m_maxRenderClusters);
  m_blasInput.maxTotalClusterCount = m_maxRenderClusters;

  if(m_config.useBlasSharing && m_config.useBlasMerging)
  {
    // we are guaranteeing only 2 BLAS per geometry that has multiple instances.
    // one through sharing, one through merging
    // the low-detail is pre-built.
    m_maxBlasBuilds = uint32_t(rscene.scene->getActiveGeometryCount() * 2);
  }
  else
  {
    m_maxBlasBuilds = uint32_t(m_renderInstances.size());
  }

  if(m_config.useBlasCaching)
  {
    // With caching we might build a few extra BLAS per-frame.
    // This value is at maximum `rscene.scene->getActiveGeometryCount()` plus some rounding/alignment.
    m_maxBlasBuilds += rscene.sceneStreaming.getMaxCachedBlasBuilds();
  }

  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = m_maxBlasBuilds;
  inputs.opMode                                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputs.opInput.pClustersBottomLevel = &m_blasInput;
  inputs.flags                        = m_config.clusterBlasFlags;

  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  scratchSize = std::max(scratchSize, sizesInfo.buildScratchSize);
  LOGI("raytracer: BLAS build scratchsize %d KiB\n", uint32_t((sizesInfo.buildScratchSize + 1023) / 1024));

  m_blasDataSize = sizesInfo.accelerationStructureSize;

  if(m_config.useBlasCaching)
  {
    const StreamingConfig& streamingConfig = rscene.sceneStreaming.getStreamingConfig();

    m_blasMoveInput               = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV};
    m_blasMoveInput.noMoveOverlap = VK_TRUE;
    m_blasMoveInput.maxMovedBytes = streamingConfig.maxBlasCachingMegaBytes * 1024 * 1024;
    m_blasMoveInput.type          = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_CLUSTERS_BOTTOM_LEVEL_NV;

    inputs.maxAccelerationStructureCount = rscene.sceneStreaming.getMaxCachedBlasBuilds();
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
    inputs.opInput.pMoveObjects          = &m_blasMoveInput;
    inputs.flags                         = 0;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    scratchSize = std::max(scratchSize, sizesInfo.buildScratchSize);
    LOGI("raytracer: BLAS move scratchsize %d KiB\n", uint32_t((sizesInfo.buildScratchSize + 1023) / 1024));
  }

  return true;
}

void RendererRayTraceClustersLod::initRayTracingPipeline(Resources& res)
{
  VkDevice device = res.m_device;

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissAO,
    eClosestHit,
    eClosestHitAO,
    // eAnyHit is last so it can be dropped from stageCount when the scene has no alpha-masked geometry
    eAnyHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  std::array<VkShaderModuleCreateInfo, eShaderGroupCount>        stageShaders{};
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stageShaders[s].sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  }
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stages[s].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[s].pNext = &stageShaders[s];
    stages[s].pName = "main";
  }

  stages[eRaygen].stage                = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stageShaders[eRaygen].codeSize       = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayGen);
  stageShaders[eRaygen].pCode          = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayGen);
  stages[eMiss].stage                  = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMiss].codeSize         = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMiss);
  stageShaders[eMiss].pCode            = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMiss);
  stages[eMissAO].stage                = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMissAO].codeSize       = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMissAO);
  stageShaders[eMissAO].pCode          = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMissAO);
  stages[eClosestHit].stage            = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stageShaders[eClosestHit].codeSize   = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayClosestHit);
  stageShaders[eClosestHit].pCode      = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayClosestHit);
  stages[eClosestHitAO].stage          = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stageShaders[eClosestHitAO].codeSize = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayClosestHitAO);
  stageShaders[eClosestHitAO].pCode    = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayClosestHitAO);
  // eAnyHit is the last stage; assigned (and counted) only when the scene has alpha-masked geometry
  if(m_hasAlphaMask)
  {
    stages[eAnyHit].stage          = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stageShaders[eAnyHit].codeSize = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayAnyHit);
    stageShaders[eAnyHit].pCode    = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayAnyHit);
  }

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             .generalShader      = VK_SHADER_UNUSED_KHR,
                                             .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                             .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                             .intersectionShader = VK_SHADER_UNUSED_KHR};

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // Miss Ao
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissAO;
  shaderGroups.push_back(group);

  // hit group
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  shaderGroups.push_back(group);

  // hit group with any-hit, for alpha-masked geometry (geometry index 1); present only when the scene has
  // such geometry. The AO ray's SBT record offset (render_shading.glsl) adapts to the primary hit-group
  // count so the AO groups below stay addressable either way.
  if(m_hasAlphaMask)
  {
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.anyHitShader     = eAnyHit;
    group.closestHitShader = eClosestHit;
    shaderGroups.push_back(group);
  }

  // closest hit shader for shadow hit tracking (opaque geometry index 0)
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.anyHitShader     = VK_SHADER_UNUSED_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHitAO;
  shaderGroups.push_back(group);

  // same for the alpha-masked geometry slot (geometry index 1),
  // which additionally runs the alpha-test any-hit shader
  if(m_hasAlphaMask)
  {
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.anyHitShader     = eAnyHit;
    group.closestHitShader = eClosestHitAO;
    shaderGroups.push_back(group);
  }

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .flags = res.getPipelineCreateFlags(),
      // eAnyHit is the last stage, so dropping it is just a smaller count
      .stageCount = uint32_t(m_hasAlphaMask ? eShaderGroupCount : eShaderGroupCount - 1),
      .pStages    = stages.data(),
      .groupCount = static_cast<uint32_t>(shaderGroups.size()),
      .pGroups    = shaderGroups.data(),
      // The path tracer issues all rays (primary, indirect, shadow) from the ray-gen shader;
      // hit/miss shaders never trace, so a recursion depth of 1 suffices.
      .maxPipelineRayRecursionDepth = m_config.usePathtrace ? 1u : (m_config.useShading ? 2u : 1u),
      .layout                       = m_pipelineLayout,
  };

  // NEW for clusters! we need to enable their usage explicitly for a ray tracing pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
  pipeClusters.allowClusterAccelerationStructure = true;

  rayPipelineInfo.pNext = &pipeClusters;

  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_pipelines.rayTracing));
  NVVK_DBG_NAME(m_pipelines.rayTracing);
  res.dumpPipelineInternals(m_pipelines.rayTracing, m_shaders.rayGen);

  // Creating the SBT
  {
    // Shader Binding Table (SBT) setup
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(res.m_device, m_rtProperties);

    // Prepare SBT data from ray pipeline
    size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_pipelines.rayTracing, rayPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(res.m_allocator.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
                                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    nvvk::StagingUploader uploader;
    uploader.init(&res.m_allocator);

    void* mapping = nullptr;
    NVVK_CHECK(uploader.appendBufferMapping(m_sbtBuffer, 0, bufferSize, mapping));
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_sbtBuffer.address, bufferSize, mapping));

    VkCommandBuffer cmd = res.createTempCmdBuffer();
    uploader.cmdUploadAppended(cmd);
    res.tempSyncSubmit(cmd);
    uploader.deinit();

    // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
    m_sbtRegions = sbtGenerator.getSBTRegions();

    sbtGenerator.deinit();
  }
}

void RendererRayTraceClustersLod::initRayTracingTlas(Resources& res, RenderScene& rscene, VkDeviceSize& scratchSize)
{
  std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(m_renderInstances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    VkAccelerationStructureInstanceKHR instance{};
    memcpy(&instance.transform, &m_renderInstances[i].worldMatrix, sizeof(VkTransformMatrixKHR));
    instance.instanceCustomIndex = static_cast<uint32_t>(i);  // gl_InstanceCustomIndexEX
    instance.mask                = 0xFF;                      // All objects
    instance.flags               = 0;

    // no need to use m_renderInstances[i].flipWinding, as ray tracing handles
    // negative determinants automatically
    if(m_config.flipWinding)
    {
      instance.flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
    }

    // Seeded with the geometry's pre-built low detail blas, which `initClas` has already built
    // for every geometry, so an instance the early-out skips from the first frame on is valid.
    instance.accelerationStructureReference =
        rscene.useStreaming ? rscene.sceneStreaming.getLowDetailBlasAddress(m_renderInstances[i].geometryID) :
                              rscene.scenePreloaded.getLowDetailBlasAddress(m_renderInstances[i].geometryID);
    tlasInstances[i] = instance;
  }

  // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
  res.createBuffer(m_tlasInstancesBuffer, tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_tlasInstancesBuffer.bufferSize, "operations", "rt instances");
  res.simpleUploadBuffer(m_tlasInstancesBuffer, tlasInstances.data());

  VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_tlasInstancesBuffer.buffer};
  VkDeviceAddress instBufferAddr = vkGetBufferDeviceAddress(res.m_device, &bufferInfo);

  // Wraps a device pointer to the above uploaded instances.
  VkAccelerationStructureGeometryInstancesDataKHR instancesVk{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
  instancesVk.data.deviceAddress = instBufferAddr;

  // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
  m_tlasGeometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  m_tlasGeometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  m_tlasGeometry.geometry.instances = instancesVk;

  // Find sizes
  m_tlasBuildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  m_tlasBuildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  m_tlasBuildInfo.geometryCount = 1;
  m_tlasBuildInfo.pGeometries   = &m_tlasGeometry;
  // FIXME
  m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  m_tlasBuildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

  uint32_t                                 instanceCount = uint32_t(m_renderInstances.size());
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(res.m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &m_tlasBuildInfo, &instanceCount, &sizeInfo);

  // Create TLAS
  VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  createInfo.size = sizeInfo.accelerationStructureSize;

  res.m_allocator.createAcceleration(m_tlas, createInfo);
  m_resourceReservedUsage.rtTlasMemBytes += createInfo.size;

  scratchSize = std::max(scratchSize, sizeInfo.buildScratchSize);
  LOGI("raytracer: TLAS build scratchsize %d KiB\n", uint32_t((sizeInfo.buildScratchSize + 1023) / 1024));
}

// layout of `m_ptlasOperationsBuffer`: the number of operations, followed by the operation itself
static const VkDeviceSize PTLAS_OPERATION_OFFSET = 16;

bool RendererRayTraceClustersLod::initRayTracingPtlas(Resources& res, RenderScene& rscene, VkDeviceSize& scratchSize)
{
  const std::vector<Scene::Partition>& partitions = rscene.scene->m_partitions;

  uint32_t maxInstancesPerPartition = 0;
  for(const Scene::Partition& partition : partitions)
  {
    maxInstancesPerPartition = std::max(maxInstancesPerPartition, partition.instanceCount);
  }

  PartitionedTlas::Config config;
  config.instanceCount            = uint32_t(m_renderInstances.size());
  config.partitionCount           = uint32_t(partitions.size());
  config.maxInstancesPerPartition = maxInstancesPerPartition;
  config.buildFlags               = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

  m_partitionCount = config.partitionCount;

  if(m_ptlas.init(res, config) != VK_SUCCESS)
  {
    return false;
  }
  m_resourceReservedUsage.rtTlasMemBytes += m_ptlas.getAccelerationStructureSize();

  // shares `m_scratchBuffer`, the ptlas build is its last use within the frame
  scratchSize = std::max(scratchSize, m_ptlas.getScratchSize());

  // the operation itself is filled in by `build_setup.comp.glsl`, it depends on the number
  // of instances that changed their blas
  res.createBuffer(m_ptlasOperationsBuffer, PTLAS_OPERATION_OFFSET + sizeof(shaderio::PtlasOperation),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  NVVK_DBG_NAME(m_ptlasOperationsBuffer.buffer);
  m_resourceReservedUsage.operationsMemBytes += m_ptlasOperationsBuffer.bufferSize;

  return true;
}

void RendererRayTraceClustersLod::initRayTracingPtlasInstances(Resources& res, RenderScene& rscene)
{
  const uint32_t instanceCount = uint32_t(m_renderInstances.size());

  m_sceneBuildShaderio.tlasAddress         = m_ptlas.getAddress();
  m_sceneBuildShaderio.ptlasOperationCount = m_ptlasOperationsBuffer.address;
  m_sceneBuildShaderio.ptlasOperations     = m_ptlasOperationsBuffer.address + PTLAS_OPERATION_OFFSET;

  // The full instance descriptions are written once here. Every frame after only updates the blas
  // reference of changed instances, so the descriptions don't need to stay resident.
  // An instance may only be updated if it was written with explicit bounds or without a blas.
  // Without explicit bounds it starts empty and frame 0 updates every instance.
  std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> instanceWrites(instanceCount);
  std::vector<uint64_t>                                              instanceBlas(instanceCount);

  for(uint32_t i = 0; i < instanceCount; i++)
  {
    const Scene::Instance& sceneInstance = rscene.scene->m_instances[i];

    instanceBlas[i] = !m_config.usePtlasExplicitAabb ? 0 :
                      rscene.useStreaming ? rscene.sceneStreaming.getLowDetailBlasAddress(m_renderInstances[i].geometryID) :
                                            rscene.scenePreloaded.getLowDetailBlasAddress(m_renderInstances[i].geometryID);

    VkPartitionedAccelerationStructureWriteInstanceDataNV& instanceWrite = instanceWrites[i];

    instanceWrite = {};
    memcpy(&instanceWrite.transform, &m_renderInstances[i].worldMatrix, sizeof(VkTransformMatrixKHR));
    instanceWrite.instanceID            = i;  // gl_InstanceCustomIndexEXT
    instanceWrite.instanceMask          = 0xFF;
    instanceWrite.instanceIndex         = i;
    instanceWrite.partitionIndex        = rscene.scene->m_instancePartitions[i];
    instanceWrite.accelerationStructure = instanceBlas[i];

    // no need to use m_renderInstances[i].flipWinding, as ray tracing handles
    // negative determinants automatically
    if(m_config.flipWinding)
    {
      instanceWrite.instanceFlags |= VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FLIP_FACING_BIT_NV;
    }

    if(m_config.usePtlasExplicitAabb)
    {
      // The geometry's k-DOP hull spans every lod level, so the bounds stay valid whichever
      // blas is referenced, and transforming its vertices gives the extent the spec asks for
      // rather than the bound of a transformed box.
      shaderio::BBox bbox = rscene.scene->getInstanceWorldBBox(sceneInstance);

      instanceWrite.explicitAABB[0] = bbox.lo.x;
      instanceWrite.explicitAABB[1] = bbox.lo.y;
      instanceWrite.explicitAABB[2] = bbox.lo.z;
      instanceWrite.explicitAABB[3] = bbox.hi.x;
      instanceWrite.explicitAABB[4] = bbox.hi.y;
      instanceWrite.explicitAABB[5] = bbox.hi.z;
      instanceWrite.instanceFlags |= VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_ENABLE_EXPLICIT_BOUNDING_BOX_NV;
    }
  }

  // `instance_assign_blas.comp.glsl` only emits updates for instances that differ from this
  res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.ptlasInstanceBlas - m_sceneDataBuffer.address,
                         sizeof(uint64_t) * instanceCount, instanceBlas.data());

  nvvk::Buffer writesBuffer;
  res.createBuffer(writesBuffer, sizeof(shaderio::PtlasInstanceWrite) * instanceCount,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  res.simpleUploadBuffer(writesBuffer, instanceWrites.data());

  struct
  {
    uint32_t                 count;
    uint32_t                 _pad[(PTLAS_OPERATION_OFFSET - sizeof(uint32_t)) / sizeof(uint32_t)];
    shaderio::PtlasOperation operation;
  } initialOps = {};
  static_assert(offsetof(decltype(initialOps), operation) == PTLAS_OPERATION_OFFSET);

  initialOps.count                = 1;
  initialOps.operation.opType     = PTLAS_OP_WRITE_INSTANCE;
  initialOps.operation.argCount   = instanceCount;
  initialOps.operation.argAddress = writesBuffer.address;
  initialOps.operation.argStride  = PTLAS_INSTANCE_WRITE_SIZE;
  res.simpleUploadBuffer(m_ptlasOperationsBuffer, 0, sizeof(initialOps), &initialOps);

  VkCommandBuffer cmd = res.createTempCmdBuffer();

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
                       1, &memBarrier, 0, nullptr, 0, nullptr);

  m_ptlas.cmdBuild(cmd, m_ptlasOperationsBuffer.address + PTLAS_OPERATION_OFFSET, m_ptlasOperationsBuffer.address,
                   m_scratchBuffer.address, false);
  res.tempSyncSubmit(cmd);

  res.m_allocator.destroyBuffer(writesBuffer);
}

void RendererRayTraceClustersLod::initInstancePartitions(Resources& res, RenderScene& rscene)
{
  const std::vector<Scene::Partition>& partitions = rscene.scene->m_partitions;

  m_partitionCount = uint32_t(partitions.size());

  // membership is fixed, so the assignment, the ordering and the ranges are uploaded once
  res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.instancePartitions - m_sceneDataBuffer.address,
                         sizeof(uint32_t) * m_renderInstances.size(),
                         const_cast<uint32_t*>(rscene.scene->m_instancePartitions.data()));

  // also used by the ptlas statistics
  std::vector<glm::uvec2> ranges(partitions.size());
  for(size_t i = 0; i < partitions.size(); i++)
  {
    ranges[i] = glm::uvec2(partitions[i].instanceOffset, partitions[i].instanceCount);
  }
  res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.partitionRanges - m_sceneDataBuffer.address,
                         sizeof(glm::uvec2) * ranges.size(), ranges.data());

  if(!m_config.usePartitionEarlyOut)
    return;

  res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.partitionInstanceOrder - m_sceneDataBuffer.address,
                         sizeof(uint32_t) * m_renderInstances.size(),
                         const_cast<uint32_t*>(rscene.scene->m_partitionInstanceOrder.data()));

  std::vector<shaderio::BBox> bboxes(partitions.size());
  for(size_t i = 0; i < partitions.size(); i++)
  {
    bboxes[i] = partitions[i].bbox;
  }
  res.simpleUploadBuffer(m_sceneDataBuffer, m_sceneBuildShaderio.partitionBboxes - m_sceneDataBuffer.address,
                         sizeof(shaderio::BBox) * bboxes.size(), bboxes.data());
}

void RendererRayTraceClustersLod::updateRayTracingPtlas(VkCommandBuffer cmd)
{
  m_ptlas.cmdBuild(cmd, m_ptlasOperationsBuffer.address + PTLAS_OPERATION_OFFSET, m_ptlasOperationsBuffer.address,
                   m_scratchBuffer.address, true);
}

void RendererRayTraceClustersLod::updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, bool update)
{
  if(update)
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = m_tlas.accel;
  }
  else
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  }

  // Build Offsets info: n instances
  VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{uint32_t(m_renderInstances.size()), 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

  // Build the TLAS
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &m_tlasBuildInfo, &pBuildOffsetInfo);
}

std::unique_ptr<Renderer> makeRendererRayTraceClustersLod()
{
  return std::make_unique<RendererRayTraceClustersLod>();
}

void RendererRayTraceClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  vkDeviceWaitIdle(res.m_device);

  nvvk::WriteSetContainer writeSets;

  writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), &res.m_hizUpdate[0].farImageInfo);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth);

#if USE_DLSS
  if(m_config.useDlss && res.m_frameBuffer.dlssMode == Resources::DlssMode::eRayReconstruction
     && res.m_frameBuffer.dlssDenoiser.isActive())
  {
    // apply all
    for(uint32_t i = 0; i < DlssDenoiser::eDlssCount; i++)
    {
      VkDescriptorImageInfo renderTargetInfo =
          res.m_frameBuffer.dlssDenoiser.getDescriptorImageInfo(DlssDenoiser::DlssBufferType(i));
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDER_TARGET + i), &renderTargetInfo);
    }
  }
  else
#endif
  {
    VkDescriptorImageInfo renderTargetInfo;
    renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    renderTargetInfo.imageView   = res.m_frameBuffer.imgColor.descriptor.imageView;
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDER_TARGET), &renderTargetInfo);
  }

  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res, rscene);
}

}  // namespace lodclusters
