/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvvk/sbt_generator.hpp>

#include "renderer.hpp"
#include "partitioned_acceleration_structures.hpp"

#define USE_LARGE_BUFFER_BLAS 1

namespace lodclusters {

class RendererRayTraceClustersLod : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) override;

protected:
  // set by RendererHybridClustersLod: the traversal also emits the clusters to rasterize
  bool m_hybrid = false;

  // RendererHybridClustersLod's raster passes, run after the tlas build and before the trace
  virtual bool initRasterShaders(Resources& res, RenderScene& rscene, const shaderc::CompileOptions& options)
  {
    return true;
  }
  virtual void initRasterPipelines(Resources& res) {}
  virtual void renderRaster(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
  {
  }

  // small clusters of the hybrid raster passes go to the compute raster
  bool useHybridSWRaster() const { return m_hybrid && m_config.useComputeRaster && m_config.useCulling; }

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
    Shader computeBlasDedupKeys;
    Shader computeBlasDedupScan;
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

    Shader rayQuery;
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
    VkPipeline computeBlasDedupKeys         = nullptr;
    VkPipeline computeBlasDedupScan         = nullptr;
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

    VkPipeline rayQuery = nullptr;
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
  nvvk::Buffer            m_sceneBlasDedupTableBuffer;
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

}  // namespace lodclusters
