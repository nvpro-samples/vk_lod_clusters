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

#include <nvvk/raytraceKHR_vk.hpp>
#include <nvvk/sbtwrapper_vk.hpp>
#include <nvvkhl/pipeline_container.hpp>
#include <nvvk/images_vk.hpp>
#include <nvh/parallel_work.hpp>
#include <nvh/misc.hpp>
#include <nvh/alignment.hpp>

#include "renderer.hpp"
#include "vk_nv_cluster_acc.h"

//////////////////////////////////////////////////////////////////////////

// temporary known perf issue in current implementation when
// indirect count << maxAccelerationCount from cpu
#define USE_INDIRECT_COUNT_PERF_WORKAROUND 1

//////////////////////////////////////////////////////////////////////////

namespace lodclusters {

class RendererRayTraceClustersLod : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res);

private:
  bool initShaders(Resources& res, const RenderScene& scene, const RendererConfig& config);

  void initRayTracingPipeline(Resources& res);
  bool initRayTracingBlas(Resources& res, RenderScene& scene, const RendererConfig& config, VkDeviceSize& scratchSize);
  void initRayTracingTlas(Resources& res, const RendererConfig& config, VkDeviceSize& scratchSize);

  void updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, bool update = false);


  struct Shaders
  {
    nvvk::ShaderModuleID rayGenShader;
    nvvk::ShaderModuleID closestHitShader;
    nvvk::ShaderModuleID missShader;
    nvvk::ShaderModuleID missShaderAO;

    nvvk::ShaderModuleID computeTraversalInit;
    nvvk::ShaderModuleID computeTraversalRun;
    nvvk::ShaderModuleID computeBuildSetup;

    nvvk::ShaderModuleID computeBlasInsertClusters;
    nvvk::ShaderModuleID computeBlasSetupInsertion;
  };

  struct Pipelines
  {
    VkPipeline computeTraversalInit      = nullptr;
    VkPipeline computeTraversalRun       = nullptr;
    VkPipeline computeBuildSetup         = nullptr;
    VkPipeline computeBlasInsertClusters = nullptr;
    VkPipeline computeBlasSetupInsertion = nullptr;
  };

  RendererConfig m_config;
  uint32_t       m_maxRenderClusters = 0;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV m_rtClasProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};

  nvvk::SBTWrapper          m_rtSbt;   // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;  // Hold pipelines and layout


  nvvk::DescriptorSetContainer m_dsetContainer;

  Shaders            m_shaders;
  Pipelines          m_pipelines;
  VkShaderStageFlags m_stageFlags;

  RBuffer                 m_sceneBuildBuffer;
  RBuffer                 m_sceneDataBuffer;
  RLargeBuffer            m_sceneBlasDataBuffer;
  RBuffer                 m_sceneTraversalBuffer;
  shaderio::SceneBuilding m_sceneBuildShaderio;

  VkClusterAccelerationStructureClustersBottomLevelInputNV m_blasInput;
  VkDeviceSize                                             m_blasDataSize = 0;

  bool                                        m_tlasDoBuild = true;
  RBuffer                                     m_tlasInstancesBuffer;
  VkAccelerationStructureGeometryKHR          m_tlasGeometry;
  VkAccelerationStructureBuildGeometryInfoKHR m_tlasBuildInfo;
  nvvk::AccelKHR                              m_tlas;

  RBuffer m_scratchBuffer;
};

bool RendererRayTraceClustersLod::initShaders(Resources& res, const RenderScene& rscene, const RendererConfig& config)
{
  std::string prepend;
  prepend += nvh::stringFormat("#define CLUSTER_VERTEX_COUNT %d\n",
                               shaderio::adjustClusterProperty(rscene.scene->m_config.clusterVertices));
  prepend += nvh::stringFormat("#define CLUSTER_TRIANGLE_COUNT %d\n",
                               shaderio::adjustClusterProperty(rscene.scene->m_config.clusterTriangles));
  prepend += nvh::stringFormat("#define TARGETS_RASTERIZATION %d\n", 0);
  prepend += nvh::stringFormat("#define USE_STREAMING %d\n", rscene.useStreaming ? 1 : 0);

  m_shaders.rayGenShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "render_raytrace.rgen.glsl");
  m_shaders.closestHitShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                                                      "render_raytrace_clusters.rchit.glsl", prepend);
  m_shaders.missShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                                "#define RAYTRACING_PAYLOAD_INDEX 0\n");
  m_shaders.missShaderAO = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                                  "#define RAYTRACING_PAYLOAD_INDEX 1\n");

  m_shaders.computeTraversalInit =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", prepend);
  m_shaders.computeTraversalRun =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", prepend);
  m_shaders.computeBuildSetup =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", prepend);
  m_shaders.computeBlasInsertClusters =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "blas_clusters_insert.comp.glsl", prepend);
  m_shaders.computeBlasSetupInsertion =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "blas_setup_insertion.comp.glsl", prepend);

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res);
}

bool RendererRayTraceClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_config                = config;
  m_maxRenderClusters     = 1u << config.numRenderClusterBits;
  m_resourceReservedUsage = {};

  if(!initShaders(res, rscene, config))
  {
    LOGE("RendererRayTraceClustersLod shaders failed\n");
    return false;
  }

  if(!rscene.updateClasRequired(true))
  {
    LOGE("RendererRayTraceClustersLod rscene.updateClasRequired failed\n");
    return false;
  }

  initBasics(res, rscene, config);

  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.rtClasMemBytes     = rscene.getClasSize(true);
  m_resourceReservedUsage.operationsMemBytes = rscene.getOperationsSize();

  {
    // get ray tracing properties

    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_rtProperties};
    m_rtProperties.pNext = &m_rtClasProperties;
    vkGetPhysicalDeviceProperties2(res.m_physical, &prop2);

    VkDeviceSize scratchSize = 0;
    if(rscene.useStreaming)
    {
      scratchSize = rscene.sceneStreaming.getRequiredClasScratchSize();
    }

    if(!initRayTracingBlas(res, rscene, config, scratchSize))
    {
      LOGE("Resources exceeding max buffer allocation size\n");
      deinit(res);
      return false;
    }

    // TLAS creation
    initRayTracingTlas(res, config, scratchSize);

    // streaming also stores newly built clas in scratch
    m_scratchBuffer = res.createBuffer(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceReservedUsage.operationsMemBytes += m_scratchBuffer.info.range;

    // Update tlas build information
    m_tlasBuildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
    m_tlasBuildInfo.dstAccelerationStructure  = m_tlas.accel;
    m_tlasBuildInfo.scratchData.deviceAddress = m_scratchBuffer.address;
  }

  // scene building data

  {
    m_sceneBuildBuffer = res.createBuffer(sizeof(shaderio::SceneBuilding), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                                               | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneBuildBuffer.info.range;

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.maxRenderClusters  = uint32_t(1u << config.numRenderClusterBits);
    m_sceneBuildShaderio.maxTraversalInfos  = uint32_t(1u << config.numTraversalTaskBits);
    m_sceneBuildShaderio.tlasInstances      = m_tlasInstancesBuffer.address;

    BufferRanges mem = {};
    m_sceneBuildShaderio.renderClusterInfos =
        mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);
    m_sceneBuildShaderio.instanceStates = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    m_sceneBuildShaderio.blasBuildInfos = mem.append(sizeof(shaderio::BlasBuildInfo) * m_renderInstances.size(), 16);
    m_sceneBuildShaderio.blasBuildSizes = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    m_sceneBuildShaderio.blasClusterAddresses = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxRenderClusters, 8);

    m_sceneDataBuffer = res.createBuffer(mem.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    m_resourceReservedUsage.operationsMemBytes += m_sceneDataBuffer.info.range;

    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceStates += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasBuildSizes += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.blasClusterAddresses += m_sceneDataBuffer.address;

    m_sceneTraversalBuffer =
        res.createBuffer(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneTraversalBuffer.info.range;

    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;

    m_sceneBlasDataBuffer = res.createLargeBuffer(m_blasDataSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceReservedUsage.rtBlasMemBytes += m_blasDataSize;
  }

  // use a single common descriptor set for all operations

  {
    m_dsetContainer.init(res.m_device);

    m_stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                   | VK_SHADER_STAGE_COMPUTE_BIT;

    m_dsetContainer.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, m_stageFlags);
    if(rscene.useStreaming)
    {
      m_dsetContainer.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
      m_dsetContainer.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    }
    m_dsetContainer.addBinding(BINDINGS_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_RENDER_TARGET, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
    m_dsetContainer.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
    m_dsetContainer.initLayout();

    VkPushConstantRange pushRange;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(uint32_t);
    pushRange.stageFlags = m_stageFlags;
    m_dsetContainer.initPipeLayout(1, &pushRange);

    m_dsetContainer.initPool(1);
    std::vector<VkWriteDescriptorSet> writeSets;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_FRAME_UBO, &res.m_common.view.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_READBACK_SSBO, &res.m_common.readbackDevice.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_GEOMETRIES_SSBO, &rscene.getShaderGeometriesBuffer().info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_RENDERINSTANCES_SSBO, &m_renderInstanceBuffer.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_SCENEBUILDING_SSBO, &m_sceneBuildBuffer.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_SCENEBUILDING_UBO, &m_sceneBuildBuffer.info));
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_HIZ_TEX, &res.m_hizUpdate.farImageInfo));
    if(rscene.useStreaming)
    {
      writeSets.push_back(
          m_dsetContainer.makeWrite(0, BINDINGS_STREAMING_SSBO, &rscene.sceneStreaming.getShaderStreamingBuffer().info));
      writeSets.push_back(
          m_dsetContainer.makeWrite(0, BINDINGS_STREAMING_UBO, &rscene.sceneStreaming.getShaderStreamingBuffer().info));
    }

    VkWriteDescriptorSetAccelerationStructureKHR accelInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    accelInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR accel     = m_tlas.accel;
    accelInfo.pAccelerationStructures    = &accel;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_TLAS, &accelInfo));

    VkDescriptorImageInfo renderTargetInfo;
    renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    renderTargetInfo.imageView   = res.m_framebuffer.viewColor;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_RENDER_TARGET, &renderTargetInfo));

    VkDescriptorImageInfo raytracingDepthInfo;
    raytracingDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    raytracingDepthInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;
    writeSets.push_back(m_dsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &raytracingDepthInfo));

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
    compInfo.layout                      = m_dsetContainer.getPipeLayout();
    compInfo.flags                       = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeTraversalInit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeTraversalRun);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBlasInsertClusters);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasInsertClusters);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBlasSetupInsertion);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasSetupInsertion);
  }

  return true;
}

static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void RendererRayTraceClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  m_sceneBuildShaderio.traversalViewMatrix =
      frame.freezeCulling ? frame.frameConstantsLast.viewMatrix : frame.frameConstants.viewMatrix;
  m_sceneBuildShaderio.errorOverDistanceThreshold =
      nvclusterlod::pixelErrorToQuadricErrorOverDistance(frame.lodPixelError, frame.frameConstants.fov,
                                                         frame.frameConstants.viewportf.y);

  vkCmdUpdateBuffer(cmd, res.m_common.view.buffer, 0, sizeof(shaderio::FrameConstants) * 2, (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(cmd, res.m_common.readbackDevice.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.info.range, ~0);

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer,
                                        frame.streamingAgeThreshold, profiler);
  }


  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);
  res.cmdImageTransition(cmd, res.m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);


  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, m_scratchBuffer.address, profiler);
  }

  {
    auto timerSection = profiler.timeRecurring("Traversal Init", cmd);

    // we prepare traversal by filling in instance root nodes into the traversal queue
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetContainer.getPipeLayout(), 0, 1,
                            m_dsetContainer.getSets(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
    vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP), 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in traversal queue

    uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.timeRecurring("Traversal Run", cmd);

    // this does the main traversal
    // it returns a list of render clusters

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
    vkCmdDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP), 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // fixup kernel for counters in case we tried to add more than available space in render list

    uint32_t buildSetupID = BUILD_SETUP_BLAS_INSERTION;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);
  }

  if(rscene.useStreaming)
  {
    // this operation gives us new CLAS addresses
    rscene.sceneStreaming.cmdPostTraversal(cmd, m_scratchBuffer.address, profiler);

    // no barrier needed here, given the critical barrier prior using these addresses
    // is directly prior running `m_pipelines.computeBlasInsertCluster`
  }

  {
    auto timerSection = profiler.timeRecurring("Blas Build Preparation", cmd);

    // this kernel prepares the per-blas clas reference list starting position.
    // it also resets the per-blas clas counters.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetContainer.getPipeLayout(), 0, 1,
                            m_dsetContainer.getSets(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasSetupInsertion);
    vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, BLAS_SETUP_INSERTION_WORKGROUP), 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);

    // let's fill in the clusters from the unsorted render list, into the per-blas clas reference lists.

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasInsertClusters);
    vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchBlasInsertion));

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
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
    auto timerSection = profiler.timeRecurring("Blas Build", cmd);

    // after we prepared the build information for the blas we can run it.

    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

    // setup blas inputs
    inputs.maxAccelerationStructureCount = uint32_t(m_renderInstances.size());
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel  = &m_blasInput;
    inputs.flags                         = m_config.clusterBlasFlags;

    // we feed the generated blas addresses directly into the ray instances
    cmdInfo.dstAddressesArray.deviceAddress =
        m_tlasInstancesBuffer.address + offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference);
    cmdInfo.dstAddressesArray.size   = m_tlasInstancesBuffer.info.range;
    cmdInfo.dstAddressesArray.stride = sizeof(VkAccelerationStructureInstanceKHR);

    // for statistics we keep track of blas sizes
    cmdInfo.dstSizesArray.deviceAddress = m_sceneBuildShaderio.blasBuildSizes;
    cmdInfo.dstSizesArray.size          = sizeof(uint32_t) * m_renderInstances.size();
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = m_sceneBuildShaderio.blasBuildInfos;
    cmdInfo.srcInfosArray.size =
        sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * m_renderInstances.size();
    cmdInfo.srcInfosArray.stride = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

    // in implicit mode we provide one big chunk from which outputs are sub-allocated
    cmdInfo.dstImplicitData = m_sceneBlasDataBuffer.address;

    cmdInfo.scratchData = m_scratchBuffer.address;
    cmdInfo.input       = inputs;

    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.timeRecurring("Tlas Build", cmd);

    updateRayTracingTlas(cmd, res, !m_tlasDoBuild);
    m_tlasDoBuild = false;

    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }
#endif

  // Ray trace
  {
    auto timerSection = profiler.timeRecurring("Render", cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, 1, m_dsetContainer.getSets(), 0, nullptr);
#if !STREAMING_DEBUG_WITHOUT_RT
    const std::array<VkStridedDeviceAddressRegionKHR, 4>& bindingTables = m_rtSbt.getRegions();
    vkCmdTraceRaysKHR(cmd, &bindingTables[0], &bindingTables[1], &bindingTables[2], &bindingTables[3],
                      frame.frameConstants.viewport.x, frame.frameConstants.viewport.y, 1);
#endif
    res.cmdBeginRendering(cmd, false, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    res.cmdDynamicState(cmd);
    writeRayTracingDepthBuffer(cmd);
    vkCmdEndRendering(cmd);
  }


  if(!frame.freezeCulling)
  {
    res.cmdBuildHiz(cmd, frame, profiler);
  }

  {
    // reservation for geometry may change
    m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);

    m_resourceActualUsage                  = m_resourceReservedUsage;
    m_resourceActualUsage.geometryMemBytes = rscene.getGeometrySize(false);
    m_resourceActualUsage.rtClasMemBytes   = rscene.getClasSize(false);

    shaderio::Readback readback;
    res.getReadbackData(readback);
    m_resourceActualUsage.rtBlasMemBytes = readback.blasActualSizes;
  }
}

void RendererRayTraceClustersLod::deinit(Resources& res)
{
  deinitBasics(res);

  res.destroy(m_tlasInstancesBuffer);
  res.destroy(m_scratchBuffer);
  res.destroy(m_tlas);

  res.destroy(m_sceneBuildBuffer);
  res.destroy(m_sceneDataBuffer);
  res.destroy(m_sceneTraversalBuffer);
  res.destroy(m_sceneBlasDataBuffer);


  m_rtSbt.destroy();               // Shading binding table wrapper
  m_rtPipe.destroy(res.m_device);  // Hold pipelines and layout

  vkDestroyPipeline(res.m_device, m_pipelines.computeBlasInsertClusters, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeBlasSetupInsertion, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeTraversalInit, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeTraversalRun, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeBuildSetup, nullptr);

  res.destroyShaders(m_shaders);

  m_dsetContainer.deinit();
  m_resourceReservedUsage = {};
}


bool RendererRayTraceClustersLod::initRayTracingBlas(Resources& res, RenderScene& rscene, const RendererConfig& config, VkDeviceSize& scratchSize)
{
  // BLAS space requirement (implicit)
  // the size of the generated blas is dynamic, need to query prebuild info.

  uint32_t numInstances = (uint32_t)m_renderInstances.size();

  m_blasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
  // Just using m_hiPerGeometryClusters here is problematic, as the intermediate state
  // of a continuous lod can yield higher numbers (especially when streaming may temporarily cause overlapping of different levels).
  // Therefore, we use the highest sum of all clusters across all lod levels.
  m_blasInput.maxClusterCountPerAccelerationStructure = rscene.scene->m_maxPerGeometryClusters;
  m_blasInput.maxTotalClusterCount                    = m_maxRenderClusters;

  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = numInstances;
  inputs.opMode                                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputs.opInput.pClustersBottomLevel = &m_blasInput;
  inputs.flags                        = config.clusterBlasFlags;

  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  scratchSize = std::max(scratchSize, sizesInfo.buildScratchSize);

  m_blasDataSize = sizesInfo.accelerationStructureSize;

  return true;
}

void RendererRayTraceClustersLod::initRayTracingPipeline(Resources& res)
{
  nvvkhl::PipelineContainer& p = m_rtPipe;
  p.plines.resize(1);

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissAO,
    eClosestHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  for(auto& s : stages)
    s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

  stages[eRaygen].module     = res.m_shaderManager.getShaderModule(m_shaders.rayGenShader).module;
  stages[eRaygen].pName      = "main";
  stages[eRaygen].stage      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eMiss].module       = res.m_shaderManager.getShaderModule(m_shaders.missShader).module;
  stages[eMiss].pName        = "main";
  stages[eMiss].stage        = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMissAO].module     = res.m_shaderManager.getShaderModule(m_shaders.missShaderAO).module;
  stages[eMissAO].pName      = "main";
  stages[eMissAO].stage      = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eClosestHit].module = res.m_shaderManager.getShaderModule(m_shaders.closestHitShader).module;
  stages[eClosestHit].pName  = "main";
  stages[eClosestHit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // Miss AO
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissAO;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  shaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  //const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> dsetLayouts = {m_dsetContainer.getLayout()};  // , m_pContainer[eGraphic].dstLayout};
  VkPipelineLayoutCreateInfo layoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutCreateInfo.setLayoutCount         = static_cast<uint32_t>(dsetLayouts.size());
  layoutCreateInfo.pSetLayouts            = dsetLayouts.data();
  layoutCreateInfo.pushConstantRangeCount = 0;  //1;
  //pipeline_layout_create_info.pPushConstantRanges    = &push_constant,

  vkCreatePipelineLayout(res.m_device, &layoutCreateInfo, nullptr, &p.layout);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};

  pipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages                      = stages.data();
  pipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
  pipelineInfo.pGroups                      = shaderGroups.data();
  pipelineInfo.maxPipelineRayRecursionDepth = 2;
  pipelineInfo.layout                       = p.layout;
  pipelineInfo.flags                        = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

  // new for clusters
  {
    pipelineInfo.pNext                              = &pipeClusters;
    pipeClusters.allowClusterAccelerationStructures = true;
  }

  VkResult result = vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &pipelineInfo, nullptr, &p.plines[0]);

  // Creating the SBT
  m_rtSbt.setup(res.m_device, res.m_queueFamily, &res.m_allocator, m_rtProperties);
  m_rtSbt.create(p.plines[0], pipelineInfo);
}

void RendererRayTraceClustersLod::initRayTracingTlas(Resources& res, const RendererConfig& config, VkDeviceSize& scratchSize)
{
  std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(m_renderInstances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform                              = nvvk::toTransformMatrixKHR(m_renderInstances[i].worldMatrix);
    instance.instanceCustomIndex                    = static_cast<uint32_t>(i);  // gl_InstanceCustomIndexEX
    instance.mask                                   = 0xFF;                      // All objects
    instance.instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
        instance.flags                              = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    if(config.flipWinding)
    {
      instance.flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
    }
    // patched in at render time
    instance.accelerationStructureReference = 0;
    tlasInstances[i]                        = instance;
  }

  // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
  m_tlasInstancesBuffer = res.createBuffer(tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceReservedUsage.operationsMemBytes += tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);
  res.simpleUploadBuffer(m_tlasInstancesBuffer, tlasInstances.data());
  res.tempResetResources();

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

  m_tlas = res.createAccelKHR(createInfo);
  m_resourceReservedUsage.rtTlasMemBytes += createInfo.size;

  scratchSize = std::max(scratchSize, sizeInfo.buildScratchSize);
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

std::unique_ptr<Renderer> makeRendererRayTraceClustersTess()
{
  return std::make_unique<RendererRayTraceClustersLod>();
}

void RendererRayTraceClustersLod::updatedFrameBuffer(Resources& res)
{
  vkDeviceWaitIdle(res.m_device);
  std::array<VkWriteDescriptorSet, 3> writeSets;
  VkDescriptorImageInfo               renderTargetInfo;
  renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  renderTargetInfo.imageView   = res.m_framebuffer.viewColor;
  writeSets[0]                 = m_dsetContainer.makeWrite(0, BINDINGS_RENDER_TARGET, &renderTargetInfo);

  VkDescriptorImageInfo raytracingDepthInfo;
  raytracingDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  raytracingDepthInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;

  writeSets[1] = m_dsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &raytracingDepthInfo);
  writeSets[2] = m_dsetContainer.makeWrite(0, BINDINGS_HIZ_TEX, &res.m_hizUpdate.farImageInfo);

  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res);
}

}  // namespace lodclusters
