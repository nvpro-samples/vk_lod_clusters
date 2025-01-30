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

#include <nvh/misc.hpp>
#include <nvh/alignment.hpp>

#include "renderer.hpp"
#include "shaders/shaderio.h"

namespace lodclusters {

class RendererRasterClustersTess : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) override;
  virtual void updatedFrameBuffer(Resources& res) override;
  virtual void deinit(Resources& res) override;

private:
  bool initShaders(Resources& res, const RenderScene& scene, const RendererConfig& config);

  struct Shaders
  {
    nvvk::ShaderModuleID graphicsMesh;
    nvvk::ShaderModuleID graphicsFragment;

    nvvk::ShaderModuleID computeTraversalInit;
    nvvk::ShaderModuleID computeTraversalRun;
    nvvk::ShaderModuleID computeBuildSetup;
  };

  struct Pipelines
  {
    VkPipeline graphicsMesh         = nullptr;
    VkPipeline computeTraversalInit = nullptr;
    VkPipeline computeTraversalRun  = nullptr;
    VkPipeline computeBuildSetup    = nullptr;
  };

  RendererConfig     m_config;
  Shaders            m_shaders;
  VkShaderStageFlags m_stageFlags;
  Pipelines          m_pipelines;

  nvvk::DescriptorSetContainer m_dsetContainer;

  RBuffer m_sceneBuildBuffer;
  RBuffer m_sceneTraversalBuffer;
  RBuffer m_sceneDataBuffer;

  shaderio::SceneBuilding m_sceneBuildShaderio;
};

bool RendererRasterClustersTess::initShaders(Resources& res, const RenderScene& rscene, const RendererConfig& config)
{
  std::string prepend;
  prepend += nvh::stringFormat("#define CLUSTER_VERTEX_COUNT %d\n",
                               shaderio::adjustClusterProperty(rscene.scene->m_clusterMaxVerticesCount));
  prepend += nvh::stringFormat("#define CLUSTER_TRIANGLE_COUNT %d\n",
                               shaderio::adjustClusterProperty(rscene.scene->m_config.clusterTriangles));
  prepend += nvh::stringFormat("#define TARGETS_RASTERIZATION %d\n", 1);
  prepend += nvh::stringFormat("#define USE_STREAMING %d\n", rscene.useStreaming ? 1 : 0);
  prepend += nvh::stringFormat("#define MESHSHADER_WORKGROUP_SIZE %d\n", 32);

  m_shaders.graphicsMesh =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters.mesh.glsl", prepend);
  m_shaders.graphicsFragment =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "render_raster.frag.glsl", prepend);
  m_shaders.computeTraversalInit =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", prepend);
  m_shaders.computeTraversalRun =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", prepend);
  m_shaders.computeBuildSetup =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", prepend);

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res);
}

bool RendererRasterClustersTess::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_config = config;

  if(!initShaders(res, rscene, config))
  {
    return false;
  }

  if(!rscene.updateClasRequired(false))
  {
    return false;
  }

  initBasics(res, rscene, config);

  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.operationsMemBytes = rscene.getOperationsSize();

  {
    m_sceneBuildBuffer = res.createBuffer(sizeof(shaderio::SceneBuilding), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                                               | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneBuildBuffer.info.range;

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.maxRenderClusters  = uint32_t(1u << config.numRenderClusterBits);
    m_sceneBuildShaderio.maxTraversalInfos  = uint32_t(1u << config.numTraversalTaskBits);

    m_sceneDataBuffer = res.createBuffer(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneDataBuffer.info.range;

    m_sceneBuildShaderio.renderClusterInfos = m_sceneDataBuffer.address;

    m_sceneTraversalBuffer =
        res.createBuffer(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_resourceReservedUsage.operationsMemBytes += m_sceneTraversalBuffer.info.range;

    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;
  }

  {
    m_dsetContainer.init(res.m_device);

    m_stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

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
    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  {
    nvvk::GraphicsPipelineState     state = res.m_basicGraphicsState;
    nvvk::GraphicsPipelineGenerator gfxGen(res.m_device, m_dsetContainer.getPipeLayout(),
                                           res.m_framebuffer.pipelineRenderingInfo, state);
    state.rasterizationState.frontFace = config.flipWinding ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    gfxGen.addShader(res.m_shaderManager.get(m_shaders.graphicsMesh), VK_SHADER_STAGE_MESH_BIT_NV);
    gfxGen.addShader(res.m_shaderManager.get(m_shaders.graphicsFragment), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_pipelines.graphicsMesh = gfxGen.createPipeline();
  }

  {
    VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                 = "main";
    compInfo.layout                      = m_dsetContainer.getPipeLayout();
    //compInfo.flags                       = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeTraversalInit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeTraversalRun);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);
  }

  return true;
}

static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void RendererRasterClustersTess::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  m_sceneBuildShaderio.traversalViewMatrix =
      frame.freezeCulling ? frame.frameConstantsLast.viewMatrix : frame.frameConstants.viewMatrix;
  m_sceneBuildShaderio.errorOverDistanceThreshold =
      nvclusterlod::pixelErrorToQuadricErrorOverDistance(frame.lodPixelError, frame.frameConstants.fov,
                                                         frame.frameConstants.viewportf.y);

  const bool useSky = true;  // When using Sky, the sky is rendered first and the rest of the scene is rendered on top of it.

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

  if(useSky)
  {
    res.m_sky.skyParams() = frame.frameConstants.skyParams;
    res.m_sky.updateParameterBuffer(cmd);
    res.cmdImageTransition(cmd, res.m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    res.m_sky.draw(cmd, frame.frameConstants.viewMatrix, frame.frameConstants.projMatrix, res.m_framebuffer.scissor.extent);
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, 0, profiler);
  }

  {
    auto timerSection = profiler.timeRecurring("Traversal Init", cmd);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetContainer.getPipeLayout(), 0, 1,
                            m_dsetContainer.getSets(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
    vkCmdDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP), 1, 1);

    // this barrier covers init & streaming pre traversal
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);
  }

  {
    auto timerSection = profiler.timeRecurring("Traversal Run", cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
    vkCmdDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP), 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    uint32_t buildSetupID = BUILD_SETUP_DRAW;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
    vkCmdPushConstants(cmd, m_dsetContainer.getPipeLayout(), m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
    vkCmdDispatch(cmd, 1, 1, 1);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPostTraversal(cmd, 0, profiler);
  }

  {
    auto timerSection = profiler.timeRecurring("Draw", cmd);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetContainer.getPipeLayout(), 0, 1,
                            m_dsetContainer.getSets(), 0, nullptr);

    res.cmdBeginRendering(cmd, false, useSky ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR);

    res.cmdDynamicState(cmd);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dsetContainer.getPipeLayout(), 0, 1,
                            m_dsetContainer.getSets(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMesh);
    vkCmdDrawMeshTasksIndirectNV(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClusters), 1, 0);

    vkCmdEndRendering(cmd);
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }

  if(!frame.freezeCulling)
  {
    res.cmdBuildHiz(cmd, frame, profiler);
  }

  // reservation for geometry may change
  m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);
  m_resourceActualUsage                    = m_resourceReservedUsage;
  m_resourceActualUsage.geometryMemBytes   = rscene.getGeometrySize(false);
}

void RendererRasterClustersTess::updatedFrameBuffer(Resources& res)
{
  vkDeviceWaitIdle(res.m_device);
  std::array<VkWriteDescriptorSet, 1> writeSets;
  writeSets[0] = m_dsetContainer.makeWrite(0, BINDINGS_HIZ_TEX, &res.m_hizUpdate.farImageInfo);
  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  Renderer::updatedFrameBuffer(res);
}

void RendererRasterClustersTess::deinit(Resources& res)
{
  vkDestroyPipeline(res.m_device, m_pipelines.graphicsMesh, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeTraversalInit, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeTraversalRun, nullptr);
  vkDestroyPipeline(res.m_device, m_pipelines.computeBuildSetup, nullptr);

  res.destroy(m_sceneDataBuffer);
  res.destroy(m_sceneBuildBuffer);
  res.destroy(m_sceneTraversalBuffer);

  m_dsetContainer.deinit();

  res.destroyShaders(m_shaders);

  deinitBasics(res);
}

std::unique_ptr<Renderer> makeRendererRasterClustersTess()
{
  return std::make_unique<RendererRasterClustersTess>();
}
}  // namespace lodclusters
