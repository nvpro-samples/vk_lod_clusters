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

#include <vector>

#include <nvvk/raytraceKHR_vk.hpp>

#include "renderer.hpp"
#include "shaders/shaderio.h"
#include "vk_nv_cluster_acc.h"


namespace lodclusters {

//////////////////////////////////////////////////////////////////////////

bool RenderScene::init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_)
{
  scene        = scene_;
  useStreaming = useStreaming_;

  if(useStreaming)
  {
    return sceneStreaming.init(res, scene_, streamingConfig_);
  }
  else
  {
    ScenePreloaded::Config preloadConfig;
    preloadConfig.clasBuildFlags           = streamingConfig_.clasBuildFlags;
    preloadConfig.clasPositionTruncateBits = streamingConfig_.clasPositionTruncateBits;
    return scenePreloaded.init(res, scene_, preloadConfig);
  }
}

void RenderScene::deinit()
{
  scenePreloaded.deinit();
  sceneStreaming.deinit();
}

void RenderScene::streamingReset()
{
  if(useStreaming)
  {
    sceneStreaming.reset();
  }
}

bool RenderScene::updateClasRequired(bool state)
{
  if(useStreaming)
  {
    return sceneStreaming.updateClasRequired(state);
  }
  else
  {
    return scenePreloaded.updateClasRequired(state);
  }
}

const RBufferTyped<shaderio::Geometry>& RenderScene::getShaderGeometriesBuffer() const
{

  if(useStreaming)
    return sceneStreaming.getShaderGeometriesBuffer();
  else
    return scenePreloaded.getShaderGeometriesBuffer();
}

size_t RenderScene::getClasSize(bool reserved) const
{
  if(useStreaming)
    return sceneStreaming.getClasSize(reserved);
  else
    return scenePreloaded.getClasSize();
}

size_t RenderScene::getGeometrySize(bool reserved) const
{
  if(useStreaming)
    return sceneStreaming.getGeometrySize(reserved);
  else
    return scenePreloaded.getGeometrySize();
}

size_t RenderScene::getOperationsSize() const
{
  if(useStreaming)
    return sceneStreaming.getOperationsSize();
  else
    return scenePreloaded.getOperationsSize();
}

//////////////////////////////////////////////////////////////////////////

bool Renderer::initBasicShaders(Resources& res)
{
  m_basicShaders.fullScreenVertexShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "fullscreen.vert.glsl");
  m_basicShaders.fullScreenWriteDepthFragShader =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_write_depth.frag.glsl");

  m_basicShaders.renderInstanceBboxesMeshShader =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, "render_instance_bbox.mesh.glsl");
  m_basicShaders.renderInstanceBboxesFragmentShader =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "render_instance_bbox.frag.glsl");

  return res.verifyShaders(m_basicShaders);
}

void Renderer::initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  const Scene& scene = *rscene.scene;

  m_renderInstances.resize(scene.m_instances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    shaderio::RenderInstance&  renderInstance = m_renderInstances[i];
    const uint32_t             geometryID     = scene.m_instances[i].geometryID;
    const Scene::GeometryView& geometry       = scene.getActiveGeometry(geometryID);

    renderInstance                = {};
    renderInstance.worldMatrix    = scene.m_instances[i].matrix;
    renderInstance.geometryID     = geometryID;
    renderInstance.maxLodLevelRcp = 1.0f / float(geometry.lodLevelsCount - 1);
  }

  m_renderInstanceBuffer =
      res.createBuffer(sizeof(shaderio::RenderInstance) * m_renderInstances.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  res.simpleUploadBuffer(m_renderInstanceBuffer, m_renderInstances.data());

  initWriteRayTracingDepthBuffer(res);
  initRenderInstanceBboxes(res, rscene);
}


void Renderer::deinitBasics(Resources& res)
{
  res.destroyShaders(m_basicShaders);

  vkDestroyPipeline(res.m_device, m_writeDepthBufferPipeline, nullptr);
  m_writeDepthBufferPipeline = VK_NULL_HANDLE;

  m_writeDepthBufferDsetContainer.deinit();

  res.destroy(m_renderInstanceBuffer);
}

void Renderer::updatedFrameBufferBasics(Resources& res)
{
  {
    std::array<VkWriteDescriptorSet, 1> writeSets;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    writeSets[0] = m_writeDepthBufferDsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &imgInfo);
    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }
}

void Renderer::initWriteRayTracingDepthBuffer(Resources& res)
{
  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  nvvk::DescriptorSetContainer& dsetContainer = m_writeDepthBufferDsetContainer;

  dsetContainer.init(res.m_device);

  dsetContainer.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  dsetContainer.initLayout();
  dsetContainer.initPipeLayout();

  dsetContainer.initPool(1);
  std::array<VkWriteDescriptorSet, 1> writeSets;

  VkDescriptorImageInfo imgInfo{};
  imgInfo.imageView   = res.m_framebuffer.viewRaytracingDepth;
  imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  writeSets[0] = dsetContainer.makeWrite(0, BINDINGS_RAYTRACING_DEPTH, &imgInfo);
  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  nvvk::GraphicsPipelineState state = res.m_basicGraphicsState;

  nvvk::GraphicsPipelineGenerator gfxGen(res.m_device, dsetContainer.getPipeLayout(), res.m_framebuffer.pipelineRenderingInfo, state);

  state.setBlendAttachmentColorMask(0, 0);
  state.depthStencilState.depthWriteEnable = VK_TRUE;
  state.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
  state.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
  gfxGen.addShader(res.m_shaderManager.get(m_basicShaders.fullScreenVertexShader), VK_SHADER_STAGE_VERTEX_BIT);
  gfxGen.addShader(res.m_shaderManager.get(m_basicShaders.fullScreenWriteDepthFragShader), VK_SHADER_STAGE_FRAGMENT_BIT);
  m_writeDepthBufferPipeline = gfxGen.createPipeline();
}

void Renderer::writeRayTracingDepthBuffer(VkCommandBuffer cmd)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_writeDepthBufferDsetContainer.getPipeLayout(), 0, 1,
                          m_writeDepthBufferDsetContainer.getSets(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_writeDepthBufferPipeline);

  vkCmdDraw(cmd, 3, 1, 0, 0);
}

void Renderer::initRenderInstanceBboxes(Resources& res, RenderScene& rscene)
{
  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT;

  nvvk::DescriptorSetContainer& dsetContainer = m_renderInstanceBboxesDsetContainer;

  dsetContainer.init(res.m_device);

  dsetContainer.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  dsetContainer.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  dsetContainer.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  dsetContainer.initLayout();

  VkPushConstantRange pushRange;
  pushRange.offset     = 0;
  pushRange.size       = sizeof(uint32_t);
  pushRange.stageFlags = stageFlags;
  dsetContainer.initPipeLayout(1, &pushRange);
  dsetContainer.initPool(1);

  std::array<VkWriteDescriptorSet, 3> writeSets;
  writeSets[0] = dsetContainer.makeWrite(0, BINDINGS_FRAME_UBO, &res.m_common.view.info);
  writeSets[1] = dsetContainer.makeWrite(0, BINDINGS_GEOMETRIES_SSBO, &rscene.getShaderGeometriesBuffer().info);
  writeSets[2] = dsetContainer.makeWrite(0, BINDINGS_RENDERINSTANCES_SSBO, &m_renderInstanceBuffer.info);
  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);

  nvvk::GraphicsPipelineState state = res.m_basicGraphicsState;
  nvvk::GraphicsPipelineGenerator gfxGen(res.m_device, dsetContainer.getPipeLayout(), res.m_framebuffer.pipelineRenderingInfo, state);
  state.rasterizationState.lineWidth = res.m_framebuffer.supersample * 2;
  //state.depthStencilState.depthWriteEnable = VK_TRUE;
  gfxGen.addShader(res.m_shaderManager.get(m_basicShaders.renderInstanceBboxesMeshShader), VK_SHADER_STAGE_MESH_BIT_NV);
  gfxGen.addShader(res.m_shaderManager.get(m_basicShaders.renderInstanceBboxesFragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
  m_renderInstanceBboxesPipeline = gfxGen.createPipeline();
}

void Renderer::renderInstanceBboxes(VkCommandBuffer cmd)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_renderInstanceBboxesDsetContainer.getPipeLayout(), 0,
                          1, m_renderInstanceBboxesDsetContainer.getSets(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_renderInstanceBboxesPipeline);

  uint32_t numRenderInstances = m_renderInstances.size();

  vkCmdPushConstants(cmd, m_renderInstanceBboxesDsetContainer.getPipeLayout(),
                     VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(uint32_t), &numRenderInstances);

  vkCmdDrawMeshTasksNV(cmd, (numRenderInstances + BBOXES_PER_MESHLET - 1) / BBOXES_PER_MESHLET, 0);
}

}  // namespace lodclusters
