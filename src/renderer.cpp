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

#include <random>
#include <vector>

#include <volk.h>
#include <fmt/format.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "renderer.hpp"
#include "../shaders/shaderio.h"


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

const nvvk::BufferTyped<shaderio::Geometry>& RenderScene::getShaderGeometriesBuffer() const
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

size_t RenderScene::getBlasSize(bool reserved) const
{
  if(useStreaming)
    return sceneStreaming.getBlasSize(reserved);
  else
    return scenePreloaded.getBlasSize();
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

bool Renderer::initBasicShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  uint32_t maxPrimitiveOutputs = config.useEXTmeshShader ? res.m_meshShaderPropsEXT.maxMeshOutputPrimitives :
                                                           res.m_meshShaderPropsNV.maxMeshOutputPrimitives;
  uint32_t maxVertexOutputs    = config.useEXTmeshShader ? res.m_meshShaderPropsEXT.maxMeshOutputVertices :
                                                           res.m_meshShaderPropsNV.maxMeshOutputVertices;


  if(config.useEXTmeshShader)
  {
    // don't go over 128, shaderc is using that limit and will fail to compile otherwise
    m_meshShaderWorkgroupSize = std::min(128u, std::min(res.m_meshShaderPropsEXT.maxPreferredMeshWorkGroupInvocations,
                                                        std::min(res.m_meshShaderPropsEXT.maxMeshWorkGroupSize[0],
                                                                 res.m_meshShaderPropsEXT.maxMeshWorkGroupInvocations)));
  }
  else
  {
    // set to 32, shaderc is using that limits and will fail to compile otherwise
    m_meshShaderWorkgroupSize = 32u;
  }

  m_meshShaderBoxes =
      std::min(m_meshShaderWorkgroupSize / MESHSHADER_BBOX_THREADS,
               std::min(maxPrimitiveOutputs / MESHSHADER_BBOX_LINES, maxVertexOutputs / MESHSHADER_BBOX_VERTICES));

  shaderc::CompileOptions options = res.makeCompilerOptions();
  options.AddMacroDefinition("USE_EXT_MESH_SHADER", fmt::format("{}", config.useEXTmeshShader ? 1 : 0));
  options.AddMacroDefinition("USE_STREAMING", fmt::format("{}", rscene.useStreaming ? 1 : 0));
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", fmt::format("{}", m_meshShaderWorkgroupSize));
  options.AddMacroDefinition("MESHSHADER_BBOX_COUNT", fmt::format("{}", m_meshShaderBoxes));

  res.compileShader(m_basicShaders.fullScreenVertexShader, VK_SHADER_STAGE_VERTEX_BIT, "fullscreen.vert.glsl");
  if(res.m_supportsClusterRaytracing)
  {
    res.compileShader(m_basicShaders.fullScreenWriteDepthFragShader, VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_write_depth.frag.glsl");
  }
  res.compileShader(m_basicShaders.fullScreenBackgroundFragShader, VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_background.frag.glsl");
  res.compileShader(m_basicShaders.fullscreenAtomicRasterFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT,
                    "fullscreen_raster_atomic.frag.glsl");

  res.compileShader(m_basicShaders.renderInstanceBboxesMeshShader, VK_SHADER_STAGE_MESH_BIT_NV,
                    "render_instance_bbox.mesh.glsl", &options);
  res.compileShader(m_basicShaders.renderInstanceBboxesFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, "render_instance_bbox.frag.glsl");

  res.compileShader(m_basicShaders.renderClusterBboxesMeshShader, VK_SHADER_STAGE_MESH_BIT_NV,
                    "render_cluster_bbox.mesh.glsl", &options);
  res.compileShader(m_basicShaders.renderClusterBboxesFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, "render_cluster_bbox.frag.glsl");

  if(!res.verifyShaders(m_basicShaders))
  {
    return false;
  }

  return true;
}

void Renderer::initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  initBasicPipelines(res, rscene, config);

  const Scene& scene = *rscene.scene;

  m_renderInstances.resize(scene.m_instances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    shaderio::RenderInstance&  renderInstance = m_renderInstances[i];
    const Scene::Instance&     sceneInstance  = scene.m_instances[i];
    const Scene::GeometryView& geometry       = scene.getActiveGeometry(sceneInstance.geometryID);

    renderInstance                = {};
    renderInstance.worldMatrix    = glm::mat4x3(sceneInstance.matrix);
    renderInstance.worldMatrixI   = glm::mat4x3(glm::inverse(sceneInstance.matrix));
    renderInstance.geometryID     = sceneInstance.geometryID;
    renderInstance.materialID     = uint16_t(sceneInstance.materialID);
    renderInstance.maxLodLevelRcp = geometry.lodLevelsCount > 1 ? 1.0f / float(geometry.lodLevelsCount - 1) : 0.0f;
    renderInstance.packedColor    = glm::packUnorm4x8(sceneInstance.color);
    renderInstance.twoSided       = sceneInstance.twoSided ? 1 : 0;
    renderInstance.flipWinding =
        (!sceneInstance.twoSided && ((glm::determinant(sceneInstance.matrix) <= 0) != config.flipWinding)) ? 1 : 0;
  }

  res.createBuffer(m_renderInstanceBuffer, sizeof(shaderio::RenderInstance) * m_renderInstances.size(),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NVVK_DBG_NAME(m_renderInstanceBuffer.buffer);
  res.simpleUploadBuffer(m_renderInstanceBuffer, m_renderInstances.data());

  if(config.useSorting)
  {
    VrdxSorterStorageRequirements sorterRequirements = {};
    vrdxGetSorterKeyValueStorageRequirements(res.m_vrdxSorter, uint32_t(m_renderInstances.size()), &sorterRequirements);

    res.createBuffer(m_sortingAuxBuffer, sorterRequirements.size, sorterRequirements.usage);
    NVVK_DBG_NAME(m_sortingAuxBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sortingAuxBuffer.bufferSize, "operations", "traversal sorting");
  }
}

void Renderer::deinitBasics(Resources& res)
{
  res.destroyPipelines(m_basicPipelines);
  vkDestroyPipelineLayout(res.m_device, m_basicPipelineLayout, nullptr);

  m_basicDset.deinit();

  res.m_allocator.destroyBuffer(m_renderInstanceBuffer);
  res.m_allocator.destroyBuffer(m_sortingAuxBuffer);
}

void Renderer::updateBasicDescriptors(Resources& res, RenderScene& rscene, const nvvk::Buffer* sceneBuildBuffer)
{
  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_basicDset.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
  writeSets.append(m_basicDset.makeWrite(BINDINGS_READBACK_SSBO), res.m_commonBuffers.readBack);
  if(res.m_supportsClusterRaytracing)
  {
    writeSets.append(m_basicDset.makeWrite(BINDINGS_RAYTRACING_DEPTH), res.m_frameBuffer.imgRaytracingDepth.descriptor);
  }
  writeSets.append(m_basicDset.makeWrite(BINDINGS_RASTER_ATOMIC), res.m_frameBuffer.imgRasterAtomic.descriptor);
  writeSets.append(m_basicDset.makeWrite(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
  writeSets.append(m_basicDset.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
  if(sceneBuildBuffer)
  {
    writeSets.append(m_basicDset.makeWrite(BINDINGS_SCENEBUILDING_UBO), *sceneBuildBuffer);
  }
  if(rscene.useStreaming)
  {
    writeSets.append(m_basicDset.makeWrite(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
  }
  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);
}


void Renderer::initBasicPipelines(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_basicShaderFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT;

  nvvk::DescriptorBindings bindings;
  bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_basicShaderFlags);
  if(res.m_supportsClusterRaytracing)
  {
    bindings.addBinding(BINDINGS_RAYTRACING_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_basicShaderFlags);
  }
  bindings.addBinding(BINDINGS_RASTER_ATOMIC, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_basicShaderFlags);
  if(rscene.useStreaming)
  {
    bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_basicShaderFlags);
  }
  m_basicDset.init(bindings, res.m_device);

  nvvk::createPipelineLayout(res.m_device, &m_basicPipelineLayout, {m_basicDset.getLayout()},
                             {{m_basicShaderFlags, 0, sizeof(uint32_t)}});

  nvvk::GraphicsPipelineCreator graphicsGen;
  nvvk::GraphicsPipelineState   state                = res.m_basicGraphicsState;
  graphicsGen.pipelineInfo.layout                    = m_basicPipelineLayout;
  graphicsGen.renderingState.depthAttachmentFormat   = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
  graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
  graphicsGen.colorFormats                           = {res.m_frameBuffer.colorFormat};

  state.rasterizationState.lineWidth = float(res.m_frameBuffer.pixelScale * 2.0f);

  graphicsGen.clearShaders();
  graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderInstanceBboxesMeshShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderInstanceBboxesFragmentShader));

  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.renderInstanceBboxes);

  graphicsGen.clearShaders();

  state.rasterizationState.lineWidth = float(res.m_frameBuffer.pixelScale);
  graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderClusterBboxesMeshShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderClusterBboxesFragmentShader));

  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.renderClusterBboxes);

  state.depthStencilState.depthWriteEnable = VK_TRUE;
  state.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
  state.rasterizationState.cullMode        = VK_CULL_MODE_NONE;

  graphicsGen.clearShaders();
  graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenBackgroundFragShader));

  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.background);

  graphicsGen.clearShaders();
  graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                        nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullscreenAtomicRasterFragmentShader));

  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.atomicRaster);

  if(res.m_supportsClusterRaytracing)
  {
    state.colorWriteMasks = {0};

    graphicsGen.clearShaders();
    graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",
                          nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
    graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                          nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenWriteDepthFragShader));

    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.writeDepth);
  }
}

void Renderer::renderInstanceBboxes(VkCommandBuffer cmd)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.renderInstanceBboxes);

  uint32_t numRenderInstances = uint32_t(m_renderInstances.size());
  vkCmdPushConstants(cmd, m_basicPipelineLayout, m_basicShaderFlags, 0, sizeof(uint32_t), &numRenderInstances);

  uint32_t workGroupCount = (numRenderInstances + m_meshShaderBoxes - 1) / m_meshShaderBoxes;
  if(m_config.useEXTmeshShader)
  {
    glm::uvec3 grid = shaderio::fit16bitLaunchGrid(workGroupCount);
    vkCmdDrawMeshTasksEXT(cmd, grid.x, grid.y, grid.z);
  }
  else
  {
    vkCmdDrawMeshTasksNV(cmd, workGroupCount, 0);
  }
}

void Renderer::renderClusterBboxes(VkCommandBuffer cmd, nvvk::Buffer sceneBuildBuffer)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.renderClusterBboxes);

  if(m_config.useEXTmeshShader)
  {
    vkCmdDrawMeshTasksIndirectEXT(cmd, sceneBuildBuffer.buffer,
                                  offsetof(shaderio::SceneBuilding, indirectDrawClusterBoxesEXT), 1, 0);
  }
  else
  {
    vkCmdDrawMeshTasksIndirectNV(cmd, sceneBuildBuffer.buffer,
                                 offsetof(shaderio::SceneBuilding, indirectDrawClusterBoxesNV), 1, 0);
  }
}

void Renderer::writeAtomicRaster(VkCommandBuffer cmd)
{
  uint32_t dummy = 0;
  vkCmdPushConstants(cmd, m_basicPipelineLayout, m_basicShaderFlags, 0, sizeof(uint32_t), &dummy);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.atomicRaster);

  vkCmdDraw(cmd, 3, 1, 0, 0);
}

void Renderer::writeRayTracingDepthBuffer(VkCommandBuffer cmd)
{
  uint32_t dummy = 0;
  vkCmdPushConstants(cmd, m_basicPipelineLayout, m_basicShaderFlags, 0, sizeof(uint32_t), &dummy);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.writeDepth);

  vkCmdDraw(cmd, 3, 1, 0, 0);
}

void Renderer::writeBackgroundSky(VkCommandBuffer cmd)
{
  uint32_t dummy = 0;
  vkCmdPushConstants(cmd, m_basicPipelineLayout, m_basicShaderFlags, 0, sizeof(uint32_t), &dummy);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.background);

  vkCmdDraw(cmd, 3, 1, 0, 0);
}

}  // namespace lodclusters
