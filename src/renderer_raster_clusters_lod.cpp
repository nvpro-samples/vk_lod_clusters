/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <volk.h>
#include <nvutils/alignment.hpp>
#include <fmt/format.h>

#include "renderer.hpp"
#include "../shaders/shaderio.h"

namespace lodclusters {

class RendererRasterClustersLod : public Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) override;
  virtual void deinit(Resources& res) override;

private:
  bool initShaders(Resources& res, RenderScene& scene, const RendererConfig& config);

  struct Shaders
  {
    shaderc::SpvCompilationResult graphicsMesh;
    shaderc::SpvCompilationResult graphicsFragment;

    shaderc::SpvCompilationResult computeTraversalPresort;
    shaderc::SpvCompilationResult computeTraversalInit;
    shaderc::SpvCompilationResult computeTraversalRun;
    shaderc::SpvCompilationResult computeTraversalGroups;

    shaderc::SpvCompilationResult computeBuildSetup;

    shaderc::SpvCompilationResult computeRaster;
  };

  struct Pipelines
  {
    VkPipeline graphicsMesh            = nullptr;
    VkPipeline graphicsBboxes          = nullptr;
    VkPipeline computeTraversalPresort = nullptr;
    VkPipeline computeTraversalInit    = nullptr;
    VkPipeline computeTraversalRun     = nullptr;
    VkPipeline computeTraversalGroups  = nullptr;
    VkPipeline computeBuildSetup       = nullptr;
    VkPipeline computeRaster           = nullptr;
  };

  Shaders            m_shaders;
  Pipelines          m_pipelines;
  VkShaderStageFlags m_stageFlags{};
  VkPipelineLayout   m_pipelineLayout{};

  nvvk::DescriptorPack m_dsetPack;

  nvvk::Buffer m_sceneBuildBuffer;
  nvvk::Buffer m_sceneTraversalBuffer;
  nvvk::Buffer m_sceneDataBuffer;

  shaderio::SceneBuilding m_sceneBuildShaderio;
};

bool RendererRasterClustersLod::initShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  if(config.useComputeRaster && (config.useShading || !config.useSeparateGroups || !config.useCulling))
  {
    LOGW("Compute rasterization requires:  \nshading off == VISUALIZE_VISIBILITY_BUFFER\n  separate groups on\n  culling on\n\n");
    return false;
  }

  if(!initBasicShaders(res, rscene, config))
  {
    return false;
  }

  shaderc::CompileOptions options = res.makeCompilerOptions();

  uint32_t meshletTriangles = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterTriangles);
  uint32_t meshletVertices  = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterVertices);
  LOGI("mesh shader config: %d triangles %d vertices\n", meshletTriangles, meshletVertices);

  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));
  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", meshletVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", meshletTriangles));
  options.AddMacroDefinition("TARGETS_RASTERIZATION", "1");
  options.AddMacroDefinition("USE_STREAMING", rscene.useStreaming ? "1" : "0");
  options.AddMacroDefinition("USE_SORTING", config.useSorting ? "1" : "0");
  options.AddMacroDefinition("USE_CULLING", config.useCulling ? "1" : "0");
  options.AddMacroDefinition("USE_PRIMITIVE_CULLING", config.useCulling && config.usePrimitiveCulling ? "1" : "0");
  options.AddMacroDefinition("USE_TWO_PASS_CULLING", config.useCulling && config.useTwoPassCulling ? "1" : "0");
  options.AddMacroDefinition("USE_RENDER_STATS", config.useRenderStats ? "1" : "0");
  options.AddMacroDefinition("USE_SEPARATE_GROUPS", config.useSeparateGroups ? "1" : "0");
  options.AddMacroDefinition("USE_DLSS", "0");
  options.AddMacroDefinition("USE_BLAS_SHARING", "0");
  options.AddMacroDefinition("USE_BLAS_MERGING", "0");
  options.AddMacroDefinition("USE_BLAS_CACHING", "0");
  options.AddMacroDefinition("USE_EXT_MESH_SHADER", fmt::format("{}", config.useEXTmeshShader ? 1 : 0));
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", fmt::format("{}", m_meshShaderWorkgroupSize));
  options.AddMacroDefinition("MESHSHADER_BBOX_COUNT", fmt::format("{}", m_meshShaderBoxes));
  options.AddMacroDefinition("ALLOW_VERTEX_NORMALS", rscene.scene->m_hasVertexNormals && res.m_supportsBarycentrics ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TANGENTS", rscene.scene->m_hasVertexTangents && res.m_supportsBarycentrics ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TEXCOORDS", rscene.scene->m_hasVertexTexCoord0 ? "1" : "0");
  //options.AddMacroDefinition("ALLOW_VERTEX_TEXCOORD_1", rscene.scene->m_hasVertexTexCoord1 ? "1" : "0");
  options.AddMacroDefinition("ALLOW_SHADING", config.useShading && !config.useComputeRaster ? "1" : "0");
  options.AddMacroDefinition("USE_DEPTH_ONLY", !config.useShading && config.useDepthOnly ? "1" : "0");
  options.AddMacroDefinition("DEBUG_VISUALIZATION", config.useDebugVisualization && res.m_supportsBarycentrics ? "1" : "0");
  options.AddMacroDefinition("USE_SW_RASTER", config.useComputeRaster ? "1" : "0");
  options.AddMacroDefinition("USE_TWO_SIDED", rscene.scene->m_hasTwoSided && !config.forceTwoSided ? "1" : "0");
  options.AddMacroDefinition("USE_FORCED_TWO_SIDED", config.forceTwoSided ? "1" : "0");
  options.AddMacroDefinition("USE_FORCED_INVISIBLE_CULLING", "0");

  res.compileShader(m_shaders.graphicsMesh, VK_SHADER_STAGE_MESH_BIT_NV, "render_raster_clusters.mesh.glsl", &options);
  res.compileShader(m_shaders.graphicsFragment, VK_SHADER_STAGE_FRAGMENT_BIT, "render_raster.frag.glsl", &options);
  res.compileShader(m_shaders.computeTraversalPresort, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_presort.comp.glsl", &options);
  res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", &options);
  res.compileShader(m_shaders.computeTraversalRun, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", &options);
  res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options);

  if(config.useComputeRaster)
  {
    res.compileShader(m_shaders.computeRaster, VK_SHADER_STAGE_COMPUTE_BIT, "render_raster_clusters_sw.comp.glsl", &options);
  }

  if(config.useSeparateGroups)
  {
    res.compileShader(m_shaders.computeTraversalGroups, VK_SHADER_STAGE_COMPUTE_BIT,
                      "traversal_run_separate_groups.comp.glsl", &options);
  }

  return res.verifyShaders(m_shaders);
}

bool RendererRasterClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_resourceReservedUsage = {};
  m_config                = config;
  m_maxRenderClusters     = 1u << config.numRenderClusterBits;
  m_maxTraversalTasks     = 1u << config.numTraversalTaskBits;

  if(!initShaders(res, rscene, config))
  {
    return false;
  }

  if(!rscene.updateClasRequired(false))
  {
    return false;
  }

#if USE_DLSS
  // not supported in raster for now
  res.setFramebufferDlss(false, config.dlssQuality);
#endif

  initBasics(res, rscene, config);

  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.operationsMemBytes = logMemoryUsage(rscene.getOperationsSize(), "operations", "rscene total");

  {
    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    NVVK_DBG_NAME(m_sceneBuildBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneBuildBuffer.bufferSize, "operations", "build shaderio");

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.maxRenderClusters  = uint32_t(1u << config.numRenderClusterBits);
    m_sceneBuildShaderio.maxTraversalInfos  = uint32_t(1u << config.numTraversalTaskBits);

    m_sceneBuildShaderio.indirectDispatchGroups.gridY  = 1;
    m_sceneBuildShaderio.indirectDispatchGroups.gridZ  = 1;
    m_sceneBuildShaderio.indirectDrawClustersEXT.gridZ = 1;
    m_sceneBuildShaderio.indirectDrawClustersNV.first  = 0;
    m_sceneBuildShaderio.indirectDrawClustersSW.gridY  = 1;
    m_sceneBuildShaderio.indirectDrawClustersSW.gridZ  = 1;

    BufferRanges mem = {};
    m_sceneBuildShaderio.renderClusterInfos =
        mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);

    if(config.useComputeRaster)
    {
      m_sceneBuildShaderio.renderClusterInfosSW =
          mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);
    }

    if(config.useSorting)
    {
      m_sceneBuildShaderio.instanceSortKeys   = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.instanceSortValues = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    }

    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, 8);
    }

    if(config.useTwoPassCulling && config.useCulling)
    {
      m_sceneBuildShaderio.instanceVisibility = mem.append(sizeof(uint8_t) * m_renderInstances.size(), 4);
    }

    res.createBuffer(m_sceneDataBuffer, mem.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(m_sceneDataBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneDataBuffer.bufferSize, "operations", "build data");

    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortKeys += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortValues += m_sceneDataBuffer.address;
    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos += m_sceneDataBuffer.address;
    }
    if(config.useComputeRaster)
    {
      m_sceneBuildShaderio.renderClusterInfosSW += m_sceneDataBuffer.address;
    }
    if(config.useTwoPassCulling && config.useCulling)
    {
      m_sceneBuildShaderio.instanceVisibility += m_sceneDataBuffer.address;
    }

    res.createBuffer(m_sceneTraversalBuffer, sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(m_sceneTraversalBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneTraversalBuffer.bufferSize, "operations", "build traversal");

    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;
  }

  updateBasicDescriptors(res, rscene, &m_sceneBuildBuffer);

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.updateBindings(m_sceneBuildBuffer);
  }

  {
    m_stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        config.useCulling && config.useTwoPassCulling ? 2 : 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RASTER_ATOMIC, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags);
    if(rscene.useStreaming)
    {
      bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
      bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    }
    m_dsetPack.init(bindings, res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()}, {{m_stageFlags, 0, sizeof(uint32_t)}});

    nvvk::WriteSetContainer writeSets;

    writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), &res.m_commonBuffers.readBack);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 0), &res.m_hizUpdate[0].farImageInfo);
    if(config.useCulling && config.useTwoPassCulling)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 1), &res.m_hizUpdate[1].farImageInfo);
    }
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RASTER_ATOMIC), &res.m_frameBuffer.imgRasterAtomic);
    if(rscene.useStreaming)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_SSBO), rscene.sceneStreaming.getShaderStreamingBuffer());
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
    }
    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  {
    nvvk::GraphicsPipelineCreator graphicsGen;
    nvvk::GraphicsPipelineState   state              = res.m_basicGraphicsState;
    graphicsGen.pipelineInfo.layout                  = m_pipelineLayout;
    graphicsGen.renderingState.depthAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
    graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
    graphicsGen.colorFormats = {res.m_frameBuffer.colorFormat};

    state.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    if(config.forceTwoSided)
    {
      state.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    }
    if(config.useComputeRaster && false)
    {
      state.depthStencilState.depthWriteEnable = VK_FALSE;
      state.depthStencilState.depthTestEnable  = VK_FALSE;
      state.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
    }
    graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsMesh));
    if(!m_config.useDepthOnly)
    {
      graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsFragment));
    }
    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_pipelines.graphicsMesh);
  }

  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;

    if(config.useSorting)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalPresort);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalPresort);
    }

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalInit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalRun);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);

    if(config.useComputeRaster)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeRaster);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeRaster);
    }

    if(config.useSeparateGroups)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalGroups);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalGroups);
    }
  }

  return true;
}

static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void RendererRasterClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  {
    glm::vec2 renderScale = res.getFramebufferWindow2RenderScale();
    float     pixelScale  = std::min(renderScale.x, renderScale.y);

    m_sceneBuildShaderio.errorOverDistanceThreshold =
        clusterLodErrorOverDistance(frame.lodPixelError * pixelScale, frame.frameConstants.fov,
                                    frame.frameConstants.viewportf.y);
  }

  m_sceneBuildShaderio.traversalViewMatrix    = frame.traversalViewMatrix;
  m_sceneBuildShaderio.cullViewProjMatrix     = frame.cullViewProjMatrix;
  m_sceneBuildShaderio.cullViewProjMatrixLast = frame.cullViewProjMatrixLast;

  m_sceneBuildShaderio.frameIndex        = m_frameIndex;
  m_sceneBuildShaderio.swRasterThreshold = frame.swRasterThreshold;

  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants),
                    (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0);

  if(m_config.useComputeRaster)
  {
    VkClearColorValue clearValue;
    clearValue.uint32[0] = 0u;
    clearValue.uint32[1] = 0u;
    clearValue.uint32[2] = 0u;
    clearValue.uint32[3] = 0u;

    VkImageSubresourceRange subResource = {};
    subResource.aspectMask              = VK_IMAGE_ASPECT_COLOR_BIT;
    subResource.levelCount              = 1;
    subResource.layerCount              = 1;

    vkCmdClearColorImage(cmd, res.m_frameBuffer.imgRasterAtomic.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subResource);
  }

  if(rscene.useStreaming)
  {
    SceneStreaming::FrameSettings settings;
    settings.ageThreshold = frame.streamingAgeThreshold;

    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer, settings, profiler);
  }

  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, 0, profiler);

    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);
  }

  const uint32_t passCount = m_config.useCulling && m_config.useTwoPassCulling ? 2 : 1;
  const uint32_t lastPass  = passCount - 1;
  for(uint32_t pass = 0; pass < passCount; pass++)
  {
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Preparation");
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

      if(pass == 0 && m_config.useSorting)
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

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);
      }

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP));

      // this barrier covers init & streaming pre traversal
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Run");

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP));

      if(m_config.useSeparateGroups)
      {
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                             &memBarrier, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
      }

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      uint32_t buildSetupID = BUILD_SETUP_DRAW;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT
                                 | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                               | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(pass == lastPass && rscene.useStreaming)
    {
      rscene.sceneStreaming.cmdPostTraversal(cmd, 0, true, profiler);
    }
    else if(rscene.useStreaming)
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Stream Post Traversal");
    }

    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Draw");

      if(m_config.useComputeRaster)
      {
        auto timerSection = profiler.cmdFrameSection(cmd, "SW-Raster");

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeRaster);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClustersSW));
      }

      VkAttachmentLoadOp op = pass == 1 ? VK_ATTACHMENT_LOAD_OP_LOAD :
                                          (m_config.useShading ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR);

      res.cmdBeginRendering(cmd, false, op, op);

      if(pass == 0 && m_config.useShading)
      {
        writeBackgroundSky(cmd);
      }

      {
        auto timerSection = profiler.cmdFrameSection(cmd, "HW-Raster");

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMesh);

        if(m_config.useEXTmeshShader)
        {
          vkCmdDrawMeshTasksIndirectEXT(cmd, m_sceneBuildBuffer.buffer,
                                        offsetof(shaderio::SceneBuilding, indirectDrawClustersEXT), 1, 0);
        }
        else
        {
          vkCmdDrawMeshTasksIndirectNV(cmd, m_sceneBuildBuffer.buffer,
                                       offsetof(shaderio::SceneBuilding, indirectDrawClustersNV), 1, 0);
        }
      }

      if(m_config.useComputeRaster)
      {
        VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, 0, 0, nullptr);

        writeAtomicRaster(cmd);
      }

      if(frame.showClusterBboxes)
      {
        renderClusterBboxes(cmd, m_sceneBuildBuffer);
      }

      if(pass == lastPass && frame.showInstanceBboxes)
      {
        renderInstanceBboxes(cmd);
      }

      vkCmdEndRendering(cmd);
    }

    if(!frame.freezeCulling)
    {
      if(m_config.useTwoPassCulling)
      {
        res.cmdBuildHiz(cmd, frame, profiler, pass ^ 1);
      }
      else
      {
        res.cmdBuildHiz(cmd, frame, profiler, 0);
      }
    }
  }
  if(passCount > 1)
  {
    profiler.getProfilerTimeline()->frameAccumulationSplit();
  }

  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }

  // reservation for geometry may change
  m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);
  m_resourceActualUsage                    = m_resourceReservedUsage;
  m_resourceActualUsage.geometryMemBytes   = rscene.getGeometrySize(false);

  m_frameIndex++;
}

void RendererRasterClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  vkDeviceWaitIdle(res.m_device);

  VkWriteDescriptorSet writes[3];

  writes[0]            = m_dsetPack.makeWrite(BINDINGS_RASTER_ATOMIC);
  writes[0].pImageInfo = &res.m_frameBuffer.imgRasterAtomic.descriptor;
  writes[1]            = m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 0);
  writes[1].pImageInfo = &res.m_hizUpdate[0].farImageInfo;

  if(m_config.useCulling && m_config.useTwoPassCulling)
  {
    writes[2]            = m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 1);
    writes[2].pImageInfo = &res.m_hizUpdate[1].farImageInfo;
  }

  vkUpdateDescriptorSets(res.m_device, m_config.useCulling && m_config.useTwoPassCulling ? 3 : 2, writes, 0, nullptr);

  Renderer::updatedFrameBuffer(res, rscene);
}

void RendererRasterClustersLod::deinit(Resources& res)
{
  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);

  m_dsetPack.deinit();

  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);

  deinitBasics(res);
}

std::unique_ptr<Renderer> makeRendererRasterClustersLod()
{
  return std::make_unique<RendererRasterClustersLod>();
}
}  // namespace lodclusters
