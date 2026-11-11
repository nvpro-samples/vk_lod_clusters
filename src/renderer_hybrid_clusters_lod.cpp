/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <volk.h>

#include "renderer_raytrace_clusters_lod.hpp"

// Rasterizes primary visibility into a visibility buffer, then path traces from there.
// Traversal, acceleration structures and the trace are the ray tracing renderer's.
// See docs/raster_hybrid.md

namespace lodclusters {

class RendererHybridClustersLod : public RendererRayTraceClustersLod
{
public:
  RendererHybridClustersLod() { m_hybrid = true; }

  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) override;

protected:
  virtual bool initRasterShaders(Resources& res, RenderScene& rscene, const shaderc::CompileOptions& options) override;
  virtual void initRasterPipelines(Resources& res) override;
  virtual void renderRaster(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;

private:
  struct RasterShaders
  {
    Shader mesh;
    Shader fragment;
    Shader cull;
    Shader computeRaster;
    Shader atomicMerge;
  };

  struct RasterPipelines
  {
    VkPipeline raster        = nullptr;
    VkPipeline cull          = nullptr;
    VkPipeline computeRaster = nullptr;
    VkPipeline atomicMerge   = nullptr;
  };

  RasterShaders   m_rasterShaders;
  RasterPipelines m_rasterPipelines;
  // the merge resets what it reads, a new image needs one clear
  bool m_rasterAtomicCleared = false;

  void cmdComputeRaster(VkCommandBuffer cmd, Resources& res, RenderScene& rscene);
};

bool RendererHybridClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  // shading is the path tracer's ray generation
  RendererConfig hybridConfig = config;
  hybridConfig.usePathtrace   = true;
  // the visibility buffer stores the cluster index in 24 bits, streaming clamps its resident ones
  const uint32_t maxClusterIndices = 1u << (32 - HYBRIDPATH_VISBUFFER_TRIANGLE_BITS);
  if(!rscene.useStreaming && rscene.scene->m_maxPerGeometryClusters > maxClusterIndices)
  {
    LOGW("hybrid: a geometry has %u clusters, the visibility buffer addresses %u, larger indices alias\n",
         rscene.scene->m_maxPerGeometryClusters, maxClusterIndices);
  }
  m_rasterAtomicCleared = false;
  return RendererRayTraceClustersLod::init(res, rscene, hybridConfig);
}

void RendererHybridClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  RendererRayTraceClustersLod::updatedFrameBuffer(res, rscene);
  m_rasterAtomicCleared = false;
}

void RendererHybridClustersLod::deinit(Resources& res)
{
  res.destroyPipelines(m_rasterPipelines);
  RendererRayTraceClustersLod::deinit(res);
}

bool RendererHybridClustersLod::initRasterShaders(Resources& res, RenderScene& rscene, const shaderc::CompileOptions& options)
{
  // the raster pass only writes ids, all shading is in the ray generation
  shaderc::CompileOptions rasterOptions = options;
  rasterOptions.AddMacroDefinition("ALLOW_SHADING", "0");
  rasterOptions.AddMacroDefinition("USE_DEPTH_ONLY", "0");
  rasterOptions.AddMacroDefinition("ALLOW_VERTEX_TEXCOORDS",
                                   rscene.scene->m_hasVertexTexCoord0 && res.m_supportsBarycentrics ? "1" : "0");
  rasterOptions.AddMacroDefinition("ALLOW_VERTEX_TEXCOORD_0",
                                   rscene.scene->m_hasVertexTexCoord0 && res.m_supportsBarycentrics ? "1" : "0");
  rasterOptions.AddMacroDefinition("USE_PRIMITIVE_CULLING", m_config.useCulling && m_config.usePrimitiveCulling ? "1" : "0");

  const char* meshFile = m_config.useEXTmeshShader ? "render_raster_clusters.mesh.glsl" : "render_raster_clusters_nv.mesh.glsl";
  res.compileShader(m_rasterShaders.mesh, VK_SHADER_STAGE_MESH_BIT_NV, meshFile, &rasterOptions, "hybridpath");
  res.compileShader(m_rasterShaders.fragment, VK_SHADER_STAGE_FRAGMENT_BIT, "render_hybridpath_raster.frag.glsl", &rasterOptions);

  // tests the first pass' hiz, `texHizFar[1]`
  shaderc::CompileOptions cullOptions = options;
  cullOptions.AddMacroDefinition("USE_TWO_PASS_CULLING", "1");
  res.compileShader(m_rasterShaders.cull, VK_SHADER_STAGE_COMPUTE_BIT, "render_hybridpath_cull.comp.glsl", &cullOptions);

  if(useHybridSWRaster())
  {
    // alpha tested clusters stay on the hardware path, ids only
    shaderc::CompileOptions swOptions = rasterOptions;
    swOptions.AddMacroDefinition("HAS_ALPHA_TEST", "0");
    swOptions.AddMacroDefinition("USE_DLSS_GUIDE_BUFFERS", "0");
    res.compileShader(m_rasterShaders.computeRaster, VK_SHADER_STAGE_COMPUTE_BIT, "render_raster_clusters_sw.comp.glsl",
                      &swOptions, "hybridpath");
    res.compileShader(m_rasterShaders.atomicMerge, VK_SHADER_STAGE_FRAGMENT_BIT,
                      "render_hybridpath_raster_atomic.frag.glsl", &rasterOptions);
  }

  return res.verifyShaders(m_rasterShaders);
}

void RendererHybridClustersLod::initRasterPipelines(Resources& res)
{
  VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
  compInfo.stage.pName                 = "main";
  compInfo.layout                      = m_pipelineLayout;
  res.createComputePipeline(compInfo, m_rasterShaders.cull, m_rasterPipelines.cull);
  if(useHybridSWRaster())
  {
    res.createComputePipeline(compInfo, m_rasterShaders.computeRaster, m_rasterPipelines.computeRaster);
  }

  nvvk::GraphicsPipelineCreator graphicsGen;
  nvvk::GraphicsPipelineState   state = res.m_basicGraphicsState;
  graphicsGen.pipelineInfo.layout     = m_pipelineLayout;
  graphicsGen.pipelineInfo.flags      = res.getPipelineCreateFlags();

  graphicsGen.renderingState.depthAttachmentFormat   = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
  graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
  graphicsGen.colorFormats                           = {res.m_frameBuffer.visibilityBufferFormat};

  state.colorBlendEnables[0]         = VK_FALSE;
  state.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  if(m_config.forceTwoSided)
  {
    state.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  }

  graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main", nvvkglsl::GlslCompiler::getSpirvData(m_rasterShaders.mesh));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_rasterShaders.fragment));
  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_rasterPipelines.raster);

  if(useHybridSWRaster())
  {
    // depth tested against the hardware raster of the same pass
    state.depthStencilState.depthWriteEnable = VK_TRUE;
    state.rasterizationState.cullMode        = VK_CULL_MODE_NONE;

    graphicsGen.clearShaders();
    graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",
                          nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
    graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_rasterShaders.atomicMerge));
    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_rasterPipelines.atomicMerge);
  }
}

void RendererHybridClustersLod::cmdComputeRaster(VkCommandBuffer cmd, Resources& res, RenderScene& rscene)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, 1,
                          rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rasterPipelines.computeRaster);
  vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchClustersSW));

  // the merge reads and resets the atomics
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
}

void RendererHybridClustersLod::renderRaster(VkCommandBuffer         cmd,
                                             Resources&              res,
                                             RenderScene&            rscene,
                                             const FrameConfig&      frame,
                                             nvvk::ProfilerGpuTimer& profiler)
{
  auto timerSection = profiler.cmdFrameSection(cmd, "Draw");

  const bool useSW = useHybridSWRaster();

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  if(useSW && !m_rasterAtomicCleared)
  {
    VkClearColorValue       clearValue  = {};
    VkImageSubresourceRange subResource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, res.m_frameBuffer.imgRasterAtomic.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subResource);
    m_rasterAtomicCleared = true;

    memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);
  }

  // traversal wrote the cluster lists and draw arguments
  const VkPipelineStageFlags rasterStages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV
                                            | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, rasterStages, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  res.cmdImageTransition(cmd, res.m_frameBuffer.imgVisibilityBuffer, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, true);

  VkClearValue colorClear{.color = {.uint32 = {HYBRIDPATH_VISBUFFER_EMPTY, HYBRIDPATH_VISBUFFER_EMPTY, 0, 0}}};
  VkClearValue depthClear{.depthStencil = {0.0F, 0}};

  // the first pass draws the instances visible in last frame's hiz, the second those it
  // occluded that pass the first pass' hiz. Without culling nothing is occluded.
  const uint32_t passCount = m_config.useCulling ? 2 : 1;
  for(uint32_t pass = 0; pass < passCount; pass++)
  {
    if(pass == 1)
    {
      // pass 1 loads the ids pass 0 wrote and adds to them
      memBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

      cmdBuildPassHiz(cmd, res, frame, profiler, true, 0);

      // the first pass is done reading the list, the cull refills it
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rasterPipelines.cull);
      vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchHybridReject));

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      uint32_t buildSetupID = BUILD_SETUP_HYBRID_DRAW;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);

      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, rasterStages, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }

    if(useSW)
    {
      nvvk::ProfilerGpuTimer::FrameSection timerSection = profiler.cmdFrameSection(cmd, fmt::format("SW-Raster {}", pass));
      cmdComputeRaster(cmd, res, rscene);
    }

    nvvk::ProfilerGpuTimer::FrameSection timerSectionHW = profiler.cmdFrameSection(cmd, fmt::format("HW-Raster {}", pass));

    res.cmdImageTransition(cmd, res.m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    VkAttachmentLoadOp loadOp = pass == 0 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;

    VkRenderingAttachmentInfo colorAttachment = {
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = res.m_frameBuffer.imgVisibilityBuffer.descriptor.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp      = loadOp,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = colorClear,
    };
    VkRenderingAttachmentInfo depthStencilAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = res.m_frameBuffer.imgDepthStencil.descriptor.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp      = loadOp,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = depthClear,
    };
    VkRenderingInfo renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = res.m_frameBuffer.scissor,
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthStencilAttachment,
    };

    vkCmdBeginRendering(cmd, &renderingInfo);
    vkCmdSetViewportWithCount(cmd, 1, &res.m_frameBuffer.viewport);
    vkCmdSetScissorWithCount(cmd, 1, &res.m_frameBuffer.scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 1, 1,
                            rscene.sceneTextures.getDsetPack().getSetPtr(), 0, nullptr);
    if(useSW)
    {
      // first, so the hardware raster's early depth test sees it
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipelines.atomicMerge);
      vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipelines.raster);

    cmdDrawMeshTasksIndirect(cmd, m_config.useEXTmeshShader, m_sceneBuildBuffer.buffer,
                             offsetof(shaderio::SceneBuilding, indirectDrawHybridEXT),
                             offsetof(shaderio::SceneBuilding, indirectDrawHybridNV));

    vkCmdEndRendering(cmd);
  }

  // the ray generation, or the ray query compute shader, reads the ids
  memBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
}

std::unique_ptr<Renderer> makeRendererHybridClustersLod()
{
  return std::make_unique<RendererHybridClustersLod>();
}

}  // namespace lodclusters
