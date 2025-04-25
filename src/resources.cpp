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

#include <glm/glm.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/renderpasses_vk.hpp>

#include "resources.hpp"
#include "shaders/shaderio.h"

namespace lodclusters {

void Resources::beginFrame(uint32_t cycleIndex)
{
  m_cycleIndex = cycleIndex;
}

void Resources::postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  auto sec = profiler.beginSection("Post-process", cmd);

  bool doHbao = frame.hbaoActive;

  // do hbao on the full-res input image
  if(frame.hbaoActive && (m_hbaoFullRes || !m_framebuffer.useResolved))
  {
    cmdHBAO(cmd, frame, profiler);

    doHbao = false;
  }

  if(m_framebuffer.useResolved)
  {
    // blit to resolved
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = frame.winWidth;
    region.dstOffsets[1].y           = frame.winHeight;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = m_framebuffer.renderWidth;
    region.srcOffsets[1].y           = m_framebuffer.renderHeight;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;

    cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkCmdBlitImage(cmd, m_framebuffer.imgColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_framebuffer.imgColorResolved.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

    if(doHbao)
    {
      cmdHBAO(cmd, frame, profiler);
    }

    cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  else
  {
    cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  {
    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, m_supportedSaderStageFlags, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    VkBufferCopy region;
    region.size      = sizeof(shaderio::Readback);
    region.srcOffset = 0;
    region.dstOffset = m_cycleIndex * sizeof(shaderio::Readback);
    vkCmdCopyBuffer(cmd, m_common.readbackDevice.buffer, m_common.readbackHost.buffer, 1, &region);
  }

  profiler.endSection(sec, cmd);
}

void Resources::endFrame() {}

void Resources::emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  auto sec = profiler.beginSection("Render", cmd);
  cmdBeginRendering(cmd);
  vkCmdEndRendering(cmd);
  profiler.endSection(sec, cmd);
}

void Resources::getReadbackData(shaderio::Readback& readback)
{
  const shaderio::Readback* pReadback = (const shaderio::Readback*)m_allocator.map(m_common.readbackHost);
  readback                            = pReadback[m_cycleIndex];
  if(readback._packedDepth0 == 0)
  {
    readback.clusterTriangleId = ~0u;
    readback.instanceId        = ~0u;
  }
  m_allocator.unmap(m_common.readbackHost);
}

bool Resources::init(nvvk::Context* context, const std::vector<std::string>& shaderSearchPaths)
{
  m_fboChangeID = 0;
  m_context     = context;

  m_physical = m_context->m_physicalDevice;
  m_device   = m_context->m_device;

  {
    m_queueStates.primary.init(m_device, context->m_queueGCT.queue, context->m_queueGCT.familyIndex, 0);
    m_queueStates.transfer.init(m_device, context->m_queueT.queue, context->m_queueT.familyIndex, 0);
  }

  {

    m_queue       = context->m_queueGCT;
    m_queueFamily = context->m_queueGCT.familyIndex;
  }

  nvvk::DebugUtil debugUtil(m_device);

  m_supportedSaderStageFlags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
                               | VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV | VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV
                               | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

  m_tempCommandPool.init(m_device, m_queueFamily, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_queue);

  m_memAllocator.init(m_device, m_physical);
  m_memAllocator.setAllocateFlags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, true);
  m_memAllocator.setAllowDowngrade(false);  // prefer quicker crashes
  m_allocator.init(m_device, m_physical, &m_memAllocator);

  {
    // common
    m_common.view = createBuffer(sizeof(shaderio::FrameConstants) * 2, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    DEBUGUTIL_SET_NAME(m_common.view.buffer);

    m_common.readbackDevice =
        createBuffer(sizeof(shaderio::Readback), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    DEBUGUTIL_SET_NAME(m_common.readbackDevice.buffer);

    m_common.readbackHost = createBuffer(sizeof(shaderio::Readback) * nvvk::DEFAULT_RING_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    DEBUGUTIL_SET_NAME(m_common.readbackHost.buffer);
  }

  m_shaderManager.init(m_device, 1, 2);
  m_shaderManager.m_filetype        = nvh::ShaderFileManager::FILETYPE_GLSL;
  m_shaderManager.m_keepModuleSPIRV = true;
  for(const auto& it : shaderSearchPaths)
  {
    m_shaderManager.addDirectory(it);
  }

  {
    m_basicGraphicsState = nvvk::GraphicsPipelineState();

    m_basicGraphicsState.inputAssemblyState.topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    m_basicGraphicsState.rasterizationState.cullMode  = (VK_CULL_MODE_BACK_BIT);
    m_basicGraphicsState.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    m_basicGraphicsState.rasterizationState.lineWidth = float(m_framebuffer.supersample);

    m_basicGraphicsState.depthStencilState.depthTestEnable       = VK_TRUE;
    m_basicGraphicsState.depthStencilState.depthWriteEnable      = VK_TRUE;
    m_basicGraphicsState.depthStencilState.depthCompareOp        = VK_COMPARE_OP_LESS;
    m_basicGraphicsState.depthStencilState.depthBoundsTestEnable = VK_FALSE;
    m_basicGraphicsState.depthStencilState.stencilTestEnable     = VK_FALSE;
    m_basicGraphicsState.depthStencilState.minDepthBounds        = 0.0f;
    m_basicGraphicsState.depthStencilState.maxDepthBounds        = 1.0f;

    m_basicGraphicsState.multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  }

  {
    NVHizVK::Config config;
    config.msaaSamples             = 0;
    config.reversedZ               = false;
    config.supportsMinmaxFilter    = true;
    config.supportsSubGroupShuffle = true;
    m_hiz.init(m_device, config, 1);

    VkShaderModule shaderModules[NVHizVK::SHADER_COUNT];
    for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
    {
      m_hizShaders[i] = m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "nvhiz-update.comp.glsl",
                                                           m_hiz.getShaderDefines(i));

      assert(m_shaderManager.isValid(m_hizShaders[i]));

      shaderModules[i] = m_shaderManager.get(m_hizShaders[i]);
    }
    m_hiz.initPipelines(shaderModules);
  }

  {
    HbaoPass::Config config;
    config.maxFrames    = 1;
    config.targetFormat = m_framebuffer.colorFormat;

    m_hbaoPass.init(m_device, &m_allocator, &m_shaderManager, config);
  }

  m_sky.setup(m_device, &m_allocator);

  {
    VrdxSorterCreateInfo sorterCreateInfo;
    sorterCreateInfo.device         = m_device;
    sorterCreateInfo.physicalDevice = m_physical;
    sorterCreateInfo.pipelineCache  = nullptr;

    vrdxCreateSorter(&sorterCreateInfo, &m_vrdxSorter);
  }


  return true;
}

void Resources::deinit()
{
  synchronize("sync deinit");

  {
    destroy(m_common.view);
    destroy(m_common.readbackDevice);
    destroy(m_common.readbackHost);
  }

  for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
  {
    m_shaderManager.destroyShaderModule(m_hizShaders[i]);
  }

  deinitFramebuffer();
  m_hbaoPass.deinit();

  m_hiz.deinit();

  vrdxDestroySorter(m_vrdxSorter);

  m_sky.destroy();
  m_tempCommandPool.deinit();
  m_allocator.deinit();
  m_memAllocator.deinit();
  m_shaderManager.deinit();

  m_queueStates.primary.deinit();
  m_queueStates.transfer.deinit();
}


bool Resources::initFramebuffer(int winWidth, int winHeight, int supersample, bool hbaoFullRes)
{
  VkResult result;

  m_hbaoFullRes = hbaoFullRes;
  m_fboChangeID++;

  if(m_framebuffer.imgColor.image != 0)
  {
    deinitFramebuffer();
  }

  m_basicGraphicsState.rasterizationState.lineWidth = float(supersample);

  nvvk::DebugUtil debugUtil(m_device);

  bool oldResolved = m_framebuffer.supersample > 1;

  m_framebuffer.renderWidth  = winWidth * supersample;
  m_framebuffer.renderHeight = winHeight * supersample;
  m_framebuffer.supersample  = supersample;

  LOGI("framebuffer: %d x %d\n", m_framebuffer.renderWidth, m_framebuffer.renderHeight);

  m_framebuffer.useResolved = supersample > 1;

  uint32_t atomicLayers = 1;

  VkSampleCountFlagBits samplesUsed = VK_SAMPLE_COUNT_1_BIT;
  {
    // color
    VkImageCreateInfo cbImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    cbImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    cbImageInfo.format            = m_framebuffer.colorFormat;
    cbImageInfo.extent.width      = m_framebuffer.renderWidth;
    cbImageInfo.extent.height     = m_framebuffer.renderHeight;
    cbImageInfo.extent.depth      = 1;
    cbImageInfo.mipLevels         = 1;
    cbImageInfo.arrayLayers       = 1;
    cbImageInfo.samples           = samplesUsed;
    cbImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    cbImageInfo.flags             = 0;
    cbImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    cbImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    m_framebuffer.imgColor = m_allocator.createImage(cbImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgColor.image);
  }

  // depth stencil
  m_framebuffer.depthStencilFormat = nvvk::findDepthStencilFormat(m_physical);

  {
    VkImageCreateInfo dsImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    dsImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    dsImageInfo.format            = m_framebuffer.depthStencilFormat;
    dsImageInfo.extent.width      = m_framebuffer.renderWidth;
    dsImageInfo.extent.height     = m_framebuffer.renderHeight;
    dsImageInfo.extent.depth      = 1;
    dsImageInfo.mipLevels         = 1;
    dsImageInfo.arrayLayers       = 1;
    dsImageInfo.samples           = samplesUsed;
    dsImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    dsImageInfo.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    dsImageInfo.flags             = 0;
    dsImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

    m_framebuffer.imgDepthStencil = m_allocator.createImage(dsImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgDepthStencil.image);
  }

  if(m_framebuffer.useResolved)
  {
    // resolve image
    VkImageCreateInfo resImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    resImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    resImageInfo.format            = m_framebuffer.colorFormat;
    resImageInfo.extent.width      = winWidth;
    resImageInfo.extent.height     = winHeight;
    resImageInfo.extent.depth      = 1;
    resImageInfo.mipLevels         = 1;
    resImageInfo.arrayLayers       = 1;
    resImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    resImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    resImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                         | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    resImageInfo.flags         = 0;
    resImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    m_framebuffer.imgColorResolved = m_allocator.createImage(resImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgColorResolved.image);
  }

  {
    VkImageCreateInfo dsImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    dsImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    dsImageInfo.format            = m_framebuffer.raytracingDepthFormat;
    dsImageInfo.extent.width      = m_framebuffer.renderWidth;
    dsImageInfo.extent.height     = m_framebuffer.renderHeight;
    dsImageInfo.extent.depth      = 1;
    dsImageInfo.mipLevels         = 1;
    dsImageInfo.arrayLayers       = 1;
    dsImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    dsImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    dsImageInfo.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    dsImageInfo.flags             = 0;
    dsImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;


    m_framebuffer.imgRaytracingDepth = m_allocator.createImage(dsImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgRaytracingDepth.image);
  }

  {
    m_hiz.setupUpdateInfos(m_hizUpdate, m_framebuffer.renderWidth, m_framebuffer.renderHeight,
                           m_framebuffer.depthStencilFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

    // hiz
    VkImageCreateInfo hizImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    hizImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    hizImageInfo.format            = m_hizUpdate.farInfo.format;
    hizImageInfo.extent.width      = m_hizUpdate.farInfo.width;
    hizImageInfo.extent.height     = m_hizUpdate.farInfo.height;
    hizImageInfo.mipLevels         = m_hizUpdate.farInfo.mipLevels;
    hizImageInfo.extent.depth      = 1;
    hizImageInfo.arrayLayers       = 1;
    hizImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    hizImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    hizImageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    hizImageInfo.flags             = 0;
    hizImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;


    m_framebuffer.imgHizFar = m_allocator.createImage(hizImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgHizFar.image);

    m_hizUpdate.sourceImage = m_framebuffer.imgDepthStencil.image;
    m_hizUpdate.farImage    = m_framebuffer.imgHizFar.image;
    m_hizUpdate.nearImage   = VK_NULL_HANDLE;
  }

  // views after allocation handling
  {
    VkImageViewCreateInfo cbImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    cbImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    cbImageViewInfo.format                          = m_framebuffer.colorFormat;
    cbImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    cbImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    cbImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    cbImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    cbImageViewInfo.flags                           = 0;
    cbImageViewInfo.subresourceRange.levelCount     = 1;
    cbImageViewInfo.subresourceRange.baseMipLevel   = 0;
    cbImageViewInfo.subresourceRange.layerCount     = 1;
    cbImageViewInfo.subresourceRange.baseArrayLayer = 0;
    cbImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    cbImageViewInfo.image = m_framebuffer.imgColor.image;
    result                = vkCreateImageView(m_device, &cbImageViewInfo, nullptr, &m_framebuffer.viewColor);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewColor);


    if(m_framebuffer.useResolved)
    {
      cbImageViewInfo.image = m_framebuffer.imgColorResolved.image;
      result                = vkCreateImageView(m_device, &cbImageViewInfo, nullptr, &m_framebuffer.viewColorResolved);
      assert(result == VK_SUCCESS);
      DEBUGUTIL_SET_NAME(m_framebuffer.viewColorResolved);
    }
  }
  {
    VkImageViewCreateInfo dsImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dsImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    dsImageViewInfo.format                          = m_framebuffer.depthStencilFormat;
    dsImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    dsImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    dsImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    dsImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    dsImageViewInfo.flags                           = 0;
    dsImageViewInfo.subresourceRange.levelCount     = 1;
    dsImageViewInfo.subresourceRange.baseMipLevel   = 0;
    dsImageViewInfo.subresourceRange.layerCount     = 1;
    dsImageViewInfo.subresourceRange.baseArrayLayer = 0;
    dsImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

    dsImageViewInfo.image = m_framebuffer.imgDepthStencil.image;
    result                = vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_framebuffer.viewDepthStencil);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewDepthStencil);

    dsImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    result = vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_framebuffer.viewDepth);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewDepth);
  }

  {
    VkImageViewCreateInfo dsImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dsImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    dsImageViewInfo.format                          = m_framebuffer.raytracingDepthFormat;
    dsImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    dsImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    dsImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    dsImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    dsImageViewInfo.flags                           = 0;
    dsImageViewInfo.subresourceRange.levelCount     = 1;
    dsImageViewInfo.subresourceRange.baseMipLevel   = 0;
    dsImageViewInfo.subresourceRange.layerCount     = 1;
    dsImageViewInfo.subresourceRange.baseArrayLayer = 0;
    dsImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    dsImageViewInfo.image = m_framebuffer.imgRaytracingDepth.image;
    result                = vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_framebuffer.viewRaytracingDepth);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewRaytracingDepth);
  }

  m_hiz.initUpdateViews(m_hizUpdate);
  m_hiz.updateDescriptorSet(m_hizUpdate, 0);

  // initial resource transitions
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    debugUtil.setObjectName(cmd, "framebufferCmd");

    cmdImageTransition(cmd, m_framebuffer.imgRaytracingDepth, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    cmdImageTransition(cmd, m_framebuffer.imgHizFar, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

    {
      HbaoPass::FrameConfig config;
      config.blend                   = true;
      config.sourceHeightScale       = m_hbaoFullRes ? 1 : supersample;
      config.sourceWidthScale        = m_hbaoFullRes ? 1 : supersample;
      config.targetWidth             = m_hbaoFullRes ? m_framebuffer.renderWidth : winWidth;
      config.targetHeight            = m_hbaoFullRes ? m_framebuffer.renderHeight : winHeight;
      config.sourceDepth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      config.sourceDepth.imageView   = m_framebuffer.viewDepth;
      config.sourceDepth.sampler     = VK_NULL_HANDLE;
      config.targetColor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      config.targetColor.imageView =
          m_framebuffer.useResolved && !m_hbaoFullRes ? m_framebuffer.viewColorResolved : m_framebuffer.viewColor;
      config.targetColor.sampler = VK_NULL_HANDLE;

      m_hbaoPass.initFrame(m_hbaoFrame, config, cmd);
    }

    tempSyncSubmit(cmd);
  }

  {
    VkViewport vp;
    VkRect2D   sc;
    vp.x        = 0;
    vp.y        = 0;
    vp.width    = float(m_framebuffer.renderWidth);
    vp.height   = float(m_framebuffer.renderHeight);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    sc.offset.x      = 0;
    sc.offset.y      = 0;
    sc.extent.width  = m_framebuffer.renderWidth;
    sc.extent.height = m_framebuffer.renderHeight;

    m_framebuffer.viewport = vp;
    m_framebuffer.scissor  = sc;
  }

  {
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

    pipelineRenderingInfo.colorAttachmentCount    = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &m_framebuffer.colorFormat;
    pipelineRenderingInfo.depthAttachmentFormat   = m_framebuffer.depthStencilFormat;

    m_framebuffer.pipelineRenderingInfo = pipelineRenderingInfo;
  }

  {
    VkDescriptorImageInfo imageInfo{VK_NULL_HANDLE, m_framebuffer.viewColor, VK_IMAGE_LAYOUT_GENERAL};
    m_sky.setOutImage(imageInfo);
  }

  return true;
}

void Resources::deinitFramebuffer()
{
  synchronize("sync deinitFramebuffer");

  vkDestroyImageView(m_device, m_framebuffer.viewColor, nullptr);
  vkDestroyImageView(m_device, m_framebuffer.viewDepthStencil, nullptr);
  vkDestroyImageView(m_device, m_framebuffer.viewDepth, nullptr);
  m_framebuffer.viewColor        = VK_NULL_HANDLE;
  m_framebuffer.viewDepthStencil = VK_NULL_HANDLE;
  m_framebuffer.viewDepth        = VK_NULL_HANDLE;

  m_allocator.destroy(m_framebuffer.imgColor);
  m_allocator.destroy(m_framebuffer.imgDepthStencil);
  if(m_framebuffer.imgColorResolved.image)
  {
    vkDestroyImageView(m_device, m_framebuffer.viewColorResolved, nullptr);
    m_framebuffer.viewColorResolved = VK_NULL_HANDLE;

    m_allocator.destroy(m_framebuffer.imgColorResolved);
  }

  vkDestroyImageView(m_device, m_framebuffer.viewRaytracingDepth, nullptr);
  m_framebuffer.viewRaytracingDepth = VK_NULL_HANDLE;
  m_allocator.destroy(m_framebuffer.imgRaytracingDepth);

  m_hiz.deinitUpdateViews(m_hizUpdate);
  m_hbaoPass.deinitFrame(m_hbaoFrame);

  m_allocator.destroy(m_framebuffer.imgHizFar);
}

void Resources::cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  // reduce amount of timers a bit
  // auto timerSection = profiler.timeRecurring("HiZ", cmd);

  // transition depth read optimal
  cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  m_hiz.cmdUpdateHiz(cmd, m_hizUpdate, (uint32_t)0);
}

void Resources::cmdHBAO(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  auto timerSection = profiler.timeRecurring("HBAO", cmd);

  bool useResolved = m_framebuffer.useResolved && !m_hbaoFullRes;

  // transition color to general
  cmdImageTransition(cmd, useResolved ? m_framebuffer.imgColorResolved : m_framebuffer.imgColor,
                     VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

  // transition depth read optimal
  cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  m_hbaoPass.cmdCompute(cmd, m_hbaoFrame, frame.hbaoSettings);
}

void Resources::cmdDynamicState(VkCommandBuffer cmd) const
{
  vkCmdSetViewport(cmd, 0, 1, &m_framebuffer.viewport);
  vkCmdSetScissor(cmd, 0, 1, &m_framebuffer.scissor);
}

void Resources::cmdBeginRendering(VkCommandBuffer cmd, bool hasSecondary, VkAttachmentLoadOp loadOpColor, VkAttachmentLoadOp loadOpDepth)
{
  VkClearValue colorClear{.color = {m_bgColor.x, m_bgColor.y, m_bgColor.z, m_bgColor.w}};
  VkClearValue depthClear{.depthStencil = {1.0F, 0}};

  // transfers & compute
  {
    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | m_supportedSaderStageFlags,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | m_supportedSaderStageFlags, VK_FALSE, 1, &memBarrier, 0,
                         nullptr, 0, nullptr);
  }

  cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

  VkRenderingAttachmentInfo colorAttachment = {
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_framebuffer.viewColor,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp      = loadOpColor,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = colorClear,
  };

  // Shared depth attachment
  VkRenderingAttachmentInfo depthStencilAttachment{
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_framebuffer.viewDepthStencil,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp      = loadOpDepth,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = depthClear,
  };

  // Dynamic rendering information: color and depth attachments
  VkRenderingInfo renderingInfo{
      .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags                = hasSecondary ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT : VkRenderingFlags(0),
      .renderArea           = m_framebuffer.scissor,
      .layerCount           = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments    = &colorAttachment,
      .pDepthAttachment     = &depthStencilAttachment,
  };

  vkCmdBeginRendering(cmd, &renderingInfo);
}

static VkAccessFlags getLayoutAccessFlags(VkImageLayout layout)
{
  switch(layout)
  {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return 0;
    case VK_IMAGE_LAYOUT_GENERAL:
      return VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_ACCESS_TRANSFER_READ_BIT;
    default:
      assert(0);
      return 0;
  }
}

void Resources::cmdImageTransition(VkCommandBuffer cmd, RImage& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier) const
{
  if(newLayout == rimg.layout && !needBarrier)
    return;

  VkAccessFlags src = getLayoutAccessFlags(rimg.layout);
  VkAccessFlags dst = getLayoutAccessFlags(newLayout);

  VkPipelineStageFlags srcPipe = nvvk::makeAccessMaskPipelineStageFlags(src, m_supportedSaderStageFlags);
  VkPipelineStageFlags dstPipe = nvvk::makeAccessMaskPipelineStageFlags(dst, m_supportedSaderStageFlags);

  VkImageSubresourceRange range;
  memset(&range, 0, sizeof(range));
  range.aspectMask     = aspects;
  range.baseMipLevel   = 0;
  range.levelCount     = VK_REMAINING_MIP_LEVELS;
  range.baseArrayLayer = 0;
  range.layerCount     = VK_REMAINING_ARRAY_LAYERS;

  VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  memBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  memBarrier.dstAccessMask        = dst;
  memBarrier.srcAccessMask        = src;
  memBarrier.oldLayout            = rimg.layout;
  memBarrier.newLayout            = newLayout;
  memBarrier.image                = rimg.image;
  memBarrier.subresourceRange     = range;

  rimg.layout = newLayout;

  vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, nullptr, 0, nullptr, 1, &memBarrier);
}

VkCommandBuffer Resources::createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear, bool isCompute) const
{
  VkResult result;
  bool     secondary = !primary;

  // Create the command buffer.
  VkCommandBufferAllocateInfo cmdInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdInfo.commandPool                 = pool;
  cmdInfo.level                       = primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  cmdInfo.commandBufferCount          = 1;
  VkCommandBuffer cmd;
  result = vkAllocateCommandBuffers(m_device, &cmdInfo, &cmd);
  assert(result == VK_SUCCESS);

  cmdBegin(cmd, singleshot, primary, secondaryInClear, isCompute);

  return cmd;
}

RBuffer Resources::createBuffer(VkDeviceSize size, VkBufferUsageFlags flags, VkMemoryPropertyFlags memFlags, const std::vector<uint32_t>* sharingQueueFamilies)
{
  RBuffer entry = {nullptr};

  if(size)
  {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if(sharingQueueFamilies && !sharingQueueFamilies->empty())
    {
      info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
      info.queueFamilyIndexCount = uint32_t(sharingQueueFamilies->size());
      info.pQueueFamilyIndices   = sharingQueueFamilies->data();
    }

    ((nvvk::Buffer&)entry) = m_allocator.createBuffer(info, memFlags);
    entry.info.buffer      = entry.buffer;
    entry.info.offset      = 0;
    entry.info.range       = size;
    entry.address          = nvvk::getBufferDeviceAddress(m_device, entry.buffer);
    if(memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
      entry.mapping = m_allocator.map(entry);
    }
  }

  return entry;
}

RLargeBuffer Resources::createLargeBuffer(VkDeviceSize                 size,
                                          VkBufferUsageFlags           flags,
                                          VkMemoryPropertyFlags        memFlags,
                                          const std::vector<uint32_t>* sharingQueueFamilies)
{
  RLargeBuffer entry = {nullptr};

  if(size)
  {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if(sharingQueueFamilies && !sharingQueueFamilies->empty())
    {
      info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
      info.queueFamilyIndexCount = uint32_t(sharingQueueFamilies->size());
      info.pQueueFamilyIndices   = sharingQueueFamilies->data();
    }

    ((nvvk::LargeBuffer&)entry) = m_allocator.createLargeBuffer(m_queue, info, memFlags);
    entry.info.buffer           = entry.buffer;
    entry.info.offset           = 0;
    entry.info.range            = size;
    entry.address               = nvvk::getBufferDeviceAddress(m_device, entry.buffer);
  }

  return entry;
}

nvvk::AccelKHR Resources::createAccelKHR(VkAccelerationStructureCreateInfoKHR& createInfo)
{
  nvvk::AccelKHR obj = m_allocator.createAcceleration(createInfo);

  VkAccelerationStructureDeviceAddressInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  info.accelerationStructure = obj.accel;
  obj.address                = vkGetAccelerationStructureDeviceAddressKHR(m_device, &info);

  return obj;
}

void Resources::cmdBegin(VkCommandBuffer cmd, bool singleshot, bool primary, bool secondaryInClear, bool isCompute) const
{
  VkResult result;
  bool     secondary = !primary;

  VkCommandBufferInheritanceInfo inheritInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
  VkCommandBufferInheritanceRenderingInfo inheritRenderInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO};

  if(secondary && !isCompute)
  {
    inheritInfo.pNext                         = &inheritRenderInfo;
    inheritRenderInfo.rasterizationSamples    = m_basicGraphicsState.multisampleState.rasterizationSamples;
    inheritRenderInfo.colorAttachmentCount    = 1;
    inheritRenderInfo.pColorAttachmentFormats = &m_framebuffer.colorFormat;
    inheritRenderInfo.depthAttachmentFormat   = m_framebuffer.depthStencilFormat;
    inheritRenderInfo.flags                   = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
  }

  VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  // the sample is resubmitting re-use commandbuffers to the queue while they may still be executed by GPU
  // we only use fences to prevent deleting commandbuffers that are still in flight
  beginInfo.flags = singleshot ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  // the sample's secondary buffers always are called within passes as they contain drawcalls
  beginInfo.flags |= secondary && !isCompute ? VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : 0;
  beginInfo.pInheritanceInfo = &inheritInfo;

  result = vkBeginCommandBuffer(cmd, &beginInfo);
  assert(result == VK_SUCCESS);
}

void Resources::destroy(RBuffer& obj)
{
  if(obj.mapping)
  {
    m_allocator.unmap(obj);
  }
  m_allocator.destroy(obj);
  obj.info    = {nullptr};
  obj.mapping = nullptr;
}

void Resources::destroy(nvvk::AccelKHR& obj)
{
  m_allocator.destroy(obj);
}

void Resources::destroy(RLargeBuffer& obj)
{
  m_allocator.destroy(obj);
  obj.info = {nullptr};
}

void Resources::simpleUploadBuffer(const RBuffer& dst, const void* src)
{
  if(src && dst.info.range)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_allocator.getStaging()->cmdToBuffer(cmd, dst.buffer, 0, dst.info.range, src);
    tempSyncSubmit(cmd);
  }
}

void Resources::simpleUploadBuffer(const RBuffer& dst, size_t offset, size_t sz, const void* src)
{
  if(src && dst.info.range)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_allocator.getStaging()->cmdToBuffer(cmd, dst.buffer, offset, sz, src);
    tempSyncSubmit(cmd);
  }
}

VkCommandBuffer Resources::createTempCmdBuffer()
{
  VkCommandBuffer cmd = m_tempCommandPool.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

  nvvk::DebugUtil(m_device).setObjectName(cmd, "tempCmdBuffer");

  return cmd;
}

void Resources::simpleDownloadBuffer(void* dst, const RBuffer& src)
{
  if(dst && src.info.range)
  {
    VkCommandBuffer cmd    = createTempCmdBuffer();
    const void*     mapped = m_allocator.getStaging()->cmdFromBuffer(cmd, src.buffer, 0, src.info.range);
    // Ensure writes to the buffer we're mapping are accessible by the host
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    tempSyncSubmit(cmd, false);
    memcpy(dst, mapped, src.info.range);
    tempResetResources();
  }
}

void Resources::tempSyncSubmit(VkCommandBuffer cmd, bool reset)
{
  m_tempCommandPool.submitAndWait(cmd);

  if(reset)
  {
    tempResetResources();
  }
  else
  {
    synchronize("sync tempSyncSubmit");
  }
}

void Resources::tempAbort(VkCommandBuffer cmd, bool reset /*= true*/)
{
  m_tempCommandPool.destroy(cmd);
  if(reset)
  {
    tempResetResources();
  }
  else
  {
    synchronize("sync tempSyncSubmit");
  }
}

void Resources::tempResetResources()
{
  synchronize("sync resetTempResources");
  m_allocator.finalizeStaging();
  m_allocator.releaseStaging();
}

void Resources::synchronize(const char* debugMsg)
{
  VkResult result = vkDeviceWaitIdle(m_device);
  nvvk::checkResult(result, debugMsg ? debugMsg : "Resources::synchronize");
}

bool Resources::verifyShaders(size_t numShaders, nvvk::ShaderModuleID* shaders)
{
  bool valid = true;
  for(size_t i = 0; i < numShaders; i++)
  {
    if(shaders[i].isValid() && !m_shaderManager.isValid(shaders[i]))
    {
      valid = false;
      break;
    }
#if defined(_DEBUG) && 1
    else if(shaders[i].isValid())
    {
      nvvk::ShaderModuleManager::ShaderModule& module   = m_shaderManager.getShaderModule(shaders[i]);
      std::string                              filename = module.definition.filename;
      filename                                          = filename.substr(0, filename.length() - 4) + "spv";
      m_shaderManager.dumpSPIRV(shaders[i], filename.c_str());
    }
#endif
  }

  if(!valid)
  {
    for(size_t i = 0; i < numShaders; i++)
    {
      m_shaderManager.destroyShaderModule(shaders[i]);
    }
  }

  return valid;
}

void Resources::destroyShaders(size_t numShaders, nvvk::ShaderModuleID* shaders)
{
  for(size_t i = 0; i < numShaders; i++)
  {
    m_shaderManager.destroyShaderModule(shaders[i]);
  }
}

VkDeviceSize Resources::getDeviceLocalHeapSize() const
{
  const VkPhysicalDeviceMemoryProperties& memProperties = m_context->m_physicalInfo.memoryProperties;

  for(uint32_t type = 0; type < memProperties.memoryTypeCount; type++)
  {
    // find the heap that is purely tagged as device local
    if(memProperties.memoryTypes[type].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {
      return memProperties.memoryHeaps[memProperties.memoryTypes[type].heapIndex].size;
    }
  }

  // otherwise take something that is device local and host visible
  for(uint32_t type = 0; type < memProperties.memoryTypeCount; type++)
  {
    if((memProperties.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
       == (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    {
      return memProperties.memoryHeaps[memProperties.memoryTypes[type].heapIndex].size;
    }
  }
  assert(0);
  return 0;
}

bool Resources::isBufferSizeValid(VkDeviceSize size) const
{
  return size <= m_context->m_physicalInfo.properties13.maxBufferSize
         && size <= m_context->m_physicalInfo.properties11.maxMemoryAllocationSize;
}

void QueueState::init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue)
{
  assert(m_device == nullptr);

  m_device      = device;
  m_queue       = queue;
  m_familyIndex = familyIndex;

  VkSemaphoreTypeCreateInfo timelineSemaphoreCreateInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                        .pNext         = nullptr,
                                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                                        .initialValue  = initialValue};
  VkSemaphoreCreateInfo     semaphoreCreateInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineSemaphoreCreateInfo, .flags = 0};

  vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &m_timelineSemaphore);

  m_device        = device;
  m_timelineValue = initialValue + 1;
}

void QueueState::deinit()
{
  if(!m_device)
    return;
  vkDestroySemaphore(m_device, m_timelineSemaphore, nullptr);
}

VkSemaphoreSubmitInfo QueueState::getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex /*= 0*/) const
{
  VkSemaphoreSubmitInfo signalSubmitInfo{.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                         .pNext       = nullptr,
                                         .semaphore   = m_timelineSemaphore,
                                         .value       = m_timelineValue,
                                         .stageMask   = stageMask,
                                         .deviceIndex = deviceIndex};

  return signalSubmitInfo;
}

VkSemaphoreSubmitInfo QueueState::advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex /*= 0*/)
{
  VkSemaphoreSubmitInfo signalSubmitInfo{.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                         .pNext       = nullptr,
                                         .semaphore   = m_timelineSemaphore,
                                         .value       = m_timelineValue++,
                                         .stageMask   = stageMask,
                                         .deviceIndex = deviceIndex};

  return signalSubmitInfo;
}

}  // namespace lodclusters
