/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <volk.h>
#include <backends/imgui_impl_vulkan.h>
#include <nvvk/barriers.hpp>

#include "scene_streaming_vis.hpp"

namespace lodclusters {

bool StreamingAllocatorVis::init(Resources& res)
{
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_ALLOCATOR_VIS_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_dsetPack.init(bindings, res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()},
                               {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::StreamingAllocatorVisConstants)}});
  }

  {
    // stays in VK_IMAGE_LAYOUT_GENERAL for its lifetime, written by compute, sampled by imgui
    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType         = VK_IMAGE_TYPE_2D;
    imageInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent            = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    imageInfo.mipLevels         = 1;
    imageInfo.arrayLayers       = 1;
    imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImageViewCreateInfo imageViewInfo       = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    imageViewInfo.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format                      = imageInfo.format;
    imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewInfo.subresourceRange.levelCount = 1;
    imageViewInfo.subresourceRange.layerCount = 1;

    if(res.m_allocator.createImage(m_image, imageInfo, imageViewInfo) != VK_SUCCESS)
    {
      deinit(res);
      return false;
    }
    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.imageView);
  }

  // initial layout and content
  {
    VkCommandBuffer cmd = res.createTempCmdBuffer();
    res.cmdImageTransition(cmd, m_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

    VkClearColorValue       clearColor = {};
    VkImageSubresourceRange range      = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_image.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    res.tempSyncSubmit(cmd);
  }

  {
    nvvk::WriteSetContainer writeSets;
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_ALLOCATOR_VIS_IMAGE), m_image.descriptor);
    vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);
  }

  {
    const size_t binCount = STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS;

    NVVK_CHECK(res.createBuffer(m_histogramBuffer, sizeof(uint32_t) * binCount,
                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
                                    | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    NVVK_DBG_NAME(m_histogramBuffer.buffer);

    NVVK_CHECK(res.createBufferTyped(m_histogramHostBuffer, binCount * HISTOGRAM_CYCLES,
                                     VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    NVVK_DBG_NAME(m_histogramHostBuffer.buffer);

    memset(m_histogramHostBuffer.data(), 0, sizeof(uint32_t) * binCount * HISTOGRAM_CYCLES);
  }

  // the imgui backend owns the sampler, the UI picks nearest per draw command
  m_imguiTexture = ImGui_ImplVulkan_AddTexture(m_image.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL);

  if(!initShadersAndPipelines(res))
  {
    deinit(res);
    return false;
  }

  return true;
}

void StreamingAllocatorVis::deinit(Resources& res)
{
  deinitShadersAndPipelines(res);

  if(m_imguiTexture)
  {
    ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);
    m_imguiTexture = nullptr;
  }
  res.m_allocator.destroyImage(m_image);
  res.m_allocator.destroyBuffer(m_histogramBuffer);
  res.m_allocator.destroyBuffer(m_histogramHostBuffer);

  m_dsetPack.deinit();
  if(m_pipelineLayout)
  {
    vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);
    m_pipelineLayout = nullptr;
  }
}

bool StreamingAllocatorVis::reloadShaders(Resources& res)
{
  res.destroyPipelines(m_pipelines);
  return initShadersAndPipelines(res);
}

bool StreamingAllocatorVis::initShadersAndPipelines(Resources& res)
{
  shaderc::CompileOptions optionsMemory = res.makeCompilerOptions();
  // both passes always use a plain 1D dispatch
  optionsMemory.AddMacroDefinition("USE_16BIT_DISPATCH", "0");
  shaderc::CompileOptions optionsGroups    = optionsMemory;
  shaderc::CompileOptions optionsHistogram = optionsMemory;

  optionsMemory.AddMacroDefinition("STREAM_ALLOCATOR_VIS_PASS", "STREAM_ALLOCATOR_VIS_PASS_MEMORY");
  optionsGroups.AddMacroDefinition("STREAM_ALLOCATOR_VIS_PASS", "STREAM_ALLOCATOR_VIS_PASS_GROUPS");
  optionsHistogram.AddMacroDefinition("STREAM_ALLOCATOR_VIS_PASS", "STREAM_ALLOCATOR_VIS_PASS_HISTOGRAM");

  res.compileShader(m_shaders.computeMemory, VK_SHADER_STAGE_COMPUTE_BIT, "stream_allocator_vis.comp.glsl", &optionsMemory);
  res.compileShader(m_shaders.computeGroups, VK_SHADER_STAGE_COMPUTE_BIT, "stream_allocator_vis.comp.glsl", &optionsGroups);
  res.compileShader(m_shaders.computeHistogram, VK_SHADER_STAGE_COMPUTE_BIT, "stream_allocator_vis.comp.glsl", &optionsHistogram);

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  VkShaderModuleCreateInfo    shaderInfo = {};
  compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
  compInfo.stage.pName                   = "main";
  compInfo.stage.pNext                   = &shaderInfo;
  compInfo.layout                        = m_pipelineLayout;

  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeMemory);
  NVVK_CHECK(vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeMemory));

  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeGroups);
  NVVK_CHECK(vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeGroups));

  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeHistogram);
  NVVK_CHECK(vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeHistogram));

  return true;
}

void StreamingAllocatorVis::deinitShadersAndPipelines(Resources& res)
{
  res.destroyPipelines(m_pipelines);
}

bool StreamingAllocatorVis::cmdUpdate(VkCommandBuffer cmd, const shaderio::StreamingAllocator& allocator, const FrameInfo& frameInfo)
{
  if(!m_pipelines.computeMemory || !allocator.usedBits || !allocator.sectorCount || !frameInfo.streamingAddress)
  {
    return false;
  }

  // A sector holds `1 << sectorSizeShift` uint32, hence that many times 32 allocation units.
  const uint32_t sectorUnitsShift = allocator.sectorSizeShift + 5;

  // Generated to fit the pixels the UI has, so it is only ever magnified when drawn.
  // Detail is reduced here instead, by covering more memory per pixel and per row.
  const uint32_t displayWidth =
      std::clamp(frameInfo.displayWidth ? frameInfo.displayWidth : IMAGE_WIDTH, MIN_MAPPED_WIDTH, IMAGE_WIDTH);
  const uint32_t displayHeight = frameInfo.displayHeight ? frameInfo.displayHeight : IMAGE_HEIGHT;

  // whole sectors per row, so an allocation never straddles rows either
  const uint32_t maxRows       = std::min(IMAGE_HEIGHT, std::max(1u, displayHeight));
  const uint32_t sectorsPerRow = (allocator.sectorCount + maxRows - 1) / maxRows;
  const uint32_t rows          = (allocator.sectorCount + sectorsPerRow - 1) / sectorsPerRow;

  const uint64_t unitsPerRow = uint64_t(sectorsPerRow) << sectorUnitsShift;

  // any integer is fine, the memory pass integrates the bit field over the pixel
  // rather than point sampling it, so this cannot alias
  const uint32_t unitsPerPixel = std::max(1u, uint32_t((unitsPerRow + displayWidth - 1) / displayWidth));

  // as tall as the height affords, the last image row of a pitch >= 2 is a blank separator
  const uint32_t rowPitch = std::clamp(displayHeight / rows, 1u, IMAGE_HEIGHT / rows);

  m_granularityByteShift = allocator.granularityByteShift;
  m_rowCount             = rows;

  if(m_maxAllocationSize != allocator.maxAllocationSize)
  {
    // bins cover a different size range now, what the slots hold no longer applies
    m_histogramPlotBins    = 0;
    m_histogramSlotsFilled = 0;
  }

  m_constants                  = {};
  m_constants.streamingAddress = frameInfo.streamingAddress;
  m_constants.usedWidth        = uint32_t((unitsPerRow + unitsPerPixel - 1) / unitsPerPixel);
  m_constants.usedHeight       = rows * rowPitch;
  m_constants.rowPitch         = rowPitch;
  m_constants.rowContent       = rowPitch > 1 ? rowPitch - 1 : 1;
  m_constants.unitsPerPixel    = unitsPerPixel;
  m_constants.unitsPerRow      = uint32_t(unitsPerRow);
  m_constants.totalUnits       = uint32_t(uint64_t(allocator.sectorCount) << sectorUnitsShift);
  m_constants.groupCount       = frameInfo.activeGroupsCount;
  m_constants.colorXor         = frameInfo.colorXor;

  m_maxAllocationSize = allocator.maxAllocationSize;

  const bool runHistogram = frameInfo.wantHistogram && m_maxAllocationSize != 0 && frameInfo.activeGroupsCount != 0
                            && m_pipelines.computeHistogram;
  if(runHistogram)
  {
    m_constants.histogramAddress  = m_histogramBuffer.address;
    m_constants.maxAllocationSize = m_maxAllocationSize;
  }

  assert(m_constants.usedWidth <= IMAGE_WIDTH && m_constants.usedHeight <= IMAGE_HEIGHT);

  // the streaming kernels of this frame wrote the allocator state we are about to read
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_constants), &m_constants);

  // occupancy, this also clears the rectangle for the group pass
  const uint32_t pixelCount = m_constants.usedWidth * m_constants.usedHeight;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeMemory);
  vkCmdDispatch(cmd, (pixelCount + STREAM_ALLOCATOR_VIS_WORKGROUP - 1) / STREAM_ALLOCATOR_VIS_WORKGROUP, 1, 1);

  // then the individual group allocations on top
  if(frameInfo.showGroups && frameInfo.activeGroupsCount)
  {
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeGroups);
    vkCmdDispatch(cmd, (frameInfo.activeGroupsCount + STREAM_ALLOCATOR_VIS_WORKGROUP - 1) / STREAM_ALLOCATOR_VIS_WORKGROUP, 1, 1);
  }

  // imgui samples the image later in this frame
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

  // an extra dispatch and readback nothing else needs, only while the UI plots it
  if(runHistogram)
  {
    const VkDeviceSize histogramSize = sizeof(uint32_t) * STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS;

    vkCmdFillBuffer(cmd, m_histogramBuffer.buffer, 0, histogramSize, 0);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeHistogram);
    vkCmdDispatch(cmd, (frameInfo.activeGroupsCount + STREAM_ALLOCATOR_VIS_WORKGROUP - 1) / STREAM_ALLOCATOR_VIS_WORKGROUP, 1, 1);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferCopy region = {};
    region.srcOffset    = 0;
    region.dstOffset    = VkDeviceSize(frameInfo.cycleIndex % HISTOGRAM_CYCLES) * histogramSize;
    region.size         = histogramSize;
    vkCmdCopyBuffer(cmd, m_histogramBuffer.buffer, m_histogramHostBuffer.buffer, 1, &region);

    // readable once this slot comes around again, the app waits on that frame first
    m_histogramSlotsFilled |= 1u << (frameInfo.cycleIndex % HISTOGRAM_CYCLES);
  }

  return true;
}

const uint32_t* StreamingAllocatorVis::getHistogram(uint32_t cycleIndex) const
{
  if(!m_histogramHostBuffer.buffer || !m_maxAllocationSize || !(m_histogramSlotsFilled & (1u << (cycleIndex % HISTOGRAM_CYCLES))))
  {
    return nullptr;
  }

  return m_histogramHostBuffer.data() + size_t(cycleIndex % HISTOGRAM_CYCLES) * STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS;
}

uint32_t StreamingAllocatorVis::getHistogramPlotBins(const uint32_t* counts)
{
  const uint64_t binBytes = getHistogramBinBytes();
  if(!counts || !binBytes)
  {
    return STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS;
  }

  uint32_t lastUsed = 0;
  for(uint32_t i = 0; i < STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS; i++)
  {
    if(counts[i])
    {
      lastUsed = i;
    }
  }

  // rounded up to a multiple of the step and never shrinking, so the axis stays put
  const uint64_t peakBytes = uint64_t(lastUsed + 1) * binBytes;
  const uint64_t axisBytes = nvutils::align_up(std::max(peakBytes, HISTOGRAM_AXIS_STEP_BYTES), HISTOGRAM_AXIS_STEP_BYTES);
  const uint32_t bins = std::min(uint32_t(STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS), uint32_t((axisBytes + binBytes - 1) / binBytes));

  m_histogramPlotBins = std::max(m_histogramPlotBins, bins);

  return m_histogramPlotBins;
}

uint64_t StreamingAllocatorVis::getHistogramBinBytes() const
{
  // bins split the 1..maxAllocationSize unit range evenly
  const uint64_t unitsPerBin = (m_maxAllocationSize + STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS - 1) / STREAM_ALLOCATOR_VIS_HISTOGRAM_BINS;
  return unitsPerBin << m_granularityByteShift;
}

}  // namespace lodclusters
