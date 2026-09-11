/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resources.hpp"

namespace lodclusters {

//////////////////////////////////////////////////////////////////////////
//
// StreamingAllocatorVis
//
// Renders the memory state of the persistent CLAS allocator into an offscreen texture
// that the UI displays, and bins the resident groups by their allocation size.
// Only updated while the UI widget is visible. Everything but the storage image goes
// through push constants, so it is independent of the scene's lifetime.
//
// see `stream_allocator_vis.comp.glsl`

class StreamingAllocatorVis
{
public:
  // upper bound, the mapped sub-rectangle is sized to the pixels the UI has for it
  static const uint32_t IMAGE_WIDTH      = 2048;
  static const uint32_t IMAGE_HEIGHT     = 2048;
  static const uint32_t MIN_MAPPED_WIDTH = 16;

  struct FrameInfo
  {
    // buffer holding `shaderio::SceneStreaming`
    VkDeviceAddress streamingAddress = 0;
    // may lag the device by a frame, the shader clamps against the device side count
    uint32_t activeGroupsCount = 0;
    // as in the `VISUALIZE_GROUP` render mode
    uint32_t colorXor   = 0;
    bool     showGroups = true;
    // pixels the UI has for the texture, it is generated at that size. Zero uses the image.
    uint32_t displayWidth  = 0;
    uint32_t displayHeight = 0;
    // costs an extra dispatch and a readback, so only while the plot is open
    bool     wantHistogram = false;
    uint32_t cycleIndex    = 0;
  };

  // histogram readback ring, `FrameInfo::cycleIndex` picks a slot
  static const uint32_t HISTOGRAM_CYCLES = 4;
  // the histogram's size axis ends on a multiple of this
  static const uint64_t HISTOGRAM_AXIS_STEP_BYTES = 32 * 1024;

  bool init(Resources& res);
  void deinit(Resources& res);

  bool reloadShaders(Resources& res);

  // must run after all streaming operations of the frame were recorded.
  // false if there is nothing to visualize yet, the texture is left untouched then.
  bool cmdUpdate(VkCommandBuffer cmd, const shaderio::StreamingAllocator& allocator, const FrameInfo& frameInfo);

  VkDescriptorSet getImguiTexture() const { return m_imguiTexture; }

  // the below are valid after a `cmdUpdate` that returned true

  VkExtent2D getMappedExtent() const { return {m_constants.usedWidth, m_constants.usedHeight}; }
  uint32_t   getRowCount() const { return m_rowCount; }
  // image rows per memory row, from two on the last one is a blank separator, which a
  // fractional vertical scale would sample unevenly
  uint32_t getRowPitch() const { return m_constants.rowPitch; }
  // CLAS bytes a pixel resp. row of the mapped rectangle covers
  uint64_t getPixelBytes() const { return uint64_t(m_constants.unitsPerPixel) << m_granularityByteShift; }
  uint64_t getRowBytes() const { return uint64_t(m_constants.unitsPerRow) << m_granularityByteShift; }

  // resident groups binned by allocation size, null until a `wantHistogram` update completed
  const uint32_t* getHistogram(uint32_t cycleIndex) const;
  // bins worth plotting, grows to the largest allocation seen and never shrinks again
  uint32_t getHistogramPlotBins(const uint32_t* counts);
  uint64_t getHistogramBinBytes() const;

private:
  struct Shaders
  {
    shaderc::SpvCompilationResult computeMemory;
    shaderc::SpvCompilationResult computeGroups;
    shaderc::SpvCompilationResult computeHistogram;
  };

  struct Pipelines
  {
    VkPipeline computeMemory    = nullptr;
    VkPipeline computeGroups    = nullptr;
    VkPipeline computeHistogram = nullptr;
  };

  bool initShadersAndPipelines(Resources& res);
  void deinitShadersAndPipelines(Resources& res);

  Shaders              m_shaders;
  Pipelines            m_pipelines;
  VkPipelineLayout     m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;

  nvvk::Image     m_image{};
  VkDescriptorSet m_imguiTexture{};

  nvvk::Buffer                m_histogramBuffer{};
  nvvk::BufferTyped<uint32_t> m_histogramHostBuffer{};

  shaderio::StreamingAllocatorVisConstants m_constants{};
  uint32_t                                 m_granularityByteShift = 0;
  uint32_t                                 m_rowCount             = 0;
  uint32_t                                 m_maxAllocationSize    = 0;
  uint32_t                                 m_histogramPlotBins    = 0;
  // one bit per readback slot, set once filled for the current allocator scale
  uint32_t m_histogramSlotsFilled = 0;
};

}  // namespace lodclusters
