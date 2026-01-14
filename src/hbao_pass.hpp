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

#pragma once

#include <cassert>

#include <vulkan/vulkan_core.h>
#include <nvvkglsl/glsl.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <glm/glm.hpp>

//////////////////////////////////////////////////////////////////////////

/// HbaoSystem implements a screen-space
/// ambient occlusion effect using
/// horizon-based ambient occlusion.
/// See https://github.com/nvpro-samples/gl_ssao
/// for more details
///
/// Newer version was derived from
/// https://github.com/NVIDIA-RTX/Donut/blob/main/shaders/passes/ssao_compute_cs.hlsl

class HbaoPass
{
public:
  static const int RANDOM_SIZE     = 4;
  static const int RANDOM_ELEMENTS = RANDOM_SIZE * RANDOM_SIZE;

  struct Config
  {
    VkFormat targetFormat;
    uint32_t maxFrames;
  };

  bool init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool, nvvkglsl::GlslCompiler* glslCompiler, const Config& config);
  bool reloadShaders();
  void deinit();

  struct FrameConfig
  {
    uint32_t targetWidth;
    uint32_t targetHeight;

    VkDescriptorImageInfo sourceDepth;
    VkDescriptorImageInfo targetColor;
  };

  struct FrameImages
  {
    nvvk::Image resultArray;
    nvvk::Image linearDepthArray;
    nvvk::Image viewNormal;
  };

  struct Frame
  {
    uint32_t slot = ~0u;

    FrameImages images;
    int         width;
    int         height;

    FrameConfig config;
  };

  bool initFrame(Frame& frame, const FrameConfig& config, VkCommandBuffer cmd);
  void deinitFrame(Frame& frame);


  struct View
  {
    float     nearPlane;
    float     farPlane;
    float     tanFovy;
    glm::mat4 projectionMatrix;
    glm::mat4 viewMatrix;
  };

  struct Settings
  {
    View view;

    // percentage of far plane at which we fade out AO effect
    float backgroundViewDepth = 0.5f;

    float intensity     = 1.0f;
    float radius        = 1.0f;
    float bias          = 0.1f;
    float powerExponent = 4.0;
    bool  blur          = true;
  };

  // before: must do appropriate barriers for color write access and depth read access
  // after:  from compute write to whatever output image needs
  void cmdCompute(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;

private:
  struct Shaders
  {
    shaderc::SpvCompilationResult deinterleave{};
    shaderc::SpvCompilationResult calc{};
    shaderc::SpvCompilationResult applyRaw{};
    shaderc::SpvCompilationResult applyBlurred{};
    shaderc::SpvCompilationResult viewNormal{};
  };

  struct Pipelines
  {
    VkPipeline deinterleave{};
    VkPipeline calc{};
    VkPipeline applyRaw{};
    VkPipeline applyBlurred{};
    VkPipeline viewNormal{};
  };

  VkDevice                 m_device{};
  nvvk::ResourceAllocator* m_allocator{};
  nvvk::SamplerPool*       m_samplerPool{};
  nvvkglsl::GlslCompiler*  m_glslCompiler{};

  uint64_t m_slotsUsed = {};
  Config   m_config;

  nvvk::DescriptorPack m_dsetPack;
  VkPipelineLayout     m_pipelineLayout{};

  nvvk::Buffer           m_ubo;
  VkDescriptorBufferInfo m_uboInfo;

  VkSampler m_linearSampler{};
  VkSampler m_nearestSampler{};

  Shaders   m_shaders;
  Pipelines m_pipelines;

  void updatePipelines();
  void updateUbo(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;
};
