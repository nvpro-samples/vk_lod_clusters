/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef HBAOPASS_H__
#define HBAOPASS_H__

#include <assert.h>

#include <vulkan/vulkan_core.h>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvh/trangeallocator.hpp>
#include <glm/glm.hpp>

//////////////////////////////////////////////////////////////////////////

/// HbaoSystem implements a screen-space
/// ambient occlusion effect using
/// horizon-based ambient occlusion.
/// See https://github.com/nvpro-samples/gl_ssao
/// for more details

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

  void init(VkDevice device, nvvk::ResourceAllocator* allocator, nvvk::ShaderModuleManager* shaderManager, const Config& config);
  void reloadShaders();
  void deinit();

  struct FrameConfig
  {
    bool blend;

    uint32_t sourceWidthScale;
    uint32_t sourceHeightScale;

    uint32_t targetWidth;
    uint32_t targetHeight;

    VkDescriptorImageInfo sourceDepth;
    VkDescriptorImageInfo targetColor;
  };

  struct FrameIMGs
  {
    nvvk::Texture depthlinear, viewnormal, result, blur, resultarray, deptharray;
  };

  struct Frame
  {
    uint32_t slot = ~0u;

    FrameIMGs images;
    int       width;
    int       height;

    FrameConfig config;
  };

  bool initFrame(Frame& frame, const FrameConfig& config, VkCommandBuffer cmd);
  void deinitFrame(Frame& frame);


  struct View
  {
    bool      isOrtho;
    float     nearPlane;
    float     farPlane;
    float     halfFovyTan;
    glm::mat4 projectionMatrix;
  };

  struct Settings
  {
    View view;

    float unit2viewspace = 1.0f;
    float intensity      = 1.0f;
    float radius         = 1.0f;
    float bias           = 0.1f;
    float blurSharpness  = 40.0f;
  };

  // before: must do appropriate barriers for color write access and depth read access
  // after:  from compute write to whatever output image needs
  void cmdCompute(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;

private:
  struct Shaders
  {
    nvvk::ShaderModuleID depth_linearize, viewnormal, blur, blur_apply, deinterleave, calc, reinterleave;
  };

  struct Pipelines
  {
    VkPipeline depth_linearize = VK_NULL_HANDLE;
    VkPipeline viewnormal      = VK_NULL_HANDLE;
    VkPipeline blur            = VK_NULL_HANDLE;
    VkPipeline blur_apply      = VK_NULL_HANDLE;
    VkPipeline deinterleave    = VK_NULL_HANDLE;
    VkPipeline calc            = VK_NULL_HANDLE;
    VkPipeline reinterleave    = VK_NULL_HANDLE;
  };

  VkDevice                   m_device;
  nvvk::ResourceAllocator*   m_allocator;
  nvvk::ShaderModuleManager* m_shaderManager;
  nvh::TRangeAllocator<1>    m_slots;
  Config                     m_config;

  nvvk::DescriptorSetContainer m_setup;

  nvvk::Buffer           m_ubo;
  VkDescriptorBufferInfo m_uboInfo;

  VkSampler m_linearSampler;

  Shaders   m_shaders;
  Pipelines m_pipelines;

  glm::vec4 m_hbaoRandom[RANDOM_ELEMENTS];

  void updatePipelines();
  void updateUbo(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;
};

#endif