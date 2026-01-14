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

#include <algorithm>
#include <random>

#include <volk.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/barriers.hpp>
#include <nvutils/logger.hpp>

#include "hbao_pass.hpp"
#include "../shaders/hbao.h"


bool HbaoPass::init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool, nvvkglsl::GlslCompiler* glslCompiler, const Config& config)
{
  m_device       = allocator->getDevice();
  m_allocator    = allocator;
  m_glslCompiler = glslCompiler;
  m_samplerPool  = samplerPool;

  assert(config.maxFrames <= 64);

  m_slotsUsed = 0;

  {
    VkSamplerCreateInfo createInfo = {
        .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
    };

    samplerPool->acquireSampler(m_linearSampler, createInfo);

    createInfo.minFilter = VK_FILTER_NEAREST;
    createInfo.magFilter = VK_FILTER_NEAREST;

    samplerPool->acquireSampler(m_nearestSampler, createInfo);
  }

  // descriptor sets
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(NVHBAO_MAIN_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_DEPTH, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                        VK_SHADER_STAGE_COMPUTE_BIT, &m_linearSampler);
    bindings.addBinding(NVHBAO_MAIN_TEX_VIEWNORMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_DEPTHARRAY, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_RESULTARRAY, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_VIEWNORMAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_DEPTHARRAY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_RESULTARRAY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_OUT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_dsetPack.init(bindings, m_device, config.maxFrames);

    nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_dsetPack.getLayout()}, {{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16}});
  }

  // pipelines
  if(!reloadShaders())
  {
    return false;
  }

  // ubo
  m_uboInfo.offset = 0;
  m_uboInfo.range  = (sizeof(glsl::NVHBAOData) + 255) & ~255;

  allocator->createBuffer(m_ubo, m_uboInfo.range * config.maxFrames, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  m_uboInfo.buffer = m_ubo.buffer;
  NVVK_DBG_NAME(m_ubo.buffer);


  return true;
}

static bool compileShader(nvvkglsl::GlslCompiler*        compiler,
                          shaderc::SpvCompilationResult& compiled,
                          VkShaderStageFlagBits          shaderStage,
                          const std::filesystem::path&   filePath,
                          shaderc::CompileOptions*       options = nullptr)
{
  compiled = compiler->compileFile(filePath, nvvkglsl::getShaderKind(shaderStage), options);
  if(compiled.GetCompilationStatus() == shaderc_compilation_status_success)
  {
    return true;
  }
  else
  {
    std::string errorMessage = compiled.GetErrorMessage();
    if(!errorMessage.empty())
      nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eWARNING, "%s", errorMessage.c_str());
    return false;
  }
}

bool HbaoPass::reloadShaders()
{
  shaderc::CompileOptions optionsRaw  = m_glslCompiler->options();
  shaderc::CompileOptions optionsBlur = m_glslCompiler->options();

  optionsRaw.AddMacroDefinition("NVHBAO_BLUR", "0");
  optionsBlur.AddMacroDefinition("NVHBAO_BLUR", "1");

  bool state = true;
  state = compileShader(m_glslCompiler, m_shaders.applyRaw, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_apply.comp.glsl", &optionsRaw) && state;
  state = compileShader(m_glslCompiler, m_shaders.applyBlurred, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_apply.comp.glsl", &optionsBlur)
          && state;
  state = compileShader(m_glslCompiler, m_shaders.calc, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_calc.comp.glsl") && state;
  state = compileShader(m_glslCompiler, m_shaders.deinterleave, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_deinterleave.comp.glsl") && state;
  state = compileShader(m_glslCompiler, m_shaders.viewNormal, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_viewnormal.comp.glsl") && state;

  if(state)
  {
    updatePipelines();
  }

  return state;
}


void HbaoPass::updatePipelines()
{
  vkDestroyPipeline(m_device, m_pipelines.applyRaw, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.applyBlurred, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.calc, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.deinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.viewNormal, nullptr);

  VkShaderModuleCreateInfo    shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  VkComputePipelineCreateInfo info       = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.layout                            = m_pipelineLayout;
  info.stage                             = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  info.stage.stage                       = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.pName                       = "main";
  info.stage.pNext                       = &shaderInfo;

  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.applyRaw);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.applyRaw);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.applyBlurred);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.applyBlurred);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.deinterleave);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.deinterleave);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.calc);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.calc);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.viewNormal);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.viewNormal);

  NVVK_DBG_NAME(m_pipelines.applyBlurred);
  NVVK_DBG_NAME(m_pipelines.applyRaw);
  NVVK_DBG_NAME(m_pipelines.deinterleave);
  NVVK_DBG_NAME(m_pipelines.calc);
  NVVK_DBG_NAME(m_pipelines.viewNormal);
}

void HbaoPass::deinit()
{
  m_allocator->destroyBuffer(m_ubo);
  m_samplerPool->releaseSampler(m_linearSampler);
  m_samplerPool->releaseSampler(m_nearestSampler);

  vkDestroyPipeline(m_device, m_pipelines.applyBlurred, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.applyRaw, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.calc, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.deinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.viewNormal, nullptr);

  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

  m_dsetPack.deinit();

  memset(this, 0, sizeof(HbaoPass));
}


bool HbaoPass::initFrame(Frame& frame, const FrameConfig& config, VkCommandBuffer cmd)
{
  deinitFrame(frame);

  if(m_slotsUsed == ~(0ULL))
    return false;

  for(uint32_t i = 0; i < 64; i++)
  {
    uint64_t bitMask = uint64_t(1) << i;
    if(!(m_slotsUsed & bitMask))
    {
      frame.slot = i;
      m_slotsUsed |= bitMask;
      break;
    }
  }

  frame.config          = config;
  FrameImages& textures = frame.images;

  uint32_t width  = config.targetWidth;
  uint32_t height = config.targetHeight;
  frame.width     = width;
  frame.height    = height;

  VkImageCreateInfo     info     = DEFAULT_VkImageCreateInfo;
  VkImageViewCreateInfo viewInfo = DEFAULT_VkImageViewCreateInfo;

  info.extent.width  = width;
  info.extent.height = height;
  info.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  info.format = viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  m_allocator->createImage(frame.images.viewNormal, info, viewInfo);
  frame.images.viewNormal.descriptor.sampler = m_nearestSampler;

  uint32_t quarterWidth  = ((width + 3) / 4);
  uint32_t quarterHeight = ((height + 3) / 4);

  info.extent.width  = quarterWidth;
  info.extent.height = quarterHeight;
  info.arrayLayers   = RANDOM_ELEMENTS;

  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

  info.format = viewInfo.format = VK_FORMAT_R8_UNORM;
  m_allocator->createImage(frame.images.resultArray, info, viewInfo);
  frame.images.resultArray.descriptor.sampler = m_nearestSampler;

  info.format = viewInfo.format = VK_FORMAT_R32_SFLOAT;
  m_allocator->createImage(frame.images.linearDepthArray, info, viewInfo);
  frame.images.linearDepthArray.descriptor.sampler = m_nearestSampler;

  nvvk::BarrierContainer barrierContainer;
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.viewNormal, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.resultArray, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.linearDepthArray, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.cmdPipelineBarrier(cmd, 0);


  nvvk::WriteSetContainer writes;
  VkDescriptorBufferInfo  uboInfo = m_uboInfo;
  uboInfo.offset                  = m_uboInfo.range * frame.slot;

  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_UBO, frame.slot), uboInfo);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_DEPTH, frame.slot), config.sourceDepth);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_VIEWNORMAL, frame.slot), frame.images.viewNormal);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_DEPTHARRAY, frame.slot), frame.images.linearDepthArray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_RESULTARRAY, frame.slot), frame.images.resultArray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_VIEWNORMAL, frame.slot), frame.images.viewNormal);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_DEPTHARRAY, frame.slot), frame.images.linearDepthArray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_RESULTARRAY, frame.slot), frame.images.resultArray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_OUT, frame.slot), config.targetColor);

  vkUpdateDescriptorSets(m_device, uint32_t(writes.size()), writes.data(), 0, nullptr);

  VkImage hbaoResultArray = frame.images.resultArray.image;
  VkImage hbaoDepthArray  = frame.images.linearDepthArray.image;
  VkImage hbaoViewNormal  = frame.images.viewNormal.image;
  NVVK_DBG_NAME(hbaoResultArray);
  NVVK_DBG_NAME(hbaoDepthArray);
  NVVK_DBG_NAME(hbaoViewNormal);

  return true;
}

void HbaoPass::deinitFrame(Frame& frame)
{
  if(frame.slot != ~0u)
  {
    m_slotsUsed &= ~(1ull << frame.slot);
    m_allocator->destroyImage(frame.images.resultArray);
    m_allocator->destroyImage(frame.images.linearDepthArray);
    m_allocator->destroyImage(frame.images.viewNormal);
  }

  frame = Frame();
}

void HbaoPass::updateUbo(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const
{
  const View& view   = settings.view;
  uint32_t    width  = frame.width;
  uint32_t    height = frame.height;

  uint32_t quarterWidth  = ((width + 3) / 4);
  uint32_t quarterHeight = ((height + 3) / 4);

  glsl::NVHBAOData hbaoData = {};

  glm::vec2 viewportSizeF = glm::vec2(width, height);

  hbaoData.view.matClipToView   = glm::inverse(view.projectionMatrix);
  hbaoData.view.matWorldToView  = view.viewMatrix;
  hbaoData.view.pixelOffset     = glm::vec2(0, 0);
  hbaoData.view.viewportSize    = glm::ivec2(width, height);
  hbaoData.view.invViewportSize = glm::vec2(1.0f) / viewportSizeF;

  glm::vec2 clipToWindowScale = viewportSizeF * glm::vec2(0.5f, 0.5f);
  glm::vec2 clipToWindowBias  = viewportSizeF * 0.5f;

  hbaoData.view.windowToClipScale = 1.0f / clipToWindowScale;
  hbaoData.view.windowToClipBias  = -clipToWindowBias * hbaoData.view.windowToClipScale;

  hbaoData.amount                  = settings.intensity;
  hbaoData.surfaceBias             = settings.bias;
  hbaoData.powerExponent           = settings.powerExponent;
  hbaoData.radiusWorld             = settings.radius;
  hbaoData.invQuantizedGbufferSize = glm::vec2(1.0) / (glm::vec2(quarterWidth * 4, quarterHeight * 4));
  hbaoData.invBackgroundViewDepth  = 1.0f / (settings.backgroundViewDepth * view.farPlane);

  hbaoData.radiusToScreen = viewportSizeF.y * 0.5f * abs(view.projectionMatrix[1][1]);

  hbaoData.clipToView = {view.projectionMatrix[2][3] / view.projectionMatrix[0][0],
                         view.projectionMatrix[2][3] / view.projectionMatrix[1][1]};

  vkCmdUpdateBuffer(cmd, m_uboInfo.buffer, m_uboInfo.range * frame.slot, sizeof(hbaoData), &hbaoData);
}

void HbaoPass::cmdCompute(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const
{
  uint32_t width  = frame.width;
  uint32_t height = frame.height;

  uint32_t quarterWidth  = ((width + 3) / 4);
  uint32_t quarterHeight = ((height + 3) / 4);

  glm::uvec2 gridInput((width + 7) / 8, (height + 7) / 8);
  glm::uvec2 gridQuarter((quarterWidth + 7) / 8, (quarterHeight + 7) / 8);
  glm::uvec2 gridBlur((width + 15) / 16, (width + 15) / 16);

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  updateUbo(cmd, frame, settings);
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(frame.slot), 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.deinterleave);
  vkCmdDispatch(cmd, gridInput.x, gridInput.y, 1);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.viewNormal);
  vkCmdDispatch(cmd, gridInput.x, gridInput.y, 1);

  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.calc);
  vkCmdDispatch(cmd, gridQuarter.x, gridQuarter.y, RANDOM_ELEMENTS);

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  if(settings.blur)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.applyBlurred);
    vkCmdDispatch(cmd, gridBlur.x, gridBlur.y, 1);
  }
  else
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.applyRaw);
    vkCmdDispatch(cmd, gridInput.x, gridInput.y, 1);
  }
}
