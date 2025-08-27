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
    bindings.addBinding(NVHBAO_MAIN_TEX_LINDEPTH, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_VIEWNORMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_DEPTHARRAY, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_RESULTARRAY, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_RESULT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_TEX_BLUR, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_LINDEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_VIEWNORMAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_DEPTHARRAY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_RESULTARRAY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_RESULT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bindings.addBinding(NVHBAO_MAIN_IMG_BLUR, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
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

  std::mt19937 rng;
  float        numDir = NVHBAO_NUM_DIRECTIONS;

  for(int i = 0; i < RANDOM_ELEMENTS; i++)
  {
    float Rand1 = static_cast<float>(rng()) / 4294967296.0f;
    float Rand2 = static_cast<float>(rng()) / 4294967296.0f;

    // Use random rotation angles in [0,2PI/NUM_DIRECTIONS)
    float Angle       = glm::two_pi<float>() * Rand1 / numDir;
    m_hbaoRandom[i].x = cosf(Angle);
    m_hbaoRandom[i].y = sinf(Angle);
    m_hbaoRandom[i].z = Rand2;
    m_hbaoRandom[i].w = 0;
  }

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
  bool state = true;
  state = compileShader(m_glslCompiler, m_shaders.depth_linearize, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_depthlinearize.comp.glsl")
          && state;
  state = compileShader(m_glslCompiler, m_shaders.viewnormal, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_viewnormal.comp.glsl") && state;
  state = compileShader(m_glslCompiler, m_shaders.blur, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_blur.comp.glsl") && state;
  state = compileShader(m_glslCompiler, m_shaders.blur_apply, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_blur_apply.comp.glsl") && state;
  state = compileShader(m_glslCompiler, m_shaders.calc, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_calc.comp.glsl") && state;
  state = compileShader(m_glslCompiler, m_shaders.deinterleave, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_deinterleave.comp.glsl") && state;
  state = compileShader(m_glslCompiler, m_shaders.reinterleave, VK_SHADER_STAGE_COMPUTE_BIT, "hbao_reinterleave.comp.glsl") && state;

  if(state)
  {
    updatePipelines();
  }

  return state;
}


void HbaoPass::updatePipelines()
{
  vkDestroyPipeline(m_device, m_pipelines.blur, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.blur_apply, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.calc, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.deinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.reinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.viewnormal, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.depth_linearize, nullptr);

  VkShaderModuleCreateInfo    shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  VkComputePipelineCreateInfo info       = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.layout                            = m_pipelineLayout;
  info.stage                             = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  info.stage.stage                       = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.pName                       = "main";
  info.stage.pNext                       = &shaderInfo;

  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.blur);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.blur);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.blur_apply);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.blur_apply);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.deinterleave);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.deinterleave);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.reinterleave);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.reinterleave);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.viewnormal);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.viewnormal);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.depth_linearize);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.depth_linearize);
  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.calc);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.calc);

  NVVK_DBG_NAME(m_pipelines.blur);
  NVVK_DBG_NAME(m_pipelines.blur_apply);
  NVVK_DBG_NAME(m_pipelines.deinterleave);
  NVVK_DBG_NAME(m_pipelines.reinterleave);
  NVVK_DBG_NAME(m_pipelines.viewnormal);
  NVVK_DBG_NAME(m_pipelines.depth_linearize);
  NVVK_DBG_NAME(m_pipelines.calc);
}

void HbaoPass::deinit()
{
  m_allocator->destroyBuffer(m_ubo);
  m_samplerPool->releaseSampler(m_linearSampler);
  m_samplerPool->releaseSampler(m_nearestSampler);

  vkDestroyPipeline(m_device, m_pipelines.blur, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.blur_apply, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.calc, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.deinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.reinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.viewnormal, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.depth_linearize, nullptr);

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

  frame.config        = config;
  FrameIMGs& textures = frame.images;

  uint32_t width  = config.targetWidth;
  uint32_t height = config.targetHeight;
  frame.width     = width;
  frame.height    = height;

  VkImageCreateInfo     info     = DEFAULT_VkImageCreateInfo;
  VkImageViewCreateInfo viewInfo = DEFAULT_VkImageViewCreateInfo;

  info.extent.width  = width;
  info.extent.height = height;
  info.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  info.format = viewInfo.format = VK_FORMAT_R32_SFLOAT;
  m_allocator->createImage(frame.images.depthlinear, info, viewInfo);
  frame.images.depthlinear.descriptor.sampler = m_nearestSampler;

  info.format = viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  m_allocator->createImage(frame.images.viewnormal, info, viewInfo);
  frame.images.viewnormal.descriptor.sampler = m_nearestSampler;

  info.format = viewInfo.format = VK_FORMAT_R16G16_SFLOAT;
  m_allocator->createImage(frame.images.result, info, viewInfo);
  frame.images.result.descriptor.sampler = m_linearSampler;

  info.format = viewInfo.format = VK_FORMAT_R16G16_SFLOAT;
  m_allocator->createImage(frame.images.blur, info, viewInfo);
  frame.images.blur.descriptor.sampler = m_linearSampler;

  uint32_t quarterWidth  = ((width + 3) / 4);
  uint32_t quarterHeight = ((height + 3) / 4);

  info.extent.width  = quarterWidth;
  info.extent.height = quarterHeight;
  info.arrayLayers   = RANDOM_ELEMENTS;

  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

  info.format = viewInfo.format = VK_FORMAT_R16G16_SFLOAT;
  m_allocator->createImage(frame.images.resultarray, info, viewInfo);
  frame.images.resultarray.descriptor.sampler = m_nearestSampler;

  info.format = viewInfo.format = VK_FORMAT_R32_SFLOAT;
  m_allocator->createImage(frame.images.deptharray, info, viewInfo);
  frame.images.deptharray.descriptor.sampler = m_nearestSampler;

  nvvk::BarrierContainer barrierContainer;
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.depthlinear, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.viewnormal, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.result, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.blur, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.resultarray, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.appendOptionalLayoutTransition(
      frame.images.deptharray, nvvk::makeImageMemoryBarrier({nullptr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL}));
  barrierContainer.cmdPipelineBarrier(cmd, 0);


  nvvk::WriteSetContainer writes;
  VkDescriptorBufferInfo  uboInfo = m_uboInfo;
  uboInfo.offset                  = m_uboInfo.range * frame.slot;

  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_UBO, frame.slot), uboInfo);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_DEPTH, frame.slot), config.sourceDepth);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_LINDEPTH, frame.slot), frame.images.depthlinear);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_VIEWNORMAL, frame.slot), frame.images.viewnormal);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_DEPTHARRAY, frame.slot), frame.images.deptharray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_RESULTARRAY, frame.slot), frame.images.resultarray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_RESULT, frame.slot), frame.images.result);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_TEX_BLUR, frame.slot), frame.images.blur);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_LINDEPTH, frame.slot), frame.images.depthlinear);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_VIEWNORMAL, frame.slot), frame.images.viewnormal);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_DEPTHARRAY, frame.slot), frame.images.deptharray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_RESULTARRAY, frame.slot), frame.images.resultarray);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_RESULT, frame.slot), frame.images.result);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_BLUR, frame.slot), frame.images.blur);
  writes.append(m_dsetPack.makeWrite(NVHBAO_MAIN_IMG_OUT, frame.slot), config.targetColor);

  vkUpdateDescriptorSets(m_device, uint32_t(writes.size()), writes.data(), 0, nullptr);

  VkImage hbaoBlur        = frame.images.blur.image;
  VkImage hbaoResult      = frame.images.result.image;
  VkImage hbaoResultArray = frame.images.resultarray.image;
  VkImage hbaoDepthArray  = frame.images.deptharray.image;
  VkImage hbaoDepthLin    = frame.images.depthlinear.image;
  VkImage hbaoViewNormal  = frame.images.viewnormal.image;
  NVVK_DBG_NAME(hbaoBlur);
  NVVK_DBG_NAME(hbaoResult);
  NVVK_DBG_NAME(hbaoResultArray);
  NVVK_DBG_NAME(hbaoDepthArray);
  NVVK_DBG_NAME(hbaoDepthLin);
  NVVK_DBG_NAME(hbaoViewNormal);

  return true;
}

void HbaoPass::deinitFrame(Frame& frame)
{
  if(frame.slot != ~0u)
  {
    m_slotsUsed &= ~(1ull << frame.slot);
    m_allocator->destroyImage(frame.images.blur);
    m_allocator->destroyImage(frame.images.result);
    m_allocator->destroyImage(frame.images.resultarray);
    m_allocator->destroyImage(frame.images.deptharray);
    m_allocator->destroyImage(frame.images.depthlinear);
    m_allocator->destroyImage(frame.images.viewnormal);
  }

  frame = Frame();
}

void HbaoPass::updateUbo(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const
{
  const View& view   = settings.view;
  uint32_t    width  = frame.width;
  uint32_t    height = frame.height;

  glsl::NVHBAOData hbaoData;

  // projection
  const float* P = glm::value_ptr(view.projectionMatrix);

  float projInfoPerspective[] = {
      2.0f / (P[4 * 0 + 0]),                  // (x) * (R - L)/N
      2.0f / (P[4 * 1 + 1]),                  // (y) * (T - B)/N
      -(1.0f - P[4 * 2 + 0]) / P[4 * 0 + 0],  // L/N
      -(1.0f + P[4 * 2 + 1]) / P[4 * 1 + 1],  // B/N
  };

  float projInfoOrtho[] = {
      2.0f / (P[4 * 0 + 0]),                  // ((x) * R - L)
      2.0f / (P[4 * 1 + 1]),                  // ((y) * T - B)
      -(1.0f + P[4 * 3 + 0]) / P[4 * 0 + 0],  // L
      -(1.0f - P[4 * 3 + 1]) / P[4 * 1 + 1],  // B
  };

  int useOrtho       = view.isOrtho ? 1 : 0;
  hbaoData.projOrtho = useOrtho;
  hbaoData.projInfo  = useOrtho ? glm::make_vec4(projInfoOrtho) : glm::make_vec4(projInfoPerspective);

  float projScale;
  if(useOrtho)
  {
    projScale = float(height) / (projInfoOrtho[1]);
  }
  else
  {
    projScale = float(height) / (view.halfFovyTan * 2.0f);
  }

  hbaoData.projReconstruct =
      glm::vec4(view.nearPlane * view.farPlane, view.nearPlane - view.farPlane, view.farPlane, view.isOrtho ? 0.0f : 1.0f);

  // radius
  float R                 = settings.radius * settings.unit2viewspace;
  hbaoData.R2             = R * R;
  hbaoData.NegInvR2       = -1.0f / hbaoData.R2;
  hbaoData.RadiusToScreen = R * 0.5f * projScale;

  // ao
  hbaoData.PowExponent  = std::max(settings.intensity, 0.0f);
  hbaoData.NDotVBias    = std::min(std::max(0.0f, settings.bias), 1.0f);
  hbaoData.AOMultiplier = 1.0f / (1.0f - hbaoData.NDotVBias);

  hbaoData.InvProjMatrix = glm::inverse(view.projectionMatrix);

  // resolution
  int quarterWidth  = ((width + 3) / 4);
  int quarterHeight = ((height + 3) / 4);

  hbaoData.InvQuarterResolution  = glm::vec2(1.0f / float(quarterWidth), 1.0f / float(quarterHeight));
  hbaoData.InvFullResolution     = glm::vec2(1.0f / float(width), 1.0f / float(height));
  hbaoData.SourceResolutionScale = glm::ivec2(frame.config.sourceWidthScale, frame.config.sourceHeightScale);
  hbaoData.FullResolution        = glm::ivec2(width, height);
  hbaoData.QuarterResolution     = glm::ivec2(quarterWidth, quarterHeight);

  for(int i = 0; i < RANDOM_ELEMENTS; i++)
  {
    hbaoData.float2Offsets[i] = glm::vec4(float(i % 4) + 0.5f, float(i / 4) + 0.5f, 0.0f, 0.0f);
    hbaoData.jitters[i]       = m_hbaoRandom[i];
  }

  vkCmdUpdateBuffer(cmd, m_uboInfo.buffer, m_uboInfo.range * frame.slot, sizeof(hbaoData), &hbaoData);
}

void HbaoPass::cmdCompute(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const
{
  // full res
  glsl::NVHBAOBlurPush blur;
  glsl::NVHBAOMainPush calc = {0};

  uint32_t width         = frame.width;
  uint32_t height        = frame.height;
  uint32_t quarterWidth  = ((width + 3) / 4);
  uint32_t quarterHeight = ((height + 3) / 4);

  glm::uvec2 gridFull((width + 7) / 8, (height + 7) / 8);
  glm::uvec2 gridQuarter((quarterWidth + 7) / 8, (quarterHeight + 7) / 8);

  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
  updateUbo(cmd, frame, settings);
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);

  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(calc), &calc);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(frame.slot), 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.depth_linearize);
  vkCmdDispatch(cmd, gridFull.x, gridFull.y, 1);

  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.viewnormal);
  vkCmdDispatch(cmd, gridFull.x, gridFull.y, 1);


  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

#if !NVHBAO_SKIP_INTERPASS
  // quarter
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.deinterleave);
  vkCmdDispatch(cmd, gridQuarter.x, gridQuarter.y, 1);


  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
#endif

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.calc);
  for(uint32_t i = 0; i < RANDOM_ELEMENTS; i++)
  {
    calc.layer = i;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(calc), &calc);
    vkCmdDispatch(cmd, gridQuarter.x, gridQuarter.y, 1);
  }


  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  // full res
#if !NVHBAO_SKIP_INTERPASS
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.reinterleave);
  vkCmdDispatch(cmd, gridFull.x, gridFull.y, 1);


  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
#endif

  blur.sharpness              = settings.blurSharpness / settings.unit2viewspace;
  blur.invResolutionDirection = glm::vec2(1.0f / float(frame.width), 0.0f);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.blur);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur), &blur);
  vkCmdDispatch(cmd, gridFull.x, gridFull.y, 1);

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.blur_apply);
  blur.invResolutionDirection = glm::vec2(0.0f, 1.0f / float(frame.height));
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur), &blur);
  vkCmdDispatch(cmd, gridFull.x, gridFull.y, 1);
}
