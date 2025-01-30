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

#include <random>
#include <algorithm>

#include <nvvk/pipeline_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "hbao_pass.hpp"
#include "shaders/hbao.h"

#define DEBUGUTIL_SET_NAME(var) debugUtil.setObjectName(var, #var)
#define DEBUGUTIL_SET_NAMESTR(var, name) debugUtil.setObjectName(var, name)

void HbaoPass::init(VkDevice device, nvvk::ResourceAllocator* allocator, nvvk::ShaderModuleManager* shaderManager, const Config& config)
{
  nvvk::DebugUtil debugUtil(device);

  m_device        = device;
  m_allocator     = allocator;
  m_shaderManager = shaderManager;
  m_slots.init(config.maxFrames);

  {
    VkSamplerCreateInfo info = nvvk::makeSamplerCreateInfo();
    m_linearSampler          = m_allocator->acquireSampler(info);
  }

  // shaders
  {
    // clang-format off
        m_shaders.depth_linearize   = shaderManager->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "hbao_depthlinearize.comp.glsl");
        m_shaders.viewnormal        = shaderManager->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "hbao_viewnormal.comp.glsl");
        m_shaders.blur              = shaderManager->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "hbao_blur.comp.glsl");
        m_shaders.blur_apply        = shaderManager->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "hbao_blur_apply.comp.glsl");
        m_shaders.calc              = shaderManager->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "hbao_calc.comp.glsl");
        m_shaders.deinterleave      = shaderManager->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "hbao_deinterleave.comp.glsl");
        m_shaders.reinterleave      = shaderManager->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "hbao_reinterleave.comp.glsl");
    // clang-format on
  }

  // descriptor sets
  {
    m_setup.init(device);
    m_setup.addBinding(NVHBAO_MAIN_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_TEX_DEPTH, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, &m_linearSampler);
    m_setup.addBinding(NVHBAO_MAIN_TEX_LINDEPTH, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_TEX_VIEWNORMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_TEX_DEPTHARRAY, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_TEX_RESULTARRAY, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_TEX_RESULT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_TEX_BLUR, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_IMG_LINDEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_IMG_VIEWNORMAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_IMG_DEPTHARRAY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_IMG_RESULTARRAY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_IMG_RESULT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_IMG_BLUR, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.addBinding(NVHBAO_MAIN_IMG_OUT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setup.initLayout();

    VkPushConstantRange push;
    push.offset     = 0;
    push.size       = 16;
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    m_setup.initPipeLayout(1, &push);
    m_setup.initPool(config.maxFrames);
  }

  // pipelines
  updatePipelines();

  // ubo
  m_uboInfo.offset = 0;
  m_uboInfo.range  = (sizeof(glsl::NVHBAOData) + 255) & ~255;
  m_ubo            = allocator->createBuffer(m_uboInfo.range * config.maxFrames, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  m_uboInfo.buffer = m_ubo.buffer;
  DEBUGUTIL_SET_NAMESTR(m_ubo.buffer, "hbaoUbo");

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
}

void HbaoPass::reloadShaders()
{
  m_shaderManager->reloadModule(m_shaders.blur);
  m_shaderManager->reloadModule(m_shaders.blur_apply);
  m_shaderManager->reloadModule(m_shaders.calc);
  m_shaderManager->reloadModule(m_shaders.deinterleave);
  m_shaderManager->reloadModule(m_shaders.reinterleave);
  m_shaderManager->reloadModule(m_shaders.viewnormal);
  m_shaderManager->reloadModule(m_shaders.depth_linearize);
  updatePipelines();
}


void HbaoPass::updatePipelines()
{
  nvvk::DebugUtil debugUtil(m_device);

  vkDestroyPipeline(m_device, m_pipelines.blur, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.blur_apply, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.calc, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.deinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.reinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.viewnormal, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.depth_linearize, nullptr);

  VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.layout                      = m_setup.getPipeLayout();
  info.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  info.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.pName                 = "main";

  info.stage.module = m_shaderManager->get(m_shaders.blur);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.blur);
  info.stage.module = m_shaderManager->get(m_shaders.blur_apply);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.blur_apply);
  info.stage.module = m_shaderManager->get(m_shaders.deinterleave);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.deinterleave);
  info.stage.module = m_shaderManager->get(m_shaders.reinterleave);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.reinterleave);
  info.stage.module = m_shaderManager->get(m_shaders.viewnormal);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.viewnormal);
  info.stage.module = m_shaderManager->get(m_shaders.depth_linearize);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.depth_linearize);
  info.stage.module = m_shaderManager->get(m_shaders.calc);
  vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &m_pipelines.calc);

  Pipelines hbao = m_pipelines;
  DEBUGUTIL_SET_NAME(hbao.blur);
  DEBUGUTIL_SET_NAME(hbao.blur_apply);
  DEBUGUTIL_SET_NAME(hbao.deinterleave);
  DEBUGUTIL_SET_NAME(hbao.reinterleave);
  DEBUGUTIL_SET_NAME(hbao.viewnormal);
  DEBUGUTIL_SET_NAME(hbao.depth_linearize);
  DEBUGUTIL_SET_NAME(hbao.calc);
}

void HbaoPass::deinit()
{
  m_allocator->destroy(m_ubo);
  m_allocator->releaseSampler(m_linearSampler);

  vkDestroyPipeline(m_device, m_pipelines.blur, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.blur_apply, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.calc, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.deinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.reinterleave, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.viewnormal, nullptr);
  vkDestroyPipeline(m_device, m_pipelines.depth_linearize, nullptr);

  m_shaderManager->destroyShaderModule(m_shaders.blur);
  m_shaderManager->destroyShaderModule(m_shaders.blur_apply);
  m_shaderManager->destroyShaderModule(m_shaders.calc);
  m_shaderManager->destroyShaderModule(m_shaders.deinterleave);
  m_shaderManager->destroyShaderModule(m_shaders.reinterleave);
  m_shaderManager->destroyShaderModule(m_shaders.viewnormal);
  m_shaderManager->destroyShaderModule(m_shaders.depth_linearize);

  m_setup.deinit();

  memset(this, 0, sizeof(HbaoPass));
}


bool HbaoPass::initFrame(Frame& frame, const FrameConfig& config, VkCommandBuffer cmd)
{
  nvvk::DebugUtil debugUtil(m_device);

  deinitFrame(frame);

  if(!m_slots.createID(frame.slot))
    return false;

  frame.config        = config;
  FrameIMGs& textures = frame.images;

  uint32_t width  = config.targetWidth;
  uint32_t height = config.targetHeight;
  frame.width     = width;
  frame.height    = height;

  VkSamplerCreateInfo nearestInfo = nvvk::makeSamplerCreateInfo(VK_FILTER_NEAREST, VK_FILTER_NEAREST);
  VkSamplerCreateInfo linearInfo  = nvvk::makeSamplerCreateInfo(VK_FILTER_LINEAR, VK_FILTER_LINEAR);

  VkImageCreateInfo info = nvvk::makeImage2DCreateInfo({width, height});
  info.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  info.format              = VK_FORMAT_R32_SFLOAT;
  frame.images.depthlinear = m_allocator->createTexture(cmd, 0, nullptr, info, nearestInfo, VK_IMAGE_LAYOUT_GENERAL);
  info.format              = VK_FORMAT_R8G8B8A8_UNORM;
  frame.images.viewnormal  = m_allocator->createTexture(cmd, 0, nullptr, info, nearestInfo, VK_IMAGE_LAYOUT_GENERAL);
  info.format              = VK_FORMAT_R16G16_SFLOAT;
  frame.images.result      = m_allocator->createTexture(cmd, 0, nullptr, info, linearInfo, VK_IMAGE_LAYOUT_GENERAL);
  info.format              = VK_FORMAT_R16G16_SFLOAT;
  frame.images.blur        = m_allocator->createTexture(cmd, 0, nullptr, info, linearInfo, VK_IMAGE_LAYOUT_GENERAL);

  uint32_t quarterWidth  = ((width + 3) / 4);
  uint32_t quarterHeight = ((height + 3) / 4);

  info             = nvvk::makeImage2DCreateInfo({quarterWidth, quarterHeight});
  info.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  info.arrayLayers = RANDOM_ELEMENTS;

  info.format              = VK_FORMAT_R16G16_SFLOAT;
  frame.images.resultarray = m_allocator->createTexture(cmd, 0, nullptr, info, nearestInfo, VK_IMAGE_LAYOUT_GENERAL);
  info.format              = VK_FORMAT_R32_SFLOAT;
  frame.images.deptharray  = m_allocator->createTexture(cmd, 0, nullptr, info, nearestInfo, VK_IMAGE_LAYOUT_GENERAL);

  std::vector<VkWriteDescriptorSet> writes;
  VkDescriptorBufferInfo            uboInfo = m_uboInfo;
  uboInfo.offset                            = m_uboInfo.range * frame.slot;

  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_UBO, &uboInfo));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_TEX_DEPTH, &config.sourceDepth));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_TEX_LINDEPTH, &frame.images.depthlinear.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_TEX_VIEWNORMAL, &frame.images.viewnormal.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_TEX_DEPTHARRAY, &frame.images.deptharray.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_TEX_RESULTARRAY, &frame.images.resultarray.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_TEX_RESULT, &frame.images.result.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_TEX_BLUR, &frame.images.blur.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_IMG_LINDEPTH, &frame.images.depthlinear.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_IMG_VIEWNORMAL, &frame.images.viewnormal.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_IMG_DEPTHARRAY, &frame.images.deptharray.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_IMG_RESULTARRAY, &frame.images.resultarray.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_IMG_RESULT, &frame.images.result.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_IMG_BLUR, &frame.images.blur.descriptor));
  writes.push_back(m_setup.makeWrite(frame.slot, NVHBAO_MAIN_IMG_OUT, &config.targetColor));
  vkUpdateDescriptorSets(m_device, uint32_t(writes.size()), writes.data(), 0, nullptr);

  VkImage hbaoBlur        = frame.images.blur.image;
  VkImage hbaoResult      = frame.images.result.image;
  VkImage hbaoResultArray = frame.images.resultarray.image;
  VkImage hbaoDepthArray  = frame.images.deptharray.image;
  VkImage hbaoDepthLin    = frame.images.depthlinear.image;
  VkImage hbaoViewNormal  = frame.images.viewnormal.image;
  DEBUGUTIL_SET_NAME(hbaoBlur);
  DEBUGUTIL_SET_NAME(hbaoResult);
  DEBUGUTIL_SET_NAME(hbaoResultArray);
  DEBUGUTIL_SET_NAME(hbaoDepthArray);
  DEBUGUTIL_SET_NAME(hbaoDepthLin);
  DEBUGUTIL_SET_NAME(hbaoViewNormal);

  return true;
}

void HbaoPass::deinitFrame(Frame& frame)
{
  if(frame.slot != ~0u)
  {
    m_slots.destroyID(frame.slot);
    m_allocator->destroy(frame.images.blur);
    m_allocator->destroy(frame.images.result);
    m_allocator->destroy(frame.images.resultarray);
    m_allocator->destroy(frame.images.deptharray);
    m_allocator->destroy(frame.images.depthlinear);
    m_allocator->destroy(frame.images.viewnormal);
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

  vkCmdPushConstants(cmd, m_setup.getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(calc), &calc);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_setup.getPipeLayout(), 0, 1, m_setup.getSets(frame.slot), 0, nullptr);

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
    vkCmdPushConstants(cmd, m_setup.getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(calc), &calc);
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
  vkCmdPushConstants(cmd, m_setup.getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur), &blur);
  vkCmdDispatch(cmd, gridFull.x, gridFull.y, 1);

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.blur_apply);
  blur.invResolutionDirection = glm::vec2(0.0f, 1.0f / float(frame.height));
  vkCmdPushConstants(cmd, m_setup.getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur), &blur);
  vkCmdDispatch(cmd, gridFull.x, gridFull.y, 1);
}
