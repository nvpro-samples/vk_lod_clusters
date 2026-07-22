/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thread>
#include "scene.hpp"
#include "resources.hpp"

namespace lodclusters {

struct SceneTexturesConfig
{
  uint32_t maxThreads = (std::thread::hardware_concurrency() + 3) / 4;
  // 0 means no budget
  uint32_t  maxBudgetMiB     = 4096;
  VkSampler immutableSampler = {};
};

// For now there is no streaming and
// all textures are loaded in advance, optionally limited by maxBudgetMiB.
// Only supports ktx2 and dds.
// The DescriptorPack is safe to use even if the scene had no textures,
// a single texture is always available. It's a single descriptor set available to all
// shader stages with an array of either VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE or
// VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER (`config.immutableSampler` != nullptr).
class SceneTextures
{
public:
  bool init(Resources* res, const Scene& scene, const SceneTexturesConfig& config);
  void deinit();

  const nvvk::DescriptorPack& getDsetPack() const { return m_dsetPack; }

  bool   hasTextures() const { return !m_images.empty(); }
  size_t getTextureMemBytes() const { return m_textureMemBytes; }

private:
  Resources*               m_res             = nullptr;
  size_t                   m_textureMemBytes = 0;
  nvvk::Image              m_defaultWhiteImage{};
  nvvk::Image              m_defaultBlackImage{};
  nvvk::Image              m_defaultNormalImage{};
  std::vector<nvvk::Image> m_images;
  nvvk::DescriptorPack     m_dsetPack;
};


}  // namespace lodclusters
