/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <thread>
#include "scene.hpp"
#include "resources.hpp"

namespace lodclusters {

struct SceneTexturesConfig
{
  uint32_t  maxThreads       = (std::thread::hardware_concurrency() + 3) / 4;
  VkSampler immutableSampler = {};
};

// For now there is no streaming and
// all textures are loaded fully in advance.
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

private:
  Resources*               m_res = nullptr;
  nvvk::Image              m_defaultImage{};
  std::vector<nvvk::Image> m_images;
  nvvk::DescriptorPack     m_dsetPack;
};


}  // namespace lodclusters
