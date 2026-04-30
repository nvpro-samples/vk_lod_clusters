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

#include <condition_variable>
#include <inttypes.h>
#include <mutex>
#include <nvimageformats/texture_formats.h>
#include <nvimageformats/nv_dds.h>
#include <nvimageformats/nv_ktx.h>
#include <nvutils/parallel_work.hpp>
#include <nvvk/default_structs.hpp>
#include "scene_textures.hpp"

namespace lodclusters {

namespace {

// CPU-side result of decoding an image from raw bytes (disk, buffer, or embedded).
// Used by ImageLoader; SceneVk copies this into SceneImage and then creates the Vulkan image.
struct LoadedImageData
{
  VkFormat                       format{VK_FORMAT_UNDEFINED};
  VkExtent3D                     size{0, 0, 0};
  std::vector<std::vector<char>> mipData{};
  VkComponentMapping             componentMapping{};
};

VkComponentSwizzle ktxSwizzleToVk(nv_ktx::KTX_SWIZZLE swizzle)
{
  switch(swizzle)
  {
    case nv_ktx::KTX_SWIZZLE::ZERO:
      return VK_COMPONENT_SWIZZLE_ZERO;
    case nv_ktx::KTX_SWIZZLE::ONE:
      return VK_COMPONENT_SWIZZLE_ONE;
    case nv_ktx::KTX_SWIZZLE::R:
      return VK_COMPONENT_SWIZZLE_R;
    case nv_ktx::KTX_SWIZZLE::G:
      return VK_COMPONENT_SWIZZLE_G;
    case nv_ktx::KTX_SWIZZLE::B:
      return VK_COMPONENT_SWIZZLE_B;
    case nv_ktx::KTX_SWIZZLE::A:
      return VK_COMPONENT_SWIZZLE_A;
    default:
      return VK_COMPONENT_SWIZZLE_IDENTITY;
  }
}

VkComponentMapping ktxSwizzleToVkComponentMapping(const std::array<nv_ktx::KTX_SWIZZLE, 4>& swizzle)
{
  return {ktxSwizzleToVk(swizzle[0]), ktxSwizzleToVk(swizzle[1]), ktxSwizzleToVk(swizzle[2]), ktxSwizzleToVk(swizzle[3])};
}

bool loadDds(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  const char ddsIdentifier[4] = {'D', 'D', 'S', ' '};
  if(byteLength < sizeof(ddsIdentifier) || memcmp(data, ddsIdentifier, sizeof(ddsIdentifier)) != 0)
    return false;

  nv_dds::Image        ddsImage{};
  nv_dds::ReadSettings settings{};
  const nv_dds::ErrorWithText readResult = ddsImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, settings);
  if(readResult.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_dds: %s\n", imageIDForLog, readResult.value().c_str());
    return false;
  }

  out.size.width  = ddsImage.getWidth(0);
  out.size.height = ddsImage.getHeight(0);
  out.size.depth  = 1;
  if(ddsImage.getDepth(0) > 1)
  {
    LOGW("This DDS image had a depth of %u, but loadFromMemory() cannot handle volume textures.\n", ddsImage.getDepth(0));
    return false;
  }
  if(ddsImage.getNumFaces() > 1)
  {
    LOGW("This DDS image had %u faces, but loadFromMemory() cannot handle cubemaps.\n", ddsImage.getNumFaces());
    return false;
  }
  if(ddsImage.getNumLayers() > 1)
  {
    LOGW("This DDS image had %u array elements, but loadFromMemory() cannot handle array textures.\n", ddsImage.getNumLayers());
    return false;
  }
  out.format = texture_formats::dxgiToVulkan(ddsImage.dxgiFormat);
  out.format = texture_formats::tryForceVkFormatTransferFunction(out.format, srgb);
  if(out.format == VK_FORMAT_UNDEFINED)
  {
    LOGW("Could not determine a VkFormat for DXGI format %u (%s).\n", ddsImage.dxgiFormat,
         texture_formats::getDXGIFormatName(ddsImage.dxgiFormat));
    return false;
  }

  for(uint32_t i = 0; i < ddsImage.getNumMips(); i++)
  {
    std::vector<char>& mip = ddsImage.subresource(i, 0, 0).data;
    out.mipData.push_back(std::move(mip));
  }
  return true;
}

bool loadKtx(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  const uint8_t ktxIdentifier[5] = {0xAB, 0x4B, 0x54, 0x58, 0x20};  // Common for KTX1 + KTX2
  if(byteLength < sizeof(ktxIdentifier) || memcmp(data, ktxIdentifier, sizeof(ktxIdentifier)) != 0)
    return false;

  nv_ktx::KTXImage           ktxImage;
  const nv_ktx::ReadSettings ktxReadSettings;
  const nv_ktx::ErrorWithText maybeError = ktxImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, ktxReadSettings);
  if(maybeError.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_ktx: %s\n", imageIDForLog, maybeError->c_str());
    return false;
  }

  out.size.width  = ktxImage.mip_0_width;
  out.size.height = ktxImage.mip_0_height;
  out.size.depth  = 1;
  if(ktxImage.mip_0_depth > 1)
  {
    LOGW("KTX image %" PRIu64 " had a depth of %u, but loadFromMemory() cannot handle volume textures.\n",
         imageIDForLog, ktxImage.mip_0_depth);
    return false;
  }
  if(ktxImage.num_faces > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u faces, but loadFromMemory() cannot handle cubemaps.\n", imageIDForLog, ktxImage.num_faces);
    return false;
  }
  if(ktxImage.num_layers_possibly_0 > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u array elements, but loadFromMemory() cannot handle array textures.\n",
         imageIDForLog, ktxImage.num_layers_possibly_0);
    return false;
  }
  out.format           = texture_formats::tryForceVkFormatTransferFunction(ktxImage.format, srgb);
  out.componentMapping = ktxSwizzleToVkComponentMapping(ktxImage.swizzle);

  for(uint32_t i = 0; i < ktxImage.num_mips; i++)
  {
    std::vector<char>& mip = ktxImage.subresource(i, 0, 0);
    out.mipData.push_back(std::move(mip));
  }
  return true;
}

bool loadFromMemory(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  out = LoadedImageData{};

  if(data == nullptr || byteLength == 0)
    return false;

  if(loadDds(out, data, byteLength, srgb, imageIDForLog))
    return true;
  if(loadKtx(out, data, byteLength, srgb, imageIDForLog))
    return true;
  //  if(loadStb(out, data, byteLength, srgb, imageIDForLog))
  //    return true;

  return false;
}

struct ParallelBatchUploader
{
  Resources&              res;
  std::mutex              uploaderMutex;
  std::mutex              pendingMutex;
  std::condition_variable pendingCv;
  int                     pending = 0;
  std::atomic_bool        failed  = false;
  std::atomic_bool        missing = false;
  VkCommandBuffer         cmd     = {};

  ParallelBatchUploader(Resources& res_)
      : res(res_)
  {
    res.m_uploader.setEnableLayoutBarriers(true);
  }

  ~ParallelBatchUploader() { res.m_uploader.setEnableLayoutBarriers(false); }

  void beginPending()
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    ++pending;
  }

  void endPending()
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    --pending;
    if(pending == 0)
      pendingCv.notify_all();
  }

  void flush()
  {
    if(res.m_uploader.isAppendedEmpty())
      return;

    std::unique_lock<std::mutex> lock(pendingMutex);
    pendingCv.wait(lock, [this] { return pending == 0; });
    lock.unlock();

    VkCommandBuffer cmd = res.createTempCmdBuffer();
    res.m_uploader.cmdUploadAppended(cmd);
    res.tempSyncSubmit(cmd);
    res.m_uploader.releaseStaging();
  }

  void uploadImage(const std::string& fileName, bool sRGB, uint64_t imageID, nvvk::Image& image)
  {
    nvutils::FileReadMapping readMapping;
    if(!readMapping.open(fileName))
    {
      LOGW("image loader: file not found %s\n", fileName.c_str());
      missing = true;
      return;
    }

    LoadedImageData loaded;
    if(!loadFromMemory(loaded, readMapping.data(), readMapping.size(), sRGB, imageID))
    {
      LOGW("image loader: failed to load %s\n", fileName.c_str());
      failed = true;
      return;
    }

    std::array<void*, 32>  mipMappings = {};
    std::array<size_t, 32> mipSizes    = {};

    size_t   totalSize = 0;
    uint32_t mipLevels = uint32_t(loaded.mipData.size());

    for(uint32_t m = 0; m < mipLevels; m++)
    {
      mipSizes[m] = loaded.mipData[m].size();
      totalSize += mipSizes[m];
    }

    VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
    imageCreateInfo.extent            = loaded.size;
    imageCreateInfo.mipLevels         = mipLevels;
    imageCreateInfo.format            = loaded.format;
    imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImageViewCreateInfo imageViewCreateInfo = DEFAULT_VkImageViewCreateInfo;
    imageViewCreateInfo.components            = loaded.componentMapping;
    imageViewCreateInfo.format                = loaded.format;

    if(VK_SUCCESS != NVVK_FAIL_REPORT(res.m_allocator.createImage(image, imageCreateInfo, imageViewCreateInfo)))
    {
      failed = true;
      return;
    }

    {
      std::lock_guard lock(uploaderMutex);
      if(res.m_uploader.checkAppendedSize(128 * 1024 * 1024, totalSize))
      {
        flush();
      }

      beginPending();

      VkExtent3D extent = loaded.size;
      for(uint32_t m = 0; m < mipLevels; m++)
      {
        VkImageSubresourceLayers subResource = {};
        subResource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
        subResource.mipLevel                 = m;
        subResource.baseArrayLayer           = 0;
        subResource.layerCount               = 1;

        VkOffset3D offset = {0, 0, 0};

        // bit of a hack to ensure the barriers are done on first and last mip.
        // hence manipulating the imageLayout

        if(VK_SUCCESS
           != NVVK_FAIL_REPORT(res.m_uploader.appendImageSubMapping(
               image, offset, extent, subResource, mipSizes[m], mipMappings[m],
               m == mipLevels - 1 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)))
        {
          failed = true;
          endPending();
          return;
        }

        if(m < mipLevels - 1)
        {
          image.descriptor.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        }

        extent.width  = (extent.width + 1) / 2;
        extent.height = (extent.height + 1) / 2;
        extent.depth  = (extent.depth + 1) / 2;
      }
    }

    for(size_t m = 0; m < loaded.mipData.size(); m++)
    {
      memcpy(mipMappings[m], loaded.mipData[m].data(), mipSizes[m]);
    }

    endPending();
  }
};

}  // namespace


bool SceneTextures::init(Resources* res, const Scene& scene, const SceneTexturesConfig& config)
{
  m_res = res;

  uint32_t imageCount = uint32_t(scene.m_imageFileNames.size());

  m_images.resize(imageCount, {});

  {
    VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
    imageCreateInfo.extent            = {1, 1, 1};
    imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkResult result =
        NVVK_FAIL_REPORT(res->m_allocator.createImage(m_defaultImage, imageCreateInfo, DEFAULT_VkImageViewCreateInfo));
    if(result != VK_SUCCESS)
    {
      LOGW("Failed to create default image\n");
      return false;
    }
  }

  // multi-thread the loading of scene->m_imageFileNames
  // insert a mutex to flush the staging space and one to acquire space.

  {
    ParallelBatchUploader uploader = ParallelBatchUploader(*res);

    // Upload m_defaultImage with a simple RGBA of 0xFFFFFFFF value using the existing uploader class
    {
      uint32_t defaultPixel = 0xFFFFFFFFu;
      res->m_uploader.appendImage(m_defaultImage, sizeof(uint32_t), &defaultPixel, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if(imageCount)
    {
      nvutils::parallel_batches<1>(
          imageCount, [&](uint64_t idx) { uploader.uploadImage(scene.m_imageFileNames[idx], true, idx, m_images[idx]); },
          config.maxThreads);
    }

    uploader.flush();

    if(uploader.failed)
      return false;

    if(uploader.missing)
    {
      // use default image for missing images
      for(uint32_t i = 0; i < imageCount; i++)
      {
        if(!m_images[i].image)
        {
          m_images[i] = m_defaultImage;
        }
      }
    }
  }

  std::vector<VkSampler> immutableSamplers;
  if(config.immutableSampler)
  {
    immutableSamplers.resize(std::max(imageCount, 1u), config.immutableSampler);
  }

  // always ensure 1 descriptor, even if empty.
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(0, config.immutableSampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                      std::max(imageCount, 1u), VK_SHADER_STAGE_ALL,
                      config.immutableSampler ? immutableSamplers.data() : nullptr, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

  m_dsetPack.init(bindings, res->m_device, 1);

  nvvk::WriteSetContainer writeSet;
  if(imageCount > 0)
  {
    for(uint32_t i = 0; i < imageCount; i++)
    {
      if(m_images[i].image)
      {
        writeSet.append(m_dsetPack.makeWrite(0, 0, i, 1), m_images[i]);
      }
    }
  }
  else
  {
    writeSet.append(m_dsetPack.makeWrite(0, 0, 0, 1), m_defaultImage);
  }

  vkUpdateDescriptorSets(res->m_device, writeSet.size(), writeSet.data(), 0, nullptr);

  return true;
}

void SceneTextures::deinit()
{
  for(auto& image : m_images)
  {
    if(image.image != m_defaultImage.image)
    {
      m_res->m_allocator.destroyImage(image);
    }
  }
  m_res->m_allocator.destroyImage(m_defaultImage);
  m_images.clear();
  m_dsetPack.deinit();
}

}  // namespace lodclusters
