/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <condition_variable>
#include <inttypes.h>
#include <mutex>
#include <span>
#include <nvimageformats/texture_formats.h>
#include <nvimageformats/nv_dds.h>
#include <nvimageformats/nv_ktx.h>
#include <nvutils/parallel_work.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/default_structs.hpp>
#include "scene_textures.hpp"

namespace lodclusters {

namespace {

struct MipSizes
{
  size_t   mipSizes[16]{};
  uint32_t mipCount = 0;
};

uint32_t mipDimension(uint32_t dim, uint32_t mipLevel)
{
  return std::max(1u, dim >> mipLevel);
}

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

bool isDdsFile(const char* mapped, size_t mappedSize)
{
  const char ddsIdentifier[4] = {'D', 'D', 'S', ' '};
  return mappedSize >= sizeof(ddsIdentifier) && memcmp(mapped, ddsIdentifier, sizeof(ddsIdentifier)) == 0;
}

bool isKtxFile(const char* mapped, size_t mappedSize)
{
  const uint8_t ktxIdentifier[5] = {0xAB, 0x4B, 0x54, 0x58, 0x20};
  return mappedSize >= sizeof(ktxIdentifier) && memcmp(mapped, ktxIdentifier, sizeof(ktxIdentifier)) == 0;
}

bool validateDdsImage(const nv_dds::Image& image, uint64_t imageIDForLog)
{
  if(image.mip0Depth > 1)
  {
    LOGW("Image %" PRIu64 " had a depth of %u, but scene textures cannot handle volume textures.\n", imageIDForLog,
         image.mip0Depth);
    return false;
  }
  if(image.getNumFaces() > 1)
  {
    LOGW("Image %" PRIu64 " had %u faces, but scene textures cannot handle cubemaps.\n", imageIDForLog, image.getNumFaces());
    return false;
  }
  if(image.getNumLayers() > 1)
  {
    LOGW("Image %" PRIu64 " had %u array elements, but scene textures cannot handle array textures.\n", imageIDForLog,
         image.getNumLayers());
    return false;
  }
  return true;
}

bool validateKtxImage(const nv_ktx::KTXImage& image, uint64_t imageIDForLog)
{
  if(image.mip_0_depth > 1)
  {
    LOGW("Image %" PRIu64 " had a depth of %u, but scene textures cannot handle volume textures.\n", imageIDForLog,
         image.mip_0_depth);
    return false;
  }
  if(image.num_faces > 1)
  {
    LOGW("Image %" PRIu64 " had %u faces, but scene textures cannot handle cubemaps.\n", imageIDForLog, image.num_faces);
    return false;
  }
  if(image.num_layers_possibly_0 > 1)
  {
    LOGW("Image %" PRIu64 " had %u array elements, but scene textures cannot handle array textures.\n", imageIDForLog,
         image.num_layers_possibly_0);
    return false;
  }
  return true;
}

bool mipSizesFromDds(MipSizes& out, const nv_dds::Image& image)
{
  out          = {};
  out.mipCount = std::min(image.getNumMips(), 16u);
  for(uint32_t mip = 0; mip < out.mipCount; mip++)
    out.mipSizes[mip] = image.getMipByteSizeSum(mip);
  return out.mipCount > 0;
}

bool mipSizesFromKtx(MipSizes& out, const nv_ktx::KTXImage& image)
{
  out          = {};
  out.mipCount = std::min(image.num_mips, 16u);
  for(uint32_t mip = 0; mip < out.mipCount; mip++)
    out.mipSizes[mip] = image.getMipByteSizeSum(mip);
  return out.mipCount > 0;
}

bool probeMipSizesFromMapped(MipSizes& out, const char* mapped, size_t mappedSize, uint64_t imageIDForLog)
{
  out = {};

  if(isDdsFile(mapped, mappedSize))
  {
    nv_dds::Image        image;
    nv_dds::ReadSettings readSettings{};
    readSettings.validateInputSize = false;
    if(!image.readHeaderFromMemory(mapped, mappedSize, readSettings).has_value() && validateDdsImage(image, imageIDForLog))
      return mipSizesFromDds(out, image);
  }

  if(isKtxFile(mapped, mappedSize))
  {
    nv_ktx::KTXImage     image;
    nv_ktx::ReadSettings readSettings{};
    readSettings.validate_input_size = false;
    if(!image.readHeaderFromMemory(mapped, mappedSize, readSettings).has_value() && validateKtxImage(image, imageIDForLog))
      return mipSizesFromKtx(out, image);
  }

  return false;
}

size_t mipTailMemoryUsage(const MipSizes& mipSizes, uint32_t baseMipLevel)
{
  size_t usage = 0;
  for(uint32_t mip = baseMipLevel; mip < mipSizes.mipCount; mip++)
    usage += mipSizes.mipSizes[mip];
  return usage;
}

uint32_t maxBaseMipLevel(const MipSizes& mipSizes)
{
  return mipSizes.mipCount > 0 ? mipSizes.mipCount - 1 : 0;
}

void computeLoadPlansForBudget(const std::vector<MipSizes>& textureMipSizes, std::vector<uint32_t>& baseMipLevels, size_t budgetBytes)
{
  baseMipLevels.assign(textureMipSizes.size(), 0);
  if(budgetBytes == 0 || textureMipSizes.empty())
    return;

  size_t totalUsed = 0;
  size_t minUsed   = 0;
  for(size_t textureIndex = 0; textureIndex < textureMipSizes.size(); textureIndex++)
  {
    const MipSizes& mipSizes = textureMipSizes[textureIndex];
    if(mipSizes.mipCount == 0)
      continue;

    totalUsed += mipTailMemoryUsage(mipSizes, 0);
    minUsed += mipSizes.mipSizes[maxBaseMipLevel(mipSizes)];
  }

  if(totalUsed <= budgetBytes)
    return;

  if(minUsed > budgetBytes)
  {
    LOGW("Texture budget (%zu MiB) is below the minimum required for all lowest mips (%zu MiB).\n",
         budgetBytes / (1024 * 1024), minUsed / (1024 * 1024));
    for(size_t textureIndex = 0; textureIndex < textureMipSizes.size(); textureIndex++)
    {
      if(textureMipSizes[textureIndex].mipCount > 0)
        baseMipLevels[textureIndex] = maxBaseMipLevel(textureMipSizes[textureIndex]);
    }
    return;
  }

  while(totalUsed > budgetBytes)
  {
    bool droppedAny = false;
    for(size_t textureIndex = 0; textureIndex < textureMipSizes.size(); textureIndex++)
    {
      const MipSizes& mipSizes = textureMipSizes[textureIndex];
      const uint32_t  maxBase  = maxBaseMipLevel(mipSizes);
      if(mipSizes.mipCount == 0 || baseMipLevels[textureIndex] >= maxBase)
        continue;

      totalUsed -= mipSizes.mipSizes[baseMipLevels[textureIndex]];
      baseMipLevels[textureIndex]++;
      droppedAny = true;

      if(totalUsed <= budgetBytes)
        return;
    }

    if(!droppedAny)
      break;
  }
}

struct AsyncImageLoader;

bool uploadDdsImage(AsyncImageLoader& loader, nvvk::Image& vkImage, nv_dds::Image& image, const char* mapped, size_t mappedSize, bool srgb, uint32_t baseMipLevel);

bool uploadKtxImage(AsyncImageLoader& loader, nvvk::Image& vkImage, nv_ktx::KTXImage& image, const char* mapped, size_t mappedSize, bool srgb, uint32_t baseMipLevel);

struct TextureBatchProgress
{
  const char* label = nullptr;

  uint32_t imageCount = 0;

  uint32_t   completedCount      = 0;
  uint32_t   progressLastPercent = 0;
  std::mutex progressMutex;

  nvutils::PerformanceTimer clock;
  double                    startTime = 0;

  // optional UI progress hooks (see SceneProgressInfo)
  SceneProgressInfo progressInfo;
  LoadPhase         phase = LoadPhase::LoadingTextures;  // LoadPhase for this pass

  void logBegin(const char* label_, uint32_t imageCount_, uint32_t threadCount)
  {
    label               = label_;
    imageCount          = imageCount_;
    completedCount      = 0;
    progressLastPercent = 0;
    startTime           = clock.getMicroseconds();
    progressInfo.beginPhase(phase, imageCount);
    LOGI("... %s: images %u, threads %u\n", label, imageCount, threadCount);
  }

  void logCompleted()
  {
    std::lock_guard lock(progressMutex);

    completedCount++;
    progressInfo.setCompleted(completedCount);

    if(imageCount == 0)
      return;

    const uint32_t percentage = uint32_t(double(completedCount * 100) / double(imageCount));

    constexpr uint32_t percentageGranularity = 5;
    const uint32_t     percentageSnapped     = (percentage / percentageGranularity) * percentageGranularity;

    if(percentageSnapped > progressLastPercent)
    {
      progressLastPercent = percentageSnapped;
      LOGI("... %s: %3d%%\n", label, percentageSnapped);
    }
  }

  void logEnd()
  {
    const double endTime = clock.getMicroseconds();
    LOGI("... %s: %f milliseconds\n", label, (endTime - startTime) / 1000.0f);
  }
};

// Loads scene textures from disk and uploads them through the async transfer-queue uploader.
// Thread-safe: uploadImage() may be called concurrently from worker threads; the AsyncUploader
// serializes its batches and submits internally.
struct AsyncImageLoader
{
  Resources&         res;
  std::atomic_bool   failed{false};
  std::atomic_bool   missing{false};
  std::atomic_size_t uploadedMemBytes{0};

  AsyncImageLoader(Resources& res_)
      : res(res_)
  {
  }

  void uploadImage(const std::string& fileName, bool sRGB, uint64_t imageID, nvvk::Image& image, uint32_t baseMipLevel)
  {
    nvutils::FileReadMapping readMapping;
    if(!readMapping.open(fileName))
    {
      LOGW("image loader: file not found %s\n", fileName.c_str());
      missing = true;
      return;
    }

    const char*  mapped     = (const char*)readMapping.data();
    const size_t mappedSize = readMapping.size();

    bool uploaded = false;

    if(isDdsFile(mapped, mappedSize))
    {
      nv_dds::Image        ddsImage;
      nv_dds::ReadSettings readSettings{};
      readSettings.validateInputSize = false;
      if(!ddsImage.readHeaderFromMemory(mapped, mappedSize, readSettings).has_value()
         && validateDdsImage(ddsImage, imageID) && baseMipLevel < ddsImage.getNumMips())
      {
        uploaded = uploadDdsImage(*this, image, ddsImage, mapped, mappedSize, sRGB, baseMipLevel);
      }
    }
    else if(isKtxFile(mapped, mappedSize))
    {
      nv_ktx::KTXImage     ktxImage;
      nv_ktx::ReadSettings readSettings{};
      readSettings.validate_input_size = false;
      if(!ktxImage.readHeaderFromMemory(mapped, mappedSize, readSettings).has_value()
         && validateKtxImage(ktxImage, imageID) && baseMipLevel < ktxImage.num_mips)
      {
        uploaded = uploadKtxImage(*this, image, ktxImage, mapped, mappedSize, sRGB, baseMipLevel);
      }
    }

    if(!uploaded)
    {
      LOGW("image loader: failed to load %s\n", fileName.c_str());
      failed = true;
    }
  }
};

bool uploadDdsImage(AsyncImageLoader& loader, nvvk::Image& vkImage, nv_dds::Image& image, const char* mapped, size_t mappedSize, bool srgb, uint32_t baseMipLevel)
{
  if(baseMipLevel >= image.getNumMips())
    return false;

  const VkFormat format =
      texture_formats::tryForceVkFormatTransferFunction(texture_formats::dxgiToVulkan(image.dxgiFormat), srgb);
  if(format == VK_FORMAT_UNDEFINED)
  {
    LOGW("Could not determine a VkFormat for DXGI format %u (%s).\n", image.dxgiFormat,
         texture_formats::getDXGIFormatName(image.dxgiFormat));
    return false;
  }

  const uint32_t   mipLevels = image.getNumMips() - baseMipLevel;
  const VkExtent3D extent    = {image.getWidth(baseMipLevel),   //
                                image.getHeight(baseMipLevel),  //
                                image.getDepth(baseMipLevel)};

  // Read every mip of the tail, not just the first. numMips defaults to 1, so leaving it unset would
  // read only mip `baseMipLevel` and leave the remaining mip levels (still allocated + sampled) black.
  nv_dds::SubresourceRange range{.firstMip = baseMipLevel, .numMips = mipLevels};

  VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
  imageCreateInfo.extent            = extent;
  imageCreateInfo.mipLevels         = mipLevels;
  imageCreateInfo.format            = format;
  imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  if(VK_SUCCESS != NVVK_FAIL_REPORT(loader.res.m_allocator.createImage(vkImage, imageCreateInfo, DEFAULT_VkImageViewCreateInfo)))
    return false;

  AsyncUploader& uploader = loader.res.m_asyncUploader;

  // Acquire one staging mapping per mip and decode the mip tail straight into staging (no extra copy).
  std::vector<nv_dds::SubresourceTarget>        mipTargets(mipLevels);
  std::vector<nvvk::BufferRange>                mipSpaces(mipLevels);
  std::vector<UploaderInterface::MappingHandle> mipHandles(mipLevels, 0);

  size_t totalSize = 0;
  bool   acquired  = true;
  for(uint32_t m = 0; m < mipLevels; m++)
  {
    const size_t capacity         = image.getSubresourceByteSize(baseMipLevel + m);
    mipTargets[m].capacityInBytes = capacity;
    if(uploader.acquireMapping(capacity, mipSpaces[m], mipHandles[m]) != VK_SUCCESS)
    {
      acquired = false;
      break;
    }
    mipTargets[m].data = mipSpaces[m].mapping;
    totalSize += capacity;
  }

  const bool decodeOk = acquired && !image.readSubresourcesFromMemory(mapped, mappedSize, range, mipTargets.data()).has_value();

  if(!decodeOk)
  {
    for(uint32_t m = 0; m < mipLevels; m++)
      if(mipHandles[m])
        uploader.releaseMapping(mipHandles[m]);
    return false;
  }

  VkExtent3D uploadExtent = extent;
  for(uint32_t m = 0; m < mipLevels; m++)
  {
    UploaderInterface::MappedImageSubs sub{};
    sub.offset       = {0, 0, 0};
    sub.extent       = uploadExtent;
    sub.subresource  = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
    sub.mappingSpace = mipSpaces[m];

    // each destination mip subresource starts undefined; appendImageSubMappings reads the old layout here
    vkImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if(VK_SUCCESS != NVVK_FAIL_REPORT(uploader.appendImageSubMappings(vkImage, 1, &sub, mipHandles[m], true, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)))
      return false;

    uploadExtent.width  = mipDimension(uploadExtent.width, 1);
    uploadExtent.height = mipDimension(uploadExtent.height, 1);
    uploadExtent.depth  = mipDimension(uploadExtent.depth, 1);
  }

  // final layout for the descriptor write (async uploader transitions each mip to this)
  vkImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  loader.uploadedMemBytes += totalSize;

  return true;
}

bool uploadKtxImage(AsyncImageLoader& loader, nvvk::Image& vkImage, nv_ktx::KTXImage& image, const char* mapped, size_t mappedSize, bool srgb, uint32_t baseMipLevel)
{
  if(baseMipLevel >= image.num_mips)
    return false;

  const VkFormat format = texture_formats::tryForceVkFormatTransferFunction(image.format, srgb);
  if(format == VK_FORMAT_UNDEFINED)
    return false;

  const uint32_t   mipLevels = image.num_mips - baseMipLevel;
  const VkExtent3D extent    = {mipDimension(image.mip_0_width, baseMipLevel),   //
                                mipDimension(image.mip_0_height, baseMipLevel),  //
                                mipDimension(image.mip_0_depth, baseMipLevel)};

  // Read every mip of the tail, not just the first. numMips defaults to 1, so leaving it unset would
  // read only mip `baseMipLevel` and leave the remaining mip levels (still allocated + sampled) black.
  nv_ktx::SubresourceRange range{.firstMip = baseMipLevel, .numMips = mipLevels};

  VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
  imageCreateInfo.extent            = extent;
  imageCreateInfo.mipLevels         = mipLevels;
  imageCreateInfo.format            = format;
  imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  VkImageViewCreateInfo imageViewCreateInfo = DEFAULT_VkImageViewCreateInfo;
  imageViewCreateInfo.components            = ktxSwizzleToVkComponentMapping(image.swizzle);
  imageViewCreateInfo.format                = format;

  if(VK_SUCCESS != NVVK_FAIL_REPORT(loader.res.m_allocator.createImage(vkImage, imageCreateInfo, imageViewCreateInfo)))
    return false;

  AsyncUploader& uploader = loader.res.m_asyncUploader;

  // Acquire one staging mapping per mip and decode the mip tail straight into staging (no extra copy).
  std::vector<nv_ktx::SubresourceTarget>        mipTargets(mipLevels);
  std::vector<nvvk::BufferRange>                mipSpaces(mipLevels);
  std::vector<UploaderInterface::MappingHandle> mipHandles(mipLevels, 0);

  size_t totalSize = 0;
  bool   acquired  = true;
  for(uint32_t m = 0; m < mipLevels; m++)
  {
    const size_t capacity         = image.getSubresourceByteSize(baseMipLevel + m);
    mipTargets[m].capacityInBytes = capacity;
    if(uploader.acquireMapping(capacity, mipSpaces[m], mipHandles[m]) != VK_SUCCESS)
    {
      acquired = false;
      break;
    }
    mipTargets[m].data = mipSpaces[m].mapping;
    totalSize += capacity;
  }

  const bool decodeOk = acquired && !image.readSubresourcesFromMemory(mapped, mappedSize, range, mipTargets.data()).has_value();

  if(!decodeOk)
  {
    for(uint32_t m = 0; m < mipLevels; m++)
      if(mipHandles[m])
        uploader.releaseMapping(mipHandles[m]);
    return false;
  }

  VkExtent3D uploadExtent = extent;
  for(uint32_t m = 0; m < mipLevels; m++)
  {
    UploaderInterface::MappedImageSubs sub{};
    sub.offset       = {0, 0, 0};
    sub.extent       = uploadExtent;
    sub.subresource  = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
    sub.mappingSpace = mipSpaces[m];

    // each destination mip subresource starts undefined; appendImageSubMappings reads the old layout here
    vkImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if(VK_SUCCESS != NVVK_FAIL_REPORT(uploader.appendImageSubMappings(vkImage, 1, &sub, mipHandles[m], true, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)))
      return false;

    uploadExtent.width  = mipDimension(uploadExtent.width, 1);
    uploadExtent.height = mipDimension(uploadExtent.height, 1);
    uploadExtent.depth  = mipDimension(uploadExtent.depth, 1);
  }

  // final layout for the descriptor write (async uploader transitions each mip to this)
  vkImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  loader.uploadedMemBytes += totalSize;

  return true;
}

}  // namespace


bool SceneTextures::init(Resources* res, const Scene& scene, const SceneTexturesConfig& config, const SceneProgressInfo& progressInfo)
{
  m_res = res;

  uint32_t imageCount = uint32_t(scene.m_images.size());

  m_images.resize(imageCount, {});

  {
    VkResult          result;
    VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
    imageCreateInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
    imageCreateInfo.extent            = {1, 1, 1};
    imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    result = NVVK_FAIL_REPORT(res->m_allocator.createImage(m_defaultWhiteImage, imageCreateInfo, DEFAULT_VkImageViewCreateInfo));
    if(result != VK_SUCCESS)
    {
      LOGW("Failed to create default image\n");
      return false;
    }
    result = NVVK_FAIL_REPORT(res->m_allocator.createImage(m_defaultBlackImage, imageCreateInfo, DEFAULT_VkImageViewCreateInfo));
    if(result != VK_SUCCESS)
    {
      LOGW("Failed to create default image\n");
      return false;
    }
    result = NVVK_FAIL_REPORT(res->m_allocator.createImage(m_defaultNormalImage, imageCreateInfo, DEFAULT_VkImageViewCreateInfo));
    if(result != VK_SUCCESS)
    {
      LOGW("Failed to create default image\n");
      return false;
    }
  }

  // multi-thread the loading of scene->m_images
  // insert a mutex to flush the staging space and one to acquire space.

  std::vector<uint32_t> textureBaseMipLevels(imageCount, 0);
  if(config.maxBudgetMiB != 0 && imageCount > 0)
  {
    std::vector<MipSizes> textureMipSizes(imageCount);

    TextureBatchProgress probeProgress;
    probeProgress.progressInfo = progressInfo;
    probeProgress.phase        = LoadPhase::ProbingTextures;
    probeProgress.logBegin("texture budget probing", imageCount, config.maxThreads);

    nvutils::parallel_batches<1>(
        imageCount,
        [&](uint64_t idx) {
          nvutils::FileReadMapping readMapping;
          if(!readMapping.open(scene.m_images[idx].filename))
          {
            LOGW("image probe: file not found %s\n", scene.m_images[idx].filename.c_str());
            probeProgress.logCompleted();
            return;
          }

          if(!probeMipSizesFromMapped(textureMipSizes[idx], (const char*)readMapping.data(), readMapping.size(), idx))
          {
            LOGW("image probe: failed to probe %s\n", scene.m_images[idx].filename.c_str());
          }
          probeProgress.logCompleted();
        },
        config.maxThreads);

    probeProgress.logEnd();

    size_t totalFullBytes = 0;
    for(uint32_t i = 0; i < imageCount; i++)
    {
      if(textureMipSizes[i].mipCount == 0)
        continue;
      totalFullBytes += mipTailMemoryUsage(textureMipSizes[i], 0);
    }
    LOGI("... texture budget probing: full load %.2f MiB across %u textures\n", double(totalFullBytes) / (1024.0 * 1024.0), imageCount);

    const size_t budgetBytes = size_t(config.maxBudgetMiB) * 1024 * 1024;
    computeLoadPlansForBudget(textureMipSizes, textureBaseMipLevels, budgetBytes);

    size_t plannedBytes = 0;
    for(uint32_t i = 0; i < imageCount; i++)
    {
      if(textureMipSizes[i].mipCount > 0)
        plannedBytes += mipTailMemoryUsage(textureMipSizes[i], textureBaseMipLevels[i]);
    }
    LOGI("Texture budget: using %.2f / %u MiB across %u textures.\n", double(plannedBytes) / (1024.0 * 1024.0),
         config.maxBudgetMiB, imageCount);
  }

  {
    AsyncImageLoader loader(*res);

    // Upload the 1x1 default images (white / black / normal) through the async uploader.
    {
      const VkExtent3D               extent1     = {1, 1, 1};
      const VkImageSubresourceLayers subResource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};

      uint32_t defaultWhitePixel  = 0xFFFFFFFFu;
      uint32_t defaultBlackPixel  = 0xFF000000u;
      uint32_t defaultNormalPixel = 0xFFFF8080u;  // R=128, G=128, B=255, A=255

      auto uploadDefault = [&](nvvk::Image& img, uint32_t& pixel) {
        img.descriptor.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        res->m_asyncUploader.appendImageSub(img, {0, 0, 0}, extent1, subResource, sizeof(uint32_t), &pixel,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        img.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      };
      uploadDefault(m_defaultWhiteImage, defaultWhitePixel);
      uploadDefault(m_defaultBlackImage, defaultBlackPixel);
      uploadDefault(m_defaultNormalImage, defaultNormalPixel);
    }

    TextureBatchProgress uploadProgress;
    if(imageCount)
    {
      uploadProgress.progressInfo = progressInfo;
      uploadProgress.phase        = LoadPhase::LoadingTextures;
      uploadProgress.logBegin("texture upload", imageCount, config.maxThreads);

      nvutils::parallel_batches<1>(
          imageCount,
          [&](uint64_t idx) {
            loader.uploadImage(scene.m_images[idx].filename, scene.m_images[idx].sRGB, idx, m_images[idx],
                               textureBaseMipLevels[idx]);
            uploadProgress.logCompleted();
          },
          config.maxThreads);

      uploadProgress.logEnd();
    }

    // Submit any pending copies and wait for all transfer-queue uploads to finish. The graphics-queue
    // ownership acquisition happens later per-frame in onRender via cmdDrainOwnershipBarriers().
    res->m_asyncUploader.flushPending();
    res->m_asyncUploader.waitForCompletion();

    if(loader.failed)
      return false;

    m_textureMemBytes = loader.uploadedMemBytes;
  }

  std::vector<VkSampler> immutableSamplers;
  if(config.immutableSampler)
  {
    immutableSamplers.resize(Scene::NUM_IMAGE_DEFAULTS + imageCount, config.immutableSampler);
  }

  // always ensure 1 descriptor, even if empty.
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(0, config.immutableSampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                      imageCount + Scene::NUM_IMAGE_DEFAULTS, VK_SHADER_STAGE_ALL,
                      config.immutableSampler ? immutableSamplers.data() : nullptr, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

  m_dsetPack.init(bindings, res->m_device, 1);

  nvvk::WriteSetContainer writeSet;
  writeSet.append(m_dsetPack.makeWrite(0, 0, Scene::IMAGE_DEFAULT_WHITE, 1), m_defaultWhiteImage);
  writeSet.append(m_dsetPack.makeWrite(0, 0, Scene::IMAGE_DEFAULT_BLACK, 1), m_defaultBlackImage);
  writeSet.append(m_dsetPack.makeWrite(0, 0, Scene::IMAGE_DEFAULT_NORMAL, 1), m_defaultNormalImage);
  if(imageCount > 0)
  {
    for(uint32_t i = 0; i < imageCount; i++)
    {
      if(m_images[i].image)
      {
        writeSet.append(m_dsetPack.makeWrite(0, 0, i + Scene::NUM_IMAGE_DEFAULTS, 1), m_images[i]);
      }
      else
      {
        switch(scene.m_images[i].defaultType)
        {
          case Scene::IMAGE_DEFAULT_WHITE:
            writeSet.append(m_dsetPack.makeWrite(0, 0, i + Scene::NUM_IMAGE_DEFAULTS, 1), m_defaultWhiteImage);
            break;
          case Scene::IMAGE_DEFAULT_BLACK:
            writeSet.append(m_dsetPack.makeWrite(0, 0, i + Scene::NUM_IMAGE_DEFAULTS, 1), m_defaultBlackImage);
            break;
          case Scene::IMAGE_DEFAULT_NORMAL:
            writeSet.append(m_dsetPack.makeWrite(0, 0, i + Scene::NUM_IMAGE_DEFAULTS, 1), m_defaultNormalImage);
            break;
        }
      }
    }
  }

  vkUpdateDescriptorSets(res->m_device, writeSet.size(), writeSet.data(), 0, nullptr);

  return true;
}

void SceneTextures::acquireOwnership(Resources* res)
{
  // init() uploaded the images on the transfer queue and released their queue-family ownership.
  // Perform the matching graphics-queue acquisition here (main thread) via a one-off temp submit,
  // and free the upload staging. No-op if init() ran without the async uploader having pending work.
  if(!res->m_asyncUploader.hasOwnershipBarriers())
    return;

  VkCommandBuffer cmd = res->createTempCmdBuffer();
  res->m_asyncUploader.cmdDrainOwnershipBarriers(cmd);
  res->tempSyncSubmit(cmd);
  res->m_asyncUploader.releaseCompletedAllocations();
}

void SceneTextures::deinit()
{
  for(auto& image : m_images)
  {
    m_res->m_allocator.destroyImage(image);
  }
  m_res->m_allocator.destroyImage(m_defaultWhiteImage);
  m_res->m_allocator.destroyImage(m_defaultBlackImage);
  m_res->m_allocator.destroyImage(m_defaultNormalImage);
  m_images.clear();
  m_textureMemBytes = 0;
  m_dsetPack.deinit();
}

}  // namespace lodclusters
