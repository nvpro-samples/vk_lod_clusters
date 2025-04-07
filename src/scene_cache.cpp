
/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <nvh/parallel_work.hpp>
#include <nvh/filemapping.hpp>
#include <nvh/nvprint.hpp>

#include "scene.hpp"

namespace lodclusters {


bool Scene::storeCached(const GeometryView& view, uint64_t dataSize, void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = dataAddress % nvclusterlod::detail::ALIGNMENT == 0 && dataAddress + sizeof(GeometryBase) <= dataEnd;

  if(isValid)
  {
    memcpy(reinterpret_cast<void*>(dataAddress), (const GeometryBase*)&view, sizeof(GeometryBase));
    dataAddress += (sizeof(GeometryBase) + nvclusterlod::detail::ALIGN_MASK) & ~nvclusterlod::detail::ALIGN_MASK;
  }

  if(isValid)
  {
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.vertices);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.localTriangles);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.clusterVertexRanges);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.clusterBboxes);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupLodLevels);
    nvclusterlod::detail::storeAndAdvance(isValid, dataAddress, dataEnd, view.nodeBboxes);
  }


  isValid = isValid && nvclusterlod::storeCached(view.lodMesh, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodMesh);
  isValid = isValid && nvclusterlod::storeCached(view.lodHierarchy, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodHierarchy);

  return isValid;
}

bool Scene::loadCached(GeometryView& view, uint64_t dataSize, const void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  if(dataAddress % nvclusterlod::detail::ALIGNMENT == 0 && dataAddress + sizeof(GeometryBase) <= dataEnd)
  {
    memcpy((GeometryBase*)&view, data, sizeof(GeometryBase));
    dataAddress += (sizeof(GeometryBase) + nvclusterlod::detail::ALIGN_MASK) & ~nvclusterlod::detail::ALIGN_MASK;
  }
  else
  {
    view = {};
    return false;
  }

  if(isValid)
  {
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.vertices);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.localTriangles);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.clusterVertexRanges);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.clusterBboxes);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupLodLevels);
    nvclusterlod::detail::loadAndAdvance(isValid, dataAddress, dataEnd, view.nodeBboxes);
  }

  isValid = isValid && nvclusterlod::loadCached(view.lodMesh, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodMesh);
  isValid = isValid && nvclusterlod::loadCached(view.lodHierarchy, dataEnd - dataAddress, reinterpret_cast<void*>(dataAddress));
  dataAddress += nvclusterlod::getCachedSize(view.lodHierarchy);

  return isValid;
}

bool Scene::CacheFileView::init(uint64_t dataSize, const void* data)
{
  m_dataSize  = dataSize;
  m_dataBytes = reinterpret_cast<const uint8_t*>(data);

  if(dataSize <= sizeof(CacheHeader) + sizeof(uint64_t))
  {
    m_dataSize = 0;
    return false;
  }

  const CacheHeader* fileHeader = (const CacheHeader*)data;

  if(!fileHeader->isValid())
  {
    m_dataSize = 0;
    return false;
  }

  m_geometryCount = *getPointer<uint64_t>(m_dataSize - sizeof(uint64_t));

  if(dataSize <= (sizeof(CacheHeader) + sizeof(uint64_t) * (m_geometryCount + 2)))
  {
    m_dataSize = 0;
    return false;
  }

  m_tableStart = m_dataSize - sizeof(uint64_t) * (m_geometryCount + 2);

  return true;
}

bool Scene::CacheFileView::getGeometryView(GeometryView& view, uint64_t geometryIndex) const
{
  constexpr uint64_t ALIGN_MASK = nvclusterlod::detail::ALIGNMENT - 1;

  if(geometryIndex >= m_geometryCount)
  {
    assert(0);
    return false;
  }

  const uint64_t* geometryOffsets = getPointer<uint64_t>(m_tableStart, m_geometryCount + 1);
  uint64_t        base            = geometryOffsets[geometryIndex];

  if(base + sizeof(GeometryBase) > m_tableStart)
  {
    // this must not happen on a valid file
    assert(0);
    return false;
  }

  uint64_t geometryTotalSize = geometryOffsets[geometryIndex + 1] - base;
  uint64_t baseEnd           = base + geometryTotalSize;

  const uint8_t* geoData = getPointer<uint8_t>(base, geometryTotalSize);

  return Scene::loadCached(view, geometryTotalSize, geoData);
}

bool Scene::checkCache(const nvclusterlod::LodGeometryInfo& info, size_t geometryIndex)
{
  if(m_cacheFileView.isValid() && geometryIndex < m_cacheFileView.getGeometryCount())
  {
    GeometryView cacheView = {};
    if(!m_cacheFileView.getGeometryView(cacheView, geometryIndex))
    {
      return false;
    }

    // ignore these during compare
    cacheView.lodInfo.decimationFactor = info.decimationFactor;
    cacheView.lodInfo.groupConfig      = info.groupConfig;
    cacheView.lodInfo.clusterConfig    = info.clusterConfig;

    return memcmp(&info, &cacheView.lodInfo, sizeof(nvclusterlod::LodGeometryInfo)) == 0;
  }
  return false;
}

template <typename T>
static inline void fillVector(std::vector<T>& storageVec, const std::span<const T>& viewSpan)
{
  storageVec.resize(viewSpan.size());
  memcpy(storageVec.data(), viewSpan.data(), viewSpan.size_bytes());
}

void Scene::loadCachedGeometry(GeometryStorage& storage, size_t geometryIndex)
{
  GeometryView view = {};
  m_cacheFileView.getGeometryView(view, geometryIndex);
  (GeometryBase&)storage = view;

  fillVector(storage.vertices, view.vertices);
  fillVector(storage.localTriangles, view.localTriangles);
  fillVector(storage.clusterVertexRanges, view.clusterVertexRanges);
  fillVector(storage.clusterBboxes, view.clusterBboxes);
  fillVector(storage.groupLodLevels, view.groupLodLevels);
  fillVector(storage.nodeBboxes, view.nodeBboxes);

  nvclusterlod::toStorage(view.lodMesh, storage.lodMesh);
  nvclusterlod::toStorage(view.lodHierarchy, storage.lodHierarchy);
}

bool Scene::saveCache() const
{
  uint64_t dataSize = sizeof(Scene::CacheHeader);

  std::vector<uint64_t> geometryOffsets;
  geometryOffsets.reserve(m_geometryViews.size() + 2);

  for(const GeometryView& geom : m_geometryViews)
  {
    geometryOffsets.push_back(dataSize);

    dataSize += geom.getCachedSize();
  }
  geometryOffsets.push_back(dataSize);
  geometryOffsets.push_back(m_geometryViews.size());

  dataSize += geometryOffsets.size() * sizeof(uint64_t);

  nvh::FileReadOverWriteMapping outMapping;

  std::string outFilename = m_filename + ".nvsngeo";

  if(!outMapping.open(outFilename.c_str(), dataSize))
  {
    LOGE("Scene::saveCache failed to save file %s", outFilename.c_str());
    return false;
  }

  uint8_t* mappingData = static_cast<uint8_t*>(outMapping.data());

  // write header
  Scene::CacheHeader cacheHeader;
  memcpy(mappingData, &cacheHeader, sizeof(cacheHeader));
  // write offset table at end
  memcpy(mappingData + geometryOffsets[m_geometryViews.size()], geometryOffsets.data(),
         sizeof(uint64_t) * geometryOffsets.size());

  bool hadError = false;
  nvh::parallel_batches(m_geometryViews.size(), [&](uint64_t idx) {
    const GeometryView& view = m_geometryViews[idx];

    uint64_t dataOffset = geometryOffsets[idx];
    uint64_t dataSize   = geometryOffsets[idx + 1] - dataOffset;

    if(!Scene::storeCached(view, dataSize, mappingData + dataOffset))
    {
      hadError = true;
    }
  });

  if(hadError)
  {
    LOGE("Scene::saveCache had errors while saving %s\n", outFilename.c_str());
  }
  else
  {
    LOGI("Scene::saveCache saved %s\n", outFilename.c_str());
  }

  return !hadError;
}

}  // namespace lodclusters
