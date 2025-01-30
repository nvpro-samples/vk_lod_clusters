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
#pragma once

#include <memory>

#include "resources.hpp"
#include "scene.hpp"
#include "scene_preloaded.hpp"
#include "scene_streaming.hpp"

namespace lodclusters {

// There are two implementations for a renderable scene.
// Everything is preloaded or we stream in data dynamically.
class RenderScene
{
public:
  const Scene*   scene        = nullptr;
  bool           useStreaming = false;
  ScenePreloaded scenePreloaded;
  SceneStreaming sceneStreaming;

  // pointers must stay valid during lifetime
  bool init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_);
  void deinit();

  void streamingReset();

  bool updateClasRequired(bool state);

  const RBufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const;
  size_t                                  getClasSize(bool reserved) const;
  size_t                                  getOperationsSize() const;
  size_t                                  getGeometrySize(bool reserved) const;
};

struct RendererConfig
{
  bool flipWinding = false;

  // the maximum number of renderable clusters per frame in bits i.e. (1 << number)
  uint32_t numRenderClusterBits = 20;
  // the maximum number of traversal intermediate tasks
  uint32_t numTraversalTaskBits = 20;

  // build flags for the cluster BLAS
  VkBuildAccelerationStructureFlagsKHR clusterBlasFlags = 0;
};

class Renderer
{
public:
  struct ResourceUsageInfo
  {
    size_t rtTlasMemBytes{};
    size_t rtBlasMemBytes{};
    size_t rtClasMemBytes{};
    size_t operationsMemBytes{};
    size_t geometryMemBytes{};

    void add(const ResourceUsageInfo& other)
    {
      rtTlasMemBytes += other.rtTlasMemBytes;
      rtBlasMemBytes += other.rtBlasMemBytes;
      rtClasMemBytes += other.rtClasMemBytes;
      operationsMemBytes += other.operationsMemBytes;
      geometryMemBytes += other.geometryMemBytes;
    }
    size_t getTotalSum() const
    {
      return rtTlasMemBytes + rtBlasMemBytes + rtClasMemBytes + geometryMemBytes + operationsMemBytes;
    }
  };

  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer(){};  // Defined only so that inherited classes also have virtual destructors. Use deinit().
  virtual void updatedFrameBuffer(Resources& res) { updatedFrameBufferBasics(res); };

  virtual bool supportsClusters() const { return true; }

  inline ResourceUsageInfo getResourceUsage(bool reserved) const
  {
    return reserved ? m_resourceReservedUsage : m_resourceActualUsage;
  };

protected:
  bool initBasicShaders(Resources& res);
  void initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config);
  void deinitBasics(Resources& res);

  void updatedFrameBufferBasics(Resources& res);

  void initWriteRayTracingDepthBuffer(Resources& res);
  void writeRayTracingDepthBuffer(VkCommandBuffer cmd);

  struct BasicShaders
  {
    nvvk::ShaderModuleID fullScreenVertexShader;
    nvvk::ShaderModuleID fullScreenWriteDepthFragShader;
  };

  BasicShaders m_basicShaders;

  std::vector<shaderio::RenderInstance> m_renderInstances;
  RBuffer                               m_renderInstanceBuffer;

  ResourceUsageInfo m_resourceReservedUsage{};
  ResourceUsageInfo m_resourceActualUsage{};

  nvvk::DescriptorSetContainer m_writeDepthBufferDsetContainer;
  VkPipeline                   m_writeDepthBufferPipeline = nullptr;
};

//////////////////////////////////////////////////////////////////////////

std::unique_ptr<Renderer> makeRendererRasterClustersTess();
std::unique_ptr<Renderer> makeRendererRayTraceClustersTess();

}  // namespace lodclusters
