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

#include <fmt/format.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/camera.hpp>

#include "lodclusters.hpp"

bool g_verbose = false;

namespace lodclusters {

LodClusters::LodClusters(const Info& info)
    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";

  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);

  m_info.parameterRegistry->add({"scene"}, {".gltf", ".glb"}, &m_sceneFilePath);
  m_info.parameterRegistry->add({"renderer"}, (int*)&m_tweak.renderer);
  m_info.parameterRegistry->add({"verbose"}, &g_verbose, true);
  m_info.parameterRegistry->add({"resetstats"}, &m_tweak.autoResetTimers);
  m_info.parameterRegistry->add({"supersample"}, &m_tweak.supersample);

  m_info.parameterRegistry->add({"dumpspirv", "dumps compiled spirv into working directory"}, &m_resources.m_dumpSpirv);

  m_info.parameterRegistry->add({"streaming"}, &m_tweak.useStreaming);
  m_info.parameterRegistry->add({"clasallocator"}, &m_streamingConfig.usePersistentClasAllocator);
  m_info.parameterRegistry->add({"gridcopies"}, &m_sceneGridConfig.numCopies);
  m_info.parameterRegistry->add({"gridconfig"}, &m_sceneGridConfig.gridBits);
  m_info.parameterRegistry->add({"gridunique"}, &m_sceneGridConfig.uniqueGeometriesForCopies);
  m_info.parameterRegistry->add({"clusterconfig"}, (int*)&m_tweak.clusterConfig);
  m_info.parameterRegistry->add({"loderror"}, &m_frameConfig.lodPixelError);
  m_info.parameterRegistry->add({"culling"}, &m_tweak.useCulling);
  m_info.parameterRegistry->add({"instancesorting"}, &m_rendererConfig.useSorting);
  m_info.parameterRegistry->add({"renderclusterbits"}, &m_rendererConfig.numRenderClusterBits);
  m_info.parameterRegistry->add({"rendertraversalbits"}, &m_rendererConfig.numTraversalTaskBits);
  m_info.parameterRegistry->add({"hbao"}, &m_tweak.hbaoActive);
  m_info.parameterRegistry->add({"facetshading"}, &m_tweak.facetShading);
  m_info.parameterRegistry->add({"flipwinding"}, &m_rendererConfig.flipWinding);
  m_info.parameterRegistry->add({"twosided"}, &m_rendererConfig.twoSided);
  m_info.parameterRegistry->add({"autosavecache", "automatically store cache file for loaded scene. default false"},
                                &m_sceneConfig.autoSaveCache);
  m_info.parameterRegistry->add({"autoloadcache", "automatically load cache file if found. default true"},
                                &m_sceneConfig.autoLoadCache);
  m_info.parameterRegistry->add({"mappedcache", "work from memory mapped cache file, otherwise load to sysmem. default false"},
                                &m_sceneConfig.memoryMappedCache);
  m_info.parameterRegistry->add({"processingonly", "directly terminate app once cache file was saved. default false"},
                                &m_sceneConfig.processingOnly);
  m_info.parameterRegistry->add({"processingthreadpct", "float percentage of threads during initial file load and processing into lod clusters, default 0.5 == 50 %"},
                                &m_sceneConfig.processingThreadsPct);

  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants.ambientOcclusionSamples = 1;
  m_frameConfig.frameConstants.facetShading            = 1;
  m_frameConfig.frameConstants.doShadow                = 1;
  m_frameConfig.frameConstants.ambientOcclusionRadius  = 0.1f;

  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants.wireThickness           = 2.f;
  m_frameConfig.frameConstants.wireSmoothing           = 1.f;
  m_frameConfig.frameConstants.wireColor               = {118.f / 255.f, 185.f / 255.f, 0.f};
  m_frameConfig.frameConstants.wireStipple             = 0;
  m_frameConfig.frameConstants.wireBackfaceColor       = {0.5f, 0.5f, 0.5f};
  m_frameConfig.frameConstants.wireStippleRepeats      = 5;
  m_frameConfig.frameConstants.wireStippleLength       = 0.5f;
  m_frameConfig.frameConstants.doShadow                = 1;
  m_frameConfig.frameConstants.doWireframe             = 0;
  m_frameConfig.frameConstants.ambientOcclusionRadius  = 0.1f;
  m_frameConfig.frameConstants.ambientOcclusionSamples = 16;
  m_frameConfig.frameConstants.visualize               = VISUALIZE_LOD;

  m_frameConfig.frameConstants.lightMixer = 0.5f;
  m_frameConfig.frameConstants.skyParams  = {};
}
bool LodClusters::initScene(const std::filesystem::path& filePath)
{
  deinitScene();

  std::string fileName = nvutils::utf8FromPath(filePath);

  if(!fileName.empty())
  {
    LOGI("Loading scene %s\n", fileName.c_str());

    m_scene = std::make_unique<Scene>();
    if(!m_scene->init(filePath, m_sceneConfig))
    {
      m_scene = nullptr;
      LOGW("Loading scene failed\n");
    }
    else
    {
      adjustSceneClusterConfig();

      m_scene->updateSceneGrid(m_sceneGridConfig);
      m_sceneGridConfigLast = m_sceneGridConfig;
    }

    m_sceneFilePath = filePath;

    if(m_scene)
    {
      initRenderScene();
    }

    return m_scene != nullptr && m_renderScene != nullptr;
  }

  return true;
}

void LodClusters::initRenderScene()
{
  assert(m_scene);

  m_renderScene = std::make_unique<RenderScene>();

  bool success = m_renderScene->init(&m_resources, m_scene.get(), m_streamingConfig, m_tweak.useStreaming);

  // if preload fails, try streaming
  if(!m_tweak.useStreaming && !success)
  {
    // override to use streaming
    m_tweak.useStreaming     = true;
    m_tweakLast.useStreaming = true;

    if(!m_renderScene->init(&m_resources, m_scene.get(), m_streamingConfig, true))
    {
      LOGW("Init renderscene failed\n");
      deinitRenderScene();
    }
  }
  else if(!success && m_tweak.useStreaming)
  {
    LOGW("Init renderscene failed\n");
    deinitRenderScene();
  }

  m_streamingConfigLast = m_streamingConfig;
}

void LodClusters::deinitRenderScene()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
  if(m_renderScene)
  {
    m_renderScene->deinit();
    m_renderScene = nullptr;
  }
}

void LodClusters::deinitScene()
{
  deinitRenderScene();

  if(m_scene)
  {
    m_scene->deinit();
    m_scene = nullptr;
  }
}

void LodClusters::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_windowSize = size;
  m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoFullRes);
  updateImguiImage();
  if(m_renderer)
  {
    m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
    m_rendererFboChangeID = m_resources.m_fboChangeID;
  }
}

void LodClusters::updateImguiImage()
{
  if(m_imguiTexture)
  {
    ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);
    m_imguiTexture = nullptr;
  }

  VkImageView imageView = m_resources.m_frameBuffer.useResolved ? m_resources.m_frameBuffer.imgColorResolved.descriptor.imageView :
                                                                  m_resources.m_frameBuffer.imgColor.descriptor.imageView;

  assert(imageView);

  m_imguiTexture = ImGui_ImplVulkan_AddTexture(m_imguiSampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void LodClusters::onPreRender()
{
  m_profilerTimeline->frameAdvance();
}


void LodClusters::deinitRenderer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_renderer)
  {
    m_renderer->deinit(m_resources);
    m_renderer = nullptr;
  }
}

void LodClusters::initRenderer(RendererType rtype)
{
  LOGI("Initializing renderer and compiling shaders\n");
  deinitRenderer();
  if(!m_renderScene)
    return;

  printf("init renderer %d\n", rtype);

  if(m_renderScene->useStreaming)
  {
    if(!m_renderScene->sceneStreaming.reloadShaders())
    {
      LOGE("RenderScene shaders failed\n");
      return;
    }
  }

  switch(rtype)
  {
    case RENDERER_RASTER_CLUSTERS_LOD:
      m_renderer = makeRendererRasterClustersLod();
      break;
    case RENDERER_RAYTRACE_CLUSTERS_LOD:
      m_renderer = makeRendererRayTraceClustersLod();
      break;
  }

  if(m_renderer && !m_renderer->init(m_resources, *m_renderScene, m_rendererConfig))
  {
    m_renderer = nullptr;
    LOGE("Renderer init failed\n");
  }

  m_rendererFboChangeID = m_resources.m_fboChangeID;
}

void LodClusters::postInitNewScene()
{
  assert(m_scene);

  float     sceneDimension = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);
  glm::vec3 center         = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  m_frameConfig.frameConstants.wLightPos = center + sceneDimension;
  m_frameConfig.frameConstants.sceneSize = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);

  setSceneCamera(m_sceneFilePath);

  m_frames = 0;

  m_streamingConfig.maxGroups = std::max(m_streamingConfig.maxGroups, uint32_t(m_scene->getActiveGeometryCount()));
}


void LodClusters::onAttach(nvapp::Application* app)
{
  m_app = app;

  m_tweak.supersample = std::max(1, m_tweak.supersample);

  m_info.cameraManipulator->setMode(nvutils::CameraManipulator::Fly);

  m_renderer = nullptr;

  {
    VkPhysicalDeviceProperties2 physicalProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
    physicalProperties.pNext = &smProperties;
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &physicalProperties);
    // pseudo heuristic
    m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 8;
  }

  {
    m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_CLUSTERS_LOD, "Rasterization");

    m_ui.enumAdd(GUI_BUILDMODE, 0, "default");
    m_ui.enumAdd(GUI_BUILDMODE, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR, "fast build");
    m_ui.enumAdd(GUI_BUILDMODE, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, "fast trace");

    if(!m_resources.m_supportsClusters)
    {
      LOGW("WARNING: Cluster raytracing extension not supported\n");
      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        m_tweak.renderer = RENDERER_RASTER_CLUSTERS_LOD;
      }
    }
    else
    {
      m_ui.enumAdd(GUI_RENDERER, RENDERER_RAYTRACE_CLUSTERS_LOD, "Ray tracing");
    }

    {
      for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
      {
        std::string enumStr = fmt::format("{}T_{}V", s_clusterInfos[i].tris, s_clusterInfos[i].verts);
        m_ui.enumAdd(GUI_MESHLET, s_clusterInfos[i].cfg, enumStr.c_str());
      }
    }

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_MATERIAL, "material");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GREY, "grey");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GROUP, "cluster groups");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_LOD, "lods");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLE, "triangles");
  }

  // Initialize core components

  m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  m_resources.init(app->getDevice(), app->getPhysicalDevice(), app->getInstance(), app->getQueue(0), app->getQueue(1));

  {
    NVVK_CHECK(m_resources.m_samplerPool.acquireSampler(m_imguiSampler));
    NVVK_DBG_NAME(m_imguiSampler);
  }

  m_resources.initFramebuffer({128, 128}, m_tweak.supersample, m_tweak.hbaoFullRes);
  updateImguiImage();

  updatedClusterConfig();

  // Search for default scene if none was provided on the command line
  if(m_sceneFilePath.empty())
  {
    const std::filesystem::path              exeDirectoryPath   = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> defaultSearchPaths = {
        // regular build
        std::filesystem::absolute(exeDirectoryPath / PROJECT_EXE_TO_DOWNLOAD_DIRECTORY),
        // install build
        std::filesystem::absolute(exeDirectoryPath / "resources"),
    };

    m_sceneFilePath = nvutils::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);

    if(m_sceneGridConfig.numCopies == 1)
    {
      if(m_resources.getDeviceLocalHeapSize() >= 8ull * 1024 * 1024 * 1024)
      {
        m_sceneGridConfig.numCopies = 1024;  // 32x32 grid
      }
      else
      {
        m_sceneGridConfig.numCopies = 64;
      }
    }
  }

  if(m_resources.getDeviceLocalHeapSize() >= 8ull * 1024 * 1024 * 1024)
  {
    m_streamingConfig.maxClasMegaBytes     = 2 * 1024;
    m_streamingConfig.maxGeometryMegaBytes = 2 * 1024;
  }
  else
  {
    m_streamingConfig.maxClasMegaBytes     = 1 * 1024;
    m_streamingConfig.maxGeometryMegaBytes = 1 * 1024;
  }

  if(initScene(m_sceneFilePath))
  {
    postInitNewScene();
    initRenderer(m_tweak.renderer);
  }

  m_tweakLast          = m_tweak;
  m_sceneConfigLast    = m_sceneConfig;
  m_rendererConfigLast = m_rendererConfig;
}

void LodClusters::onDetach()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  deinitRenderer();
  deinitScene();

  m_resources.m_samplerPool.releaseSampler(m_imguiSampler);
  ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);

  m_resources.deinit();

  m_profilerGpuTimer.deinit();
}

void LodClusters::saveCacheFile()
{
  if(m_scene)
  {
    m_scene->saveCache();
  }
}

void LodClusters::onFileDrop(const std::filesystem::path& filePath)
{
  // reset grid parameter (in case scene is too large to be replicated)
  m_sceneGridConfig.numCopies = 1;
  //
  if(filePath.empty())
    return;
  LOGI("Loading model: %s\n", nvutils::utf8FromPath(filePath).c_str());
  deinitRenderer();

  if(initScene(filePath))
  {
    postInitNewScene();
    initRenderer(m_tweak.renderer);
    m_tweakLast       = m_tweak;
    m_sceneConfigLast = m_sceneConfig;
  }
}


const LodClusters::ClusterInfo LodClusters::s_clusterInfos[NUM_CLUSTER_CONFIGS] = {
    {64, 64, CLUSTER_64T_64V},     {64, 128, CLUSTER_64T_128V},   {64, 192, CLUSTER_64T_192V},
    {96, 96, CLUSTER_96T_96V},     {128, 128, CLUSTER_128T_128V}, {128, 256, CLUSTER_128T_256V},
    {256, 256, CLUSTER_256T_256V},
};

void LodClusters::adjustSceneClusterConfig()
{
#if 0
  m_scene->m_clusterTriangleHistogram.resize(entry.tris + 1);
  m_scene->m_clusterVertexHistogram.resize(entry.verts + 1);

  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    const ClusterInfo& entry = s_clusterInfos[i];
    if(m_scene->m_maxClusterTriangles <= entry.tris && m_scene->m_maxClusterVertices <= entry.verts)
    {
      m_tweak.clusterConfig = entry.cfg;

      m_scene->m_config.clusterTriangles = entry.tris;
      m_sceneConfig.clusterTriangles     = entry.tris;
      m_sceneConfigLast.clusterTriangles = entry.tris;

      m_scene->m_config.clusterVertices = entry.verts;
      m_sceneConfig.clusterVertices     = entry.verts;
      m_sceneConfigLast.clusterVertices = entry.verts;
      return;
    }
  }
#endif
}

void LodClusters::updatedClusterConfig()
{
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    if(s_clusterInfos[i].cfg == m_tweak.clusterConfig)
    {
      m_sceneConfig.clusterTriangles = s_clusterInfos[i].tris;
      m_sceneConfig.clusterVertices  = s_clusterInfos[i].verts;
      return;
    }
  }
}

void LodClusters::handleChanges()
{
  if(m_tweak.clusterConfig != m_tweakLast.clusterConfig)
  {
    updatedClusterConfig();
  }

  bool sceneChanged = false;
  if(memcmp(&m_sceneConfig, &m_sceneConfigLast, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;

    deinitRenderer();
    initScene(m_sceneFilePath);
  }

  bool sceneGridChanged = false;
  if(!sceneChanged && memcmp(&m_sceneGridConfig, &m_sceneGridConfigLast, sizeof(m_sceneGridConfig)) && m_scene)
  {
    sceneGridChanged = true;

    deinitRenderer();
    m_scene->updateSceneGrid(m_sceneGridConfig);
  }

  bool shaderChanged = false;
  if(m_reloadShaders)
  {
    shaderChanged   = true;
    m_reloadShaders = false;
  }

  bool frameBufferChanged = false;
  if(tweakChanged(m_tweak.supersample) || tweakChanged(m_tweak.hbaoFullRes))
  {
    m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoFullRes);
    updateImguiImage();

    frameBufferChanged = true;
  }

  bool renderSceneChanged = false;
  if(sceneGridChanged || tweakChanged(m_tweak.useStreaming)
     || (memcmp(&m_streamingConfig, &m_streamingConfigLast, sizeof(m_streamingConfig))))
  {
    if(!sceneChanged || !sceneGridChanged)
    {
      deinitRenderer();
    }

    renderSceneChanged = true;
    deinitRenderScene();
    initRenderScene();
  }

  bool rendererChanged = false;
  if(sceneChanged || shaderChanged || renderSceneChanged || tweakChanged(m_tweak.renderer)
     || tweakChanged(m_tweak.useDebugVisualization) || rendererCfgChanged(m_rendererConfig.flipWinding)
     || rendererCfgChanged(m_rendererConfig.doCulling) || rendererCfgChanged(m_rendererConfig.twoSided)
     || rendererCfgChanged(m_rendererConfig.useSorting) || rendererCfgChanged(m_rendererConfig.numRenderClusterBits)
     || rendererCfgChanged(m_rendererConfig.numTraversalTaskBits))
  {
    rendererChanged = true;

    initRenderer(m_tweak.renderer);
  }
  else if(m_renderer && frameBufferChanged)
  {
    m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
    m_rendererFboChangeID = m_resources.m_fboChangeID;
  }

  bool hadChange = shaderChanged || memcmp(&m_tweakLast, &m_tweak, sizeof(m_tweak))
                   || memcmp(&m_rendererConfigLast, &m_rendererConfig, sizeof(m_rendererConfig))
                   || memcmp(&m_sceneConfigLast, &m_sceneConfig, sizeof(m_sceneConfig))
                   || memcmp(&m_streamingConfigLast, &m_streamingConfig, sizeof(m_streamingConfig))
                   || memcmp(&m_sceneGridConfigLast, &m_sceneGridConfig, sizeof(m_sceneGridConfig));
  m_tweakLast           = m_tweak;
  m_rendererConfigLast  = m_rendererConfig;
  m_streamingConfigLast = m_streamingConfig;
  m_sceneConfigLast     = m_sceneConfig;
  m_sceneGridConfigLast = m_sceneGridConfig;

  if(hadChange)
  {
    m_equalFrames = 0;
    if(m_tweak.autoResetTimers)
    {
      m_info.profilerManager->resetFrameSections(8);
    }
  }
}

void LodClusters::onRender(VkCommandBuffer cmd)
{
  double time = m_clock.getSeconds();

  m_resources.beginFrame(m_app->getFrameCycleIndex());

  m_frameConfig.windowSize = m_windowSize;
  m_frameConfig.hbaoActive = false;

  if(m_renderer)
  {
    uint32_t width  = m_windowSize.width;
    uint32_t height = m_windowSize.height;

    if(m_rendererFboChangeID != m_resources.m_fboChangeID)
    {
      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }

    m_frameConfig.hbaoActive = m_tweak.hbaoActive && m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD;

    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;
    if(m_frames && !m_frameConfig.freezeCulling)
    {
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    int supersample = m_tweak.supersample;

    uint32_t renderWidth  = width * m_tweak.supersample;
    uint32_t renderHeight = height * m_tweak.supersample;

    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;

    {
      frameConstants.visFilterClusterID  = ~0;
      frameConstants.visFilterInstanceID = ~0;
    }

    frameConstants.bgColor     = m_resources.m_bgColor;
    frameConstants.flipWinding = m_rendererConfig.flipWinding ? 1 : 0;
    if(m_rendererConfig.twoSided)
    {
      frameConstants.flipWinding = 2;
    }

    frameConstants.viewport    = glm::ivec2(renderWidth, renderHeight);
    frameConstants.viewportf   = glm::vec2(renderWidth, renderHeight);
    frameConstants.supersample = m_tweak.supersample;
    frameConstants.nearPlane   = m_info.cameraManipulator->getClipPlanes().x;
    frameConstants.farPlane    = m_info.cameraManipulator->getClipPlanes().y;
    frameConstants.wUpDir      = m_info.cameraManipulator->getUp();

    frameConstants.fov = glm::radians(m_info.cameraManipulator->getFov());

    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(m_info.cameraManipulator->getFov()), float(width) / float(height),
                                                 frameConstants.nearPlane, frameConstants.farPlane);
    projection[1][1] *= -1;

    glm::mat4 view  = m_info.cameraManipulator->getViewMatrix();
    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrix  = projection * view;
    frameConstants.viewProjMatrixI = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewMatrix      = view;
    frameConstants.viewMatrixI     = viewI;
    frameConstants.projMatrix      = projection;
    frameConstants.projMatrixI     = glm::inverse(projection);

    glm::mat4 viewNoTrans         = view;
    viewNoTrans[3]                = {0.0f, 0.0f, 0.0f, 1.0f};
    frameConstants.skyProjMatrixI = glm::inverse(projection * viewNoTrans);

    glm::vec4 hPos   = projection * glm::vec4(1.0f, 1.0f, -frameConstants.farPlane, 1.0f);
    glm::vec2 hCoord = glm::vec2(hPos.x / hPos.w, hPos.y / hPos.w);
    glm::vec2 dim    = glm::abs(hCoord);

    // helper to quickly get footprint of a point at a given distance
    //
    // __.__hPos (far plane is width x height)
    // \ | /
    //  \|/
    //   x camera
    //
    // here: viewPixelSize / point.w = size of point in pixels
    // * 0.5f because renderWidth/renderHeight represents [-1,1] but we need half of frustum
    frameConstants.viewPixelSize = dim * (glm::vec2(float(renderWidth), float(renderHeight)) * 0.5f) * frameConstants.farPlane;
    // here: viewClipSize / point.w = size of point in clip-space units
    // no extra scale as half clip space is 1.0 in extent
    frameConstants.viewClipSize = dim * frameConstants.farPlane;

    frameConstants.viewPos = frameConstants.viewMatrixI[3];  // position of eye in the world
    frameConstants.viewDir = -viewI[2];

    frameConstants.viewPlane   = frameConstants.viewDir;
    frameConstants.viewPlane.w = -glm::dot(glm::vec3(frameConstants.viewPos), glm::vec3(frameConstants.viewDir));

    frameConstants.wLightPos = frameConstants.viewMatrixI[3];  // place light at position of eye in the world

    {
      // hiz
      m_resources.m_hizUpdate.farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);
      frameConstants.hizSizeMax = m_resources.m_hizUpdate.farInfo.getSizeMax();
    }

    {
      // hbao setup
      auto& hbaoView                    = m_frameConfig.hbaoSettings.view;
      hbaoView.farPlane                 = frameConstants.farPlane;
      hbaoView.nearPlane                = frameConstants.nearPlane;
      hbaoView.isOrtho                  = false;
      hbaoView.projectionMatrix         = projection;
      m_frameConfig.hbaoSettings.radius = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo) * m_tweak.hbaoRadius;

      glm::vec4 hi = frameConstants.projMatrixI * glm::vec4(1, 1, -0.9, 1);
      hi /= hi.w;
      float tanx           = hi.x / fabsf(hi.z);
      float tany           = hi.y / fabsf(hi.z);
      hbaoView.halfFovyTan = tany;
    }

    if(!m_frames)
    {
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    if(m_frames)
    {
      shaderio::FrameConstants frameCurrent = m_frameConfig.frameConstants;

      if(memcmp(&frameCurrent, &m_frameConfig.frameConstantsLast, sizeof(shaderio::FrameConstants)))
        m_equalFrames = 0;
      else
        m_equalFrames++;
    }

    m_renderer->render(cmd, m_resources, *m_renderScene, m_frameConfig, m_profilerGpuTimer);
  }
  else
  {
    m_resources.emptyFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  {
    m_resources.postProcessFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  m_resources.endFrame();

  // signal new semaphore state with this command buffer's submit
  VkSemaphoreSubmitInfo semSubmit = m_resources.m_queueStates.primary.advanceSignalSubmit(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
  m_app->addSignalSemaphore(semSubmit);
  // but also enqueue waits if there are any
  while(!m_resources.m_queueStates.primary.m_pendingWaits.empty())
  {
    m_app->addWaitSemaphore(m_resources.m_queueStates.primary.m_pendingWaits.back());
    m_resources.m_queueStates.primary.m_pendingWaits.pop_back();
  }

  m_lastTime = time;
  m_frames++;
}

void LodClusters::setSceneCamera(const std::filesystem::path& filePath)
{
  nvgui::SetCameraJsonFile(filePath);

  glm::vec3 extent = m_scene->m_bbox.hi - m_scene->m_bbox.lo;
  float     radius = glm::length(extent) * 0.5f;
  glm::vec3 center = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  bool bigScene = m_scene->m_originalInstanceCount > 1000;

  if(!m_scene->m_cameras.empty())
  {
    auto& c = m_scene->m_cameras[0];
    m_info.cameraManipulator->setFov(c.fovy);


    c.eye              = glm::vec3(c.worldMatrix[3]);
    float     distance = glm::length(center - c.eye);
    glm::mat3 rotMat   = glm::mat3(c.worldMatrix);
    c.center           = {0, 0, -distance};
    c.center           = c.eye + (rotMat * c.center);
    c.up               = {0, 1, 0};

    m_info.cameraManipulator->setCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});

    nvgui::SetHomeCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});
    for(auto& cam : m_scene->m_cameras)
    {
      cam.eye            = glm::vec3(cam.worldMatrix[3]);
      float     distance = glm::length(center - cam.eye);
      glm::mat3 rotMat   = glm::mat3(cam.worldMatrix);
      cam.center         = {0, 0, -distance};
      cam.center         = cam.eye + (rotMat * cam.center);
      cam.up             = {0, 1, 0};


      nvgui::AddCamera({cam.eye, cam.center, cam.up, static_cast<float>(glm::degrees(cam.fovy))});
    }
  }
  else
  {
    glm::vec3 up  = {0, 1, 0};
    glm::vec3 dir = {1.0f, bigScene ? 0.33f : 0.75f, 1.0f};

    m_info.cameraManipulator->setLookat(center + dir * (radius * (bigScene ? 0.5f : 1.f)), center, up);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }

  if(bigScene)
  {
    m_info.cameraManipulator->setSpeed(radius * 0.02f);
    m_info.cameraManipulator->setClipPlanes(glm::vec2(0.0001F * radius, 2.0F * radius));
  }
  else
  {
    m_info.cameraManipulator->setSpeed(radius * 2.0f);
    m_info.cameraManipulator->setClipPlanes(glm::vec2(0.01F * radius, 100.0F * radius));
  }
}

float LodClusters::decodePickingDepth(const shaderio::Readback& readback)
{
  if(!isPickingValid(readback))
  {
    return 0.f;
  }
  uint32_t bits = readback._packedDepth0;
  bits ^= ~(int(bits) >> 31) | 0x80000000u;
  float res = *(float*)&bits;
  return 1.f - res;
}

bool LodClusters::isPickingValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}
}  // namespace lodclusters
