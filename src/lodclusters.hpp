/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thread>

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/parameter_sequencer.hpp>
#include <nvvk/context.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvgui/enum_registry.hpp>

#include "renderer.hpp"
#include "camera_path.hpp"
#include "scene_streaming_vis.hpp"

namespace lodclusters {

class LodClusters : public nvapp::IAppElement
{
public:
  enum RendererType
  {
    RENDERER_RASTER_CLUSTERS_LOD,
    RENDERER_RAYTRACE_CLUSTERS_LOD,
  };

  enum ClusterConfig
  {
    CLUSTER_64T_64V,
    CLUSTER_64T_128V,
    CLUSTER_64T_192V,
    CLUSTER_96T_96V,
    CLUSTER_128T_128V,
    CLUSTER_128T_256V,
    CLUSTER_256T_256V,
    NUM_CLUSTER_CONFIGS,
  };

  struct ClusterInfo
  {
    uint32_t      tris;
    uint32_t      verts;
    ClusterConfig cfg;
  };

  static const ClusterInfo s_clusterInfos[NUM_CLUSTER_CONFIGS];

  enum GuiEnums
  {
    GUI_RENDERER,
    GUI_BUILDMODE,
    GUI_SUPERSAMPLE,
    GUI_MESHLET,
    GUI_VISUALIZE,
  };

  enum ScreenshotMode
  {
    SCREENSHOT_OFF,
    SCREENSHOT_WINDOW,
    SCREENSHOT_VIEWPORT
  };

  struct Tweak
  {
    ClusterConfig clusterConfig = CLUSTER_128T_128V;

    RendererType renderer    = RENDERER_RAYTRACE_CLUSTERS_LOD;
    int          supersample = 2;

    bool facetShading = true;
    bool useStreaming = true;

    bool autoResetTimers = false;
    bool autoSharing     = true;

    bool  hbaoActive = true;
    float hbaoRadius = 0.05f;

    float mirrorBoxScale  = 0.2f;
    float clickSpeedScale = 0.33f;

    // rasterization-only solo filters; -1 disables (default)
    int32_t filterInstanceID = -1;
    int32_t filterClusterID  = -1;
  };


  struct ViewPoint
  {
    std::string name;
    glm::mat4   mat;
    float       sceneScale;
    float       fov;
  };

  struct TargetImage
  {
    VkImage     image;
    VkImageView view;
    VkFormat    format;
  };

  struct Info
  {
    nvutils::ProfilerManager*                   profilerManager{};
    nvutils::ParameterRegistry*                 parameterRegistry{};
    nvutils::ParameterParser*                   parameterParser{};
    std::shared_ptr<nvutils::CameraManipulator> cameraManipulator;
  };

  LodClusters(const Info& info);

  ~LodClusters() override { m_info.profilerManager->destroyTimeline(m_profilerTimeline); }

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onUIMenu() override;
  void onUIRender() override;
  void onPreRender() override;
  void onRender(VkCommandBuffer cmd) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onFileDrop(const std::filesystem::path& filename) override;
  void onLastHeadlessFrame() override;

  void setSupportsClusterRaytracing(bool supported) { m_resources.m_supportsClusterRaytracing = supported; }
  void setSupportsBarycentrics(bool supported) { m_resources.m_supportsBarycentrics = supported; }
  void setSupportsMeshShaderNV(bool supported) { m_resources.m_supportsMeshShaderNV = supported; }
  void setSupportsSmBuiltinsNV(bool supported) { m_resources.m_supportsSmBuiltinsNV = supported; }
  bool getShowDebugUI() const { return m_showDebugUI; }

  bool isProcessingOnly() const { return !m_sceneFilePathDropNew.empty() && m_sceneLoaderConfig.processingOnly; }
  void doProcessingOnly();

  void parameterSequenceCallback(const nvutils::ParameterSequencer::State& state);

private:
  // saves the window or the rendered viewport, headless always uses the viewport
  void saveScreenshot(ScreenshotMode mode, const std::string& filename);

  VkExtent2D                 m_windowSize;
  Info                       m_info;
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer{};
  nvapp::Application*        m_app{};

  //////////////////////////////////////////////////////////////////////////

  // key components

  Resources       m_resources;
  FrameConfig     m_frameConfig;
  double          m_lastTime = 0;
  VkDescriptorSet m_imguiTexture{};
  VkSampler       m_imguiSampler{};

  // fragmentation view of the persistent clas allocator, updated while its widget is visible
  StreamingAllocatorVis m_clasAllocatorVis;
  bool                  m_clasAllocatorVisVisible     = false;
  bool                  m_clasAllocatorVisShowGroups  = true;
  bool                  m_clasAllocatorVisHistogram   = false;
  ImVec2                m_clasAllocatorVisDisplaySize = {};

  // toggled from the "View" menu. The dockable panels are on by default, the clas
  // allocator inspector is opt-in and floats over the viewport.
  struct WindowShow
  {
    bool settings        = true;
    bool miscSettings    = true;
    bool statistics      = true;
    bool streamingMemory = true;
    bool clasAllocator   = false;
  };
  WindowShow m_showWindow;

  nvgui::EnumRegistry       m_ui;
  nvutils::PerformanceTimer m_clock;

  bool m_reloadShaders    = false;
  bool m_revealTonemapper = false;
#ifndef NDEBUG
  bool m_showDebugUI = true;
#else
  bool m_showDebugUI = false;
#endif
  int            m_frames                  = 0;
  double         m_animTime                = 0;
  ScreenshotMode m_sequenceScreenshotMode  = SCREENSHOT_OFF;
  ScreenshotMode m_screenshotMode          = SCREENSHOT_VIEWPORT;
  int            m_screenshotFrameInterval = 0;  // save a screenshot every n frames, 0 disables
  std::string    m_screenshotTimeStamp;          // session startup time, keeps screenshots of multiple runs apart

  Tweak m_tweak;
  Tweak m_tweakLast;

  uint32_t m_lastAmbientOcclusionSamples = 0;

  std::unique_ptr<Scene> m_scene;
  std::filesystem::path  m_sceneFilePath;
  std::filesystem::path  m_sceneFilePathDefault;
  std::filesystem::path  m_sceneFilePathDropLast;
  std::filesystem::path  m_sceneFilePathDropNew;
  std::string            m_sceneCacheSuffix = ".nvsngeo";
  SceneLoaderConfig      m_sceneLoaderConfig;
  SceneLoaderConfig      m_sceneLoaderConfigLast;
  SceneConfig            m_sceneConfig;
  SceneConfig            m_sceneConfigLast;
  SceneConfig            m_sceneConfigEdit;
  glm::vec3              m_sceneUpVector = glm::vec3(0, 1, 0);
  SceneGridConfig        m_sceneGridConfig;
  SceneGridConfig        m_sceneGridConfigLast;
  std::atomic_bool       m_sceneLoading = false;
  // progress signals shared with the background loader via m_sceneLoaderConfig.progressInfo
  std::atomic_uint32_t m_sceneCompletedCount = 0;  // items done in the current phase
  std::atomic_uint32_t m_sceneTotalCount     = 0;  // items total in the current phase
  std::atomic_uint32_t m_sceneProgressPhase  = 0;  // current LoadPhase
  // set by the loader thread once textures are loaded into m_renderScenePending; the main thread
  // then promotes it to m_renderScene and finishes the GPU geometry setup in handleChanges.
  std::atomic_bool m_renderSceneGeometryPending = false;
  bool             m_sceneLoadFromConfig        = false;
  std::thread      m_sceneLoadingThread;

  std::string m_cameraString;
  std::string m_cameraStringLast;
  std::string m_cameraStringCommandLine;
  float       m_cameraSpeed = 0;
  //std::filesystem::path  m_cameraFilePath;

  // Fixed camera paths (fly-through), see camera_path.hpp.
  // Defined via `--addcamerapath` (order == index) and/or the UI, and
  // played back either in real-time (UI) or with a fixed number of frames
  // (`--runcamerapath <index> <framecount>`) for deterministic benchmarking.
  enum CameraPathPlayback
  {
    CAMERA_PATH_STOP,      // not playing
    CAMERA_PATH_REALTIME,  // advance by wall-clock time over `duration` seconds
    CAMERA_PATH_FIXED,     // advance one keyframe-step per rendered frame (deterministic)
  };

  std::vector<CameraPath> m_cameraPaths;  // path definitions, index == order added
  bool m_cameraPathsExternal = false;  // paths set via --addcamerapath/--loadcamerapaths (command line, config, or sequence); global, take precedence over per-scene files
  std::filesystem::path m_cameraPathsLoadedScene;  // scene for which the per-scene paths file was last loaded
  int                   m_cameraPathActive   = -1;
  CameraPathPlayback    m_cameraPathPlayback = CAMERA_PATH_STOP;
  bool                  m_cameraPathStarted  = false;  // real-time: first frame has zero delta
  double                m_cameraPathU        = 0.0;    // normalized playback position [0,1]
  int                   m_cameraPathFrames   = 128;    // fixed: total frames for a full traversal
  int                   m_cameraPathFrame    = 0;      // fixed: current frame
  int                   m_cameraPathEditKey  = -1;     // UI: selected keyframe

  std::unique_ptr<RenderScene> m_renderScene;
  // Built on the loader thread (textures only) and promoted to m_renderScene on the main thread once
  // loading completes. Kept separate so the UI/render path never sees a half-constructed RenderScene.
  std::unique_ptr<RenderScene> m_renderScenePending;
  bool                         m_renderSceneCanPreload = false;
  // Can be used to withdraw the permission for handleChanges to implicitly re-attempt the init every frame after an
  // attempt failed. Granted again by the next explicit attempt (config change, scene reload) or by a
  // shader reload.
  bool m_renderSceneInitAllowed = true;

  StreamingConfig m_streamingConfig;
  StreamingConfig m_streamingConfigLast;

  SceneTexturesConfig m_texturesConfig;
  SceneTexturesConfig m_texturesConfigLast;

  // Memory budgets as provided on the command line or via config file.
  // > 0 is an absolute MiB value, < 0 a percentage of the device local
  // heap (-25 == 25 %), 0 keeps the built-in default.
  struct MemoryBudgetArgs
  {
    int32_t maxTransferMegaBytes    = 0;
    int32_t maxGeometryMegaBytes    = 0;
    int32_t maxClasMegaBytes        = 0;
    int32_t startClasMegaBytes      = 0;
    int32_t clasGrowMegaBytes       = 0;
    int32_t maxBlasCachingMegaBytes = 0;
    // textures differ: 0 is a valid value that disables the limit
    int32_t maxTextureMegaBytes = 4096;
  };
  MemoryBudgetArgs m_memoryBudgetArgs;
  MemoryBudgetArgs m_memoryBudgetArgsLast;

  std::unique_ptr<Renderer> m_renderer;
  uint64_t                  m_rendererFboChangeID{};
  RendererConfig            m_rendererConfig;
  RendererConfig            m_rendererConfigLast;

  std::vector<uint32_t> m_streamClasHistogram;
  std::vector<uint32_t> m_streamGeometryHistogram;
  uint32_t              m_streamHistogramMax    = 0;
  int32_t               m_streamHistogramOffset = 0;

  uint32_t m_equalFrames = 0;

  // use by-value copies for flexibility
  void initScene(std::filesystem::path filePath, std::string cacheSuffix, bool configChange);

  void setSceneCamera(const std::filesystem::path& filePath);
  void saveCacheFile();
  void applyMemoryBudgetArgs();

  void deinitScene();
  void postInitNewScene();

  void initRenderScene();
  void initRenderSceneGeometry();
  void deinitRenderScene();

  void initRenderer(RendererType rtype);
  void deinitRenderer();

  void updateImguiImage();

  ClusterConfig findSceneClusterConfig(const SceneConfig& sceneConfig);
  void          setFromClusterConfig(SceneConfig& sceneConfig, ClusterConfig clusterConfig);
  void          updatedSceneGrid();

  void handleChanges();
  void applyCameraString();

  // Camera path (see camera_path.hpp)
  void registerCameraPathParameters();   // `--addcamerapath`, `--loadcamerapaths`, `--runcamerapath`
  void updateCameraPath(double time);    // drive the manipulator during playback (called from onRender)
  bool applyCameraPathSample(double u);  // sample active path at `u` and set the camera; false if unavailable
  void cameraPathUI();
  // Per-scene camera paths file, stored next to the model file and named after
  // it (e.g. "<modeldir>/<scene>.camerapaths.txt")
  std::filesystem::path sceneCameraPathsFile(const std::filesystem::path& scenePath) const;

  float decodePickingDepth(const shaderio::Readback& readback);
  bool  isPickingValid(const shaderio::Readback& readback);

  void viewportUI(ImVec2 corner, ImVec2 imageSize);

  // the individual panels of `onUIRender`, each is a no-op when its window is hidden
  void uiSettings();
  // collapsing sections within the "Settings" panel
  void uiSettingsSceneModifiers();
  void uiSettingsRendering();
  void uiSettingsTraversal();
  void uiSettingsClusterGeneration();
  void uiSettingsStreaming();
  void uiStreamingMemory();
  void uiStatistics();
  void uiMiscSettings(bool pickingValid, const glm::dvec3& hitPos);
  void uiDebug();
  // floating inspector for the persistent clas allocator's memory
  void clasAllocatorUI();

  void loadingUI();

  template <typename T>
  bool sceneChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_sceneConfig);
    assert(offset < sizeof(m_sceneConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_sceneConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool tweakChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    assert(offset < sizeof(m_tweak));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_tweakLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool rendererCfgChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_rendererConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_rendererConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool streamingCfgChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_streamingConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_streamingConfigLast) + offset, sizeof(T)) != 0;
  }
};
}  // namespace lodclusters
