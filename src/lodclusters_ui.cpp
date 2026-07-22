/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cinttypes>
#include <filesystem>
#include <chrono>
#include <thread>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/fonts.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/window.hpp>
#include <nvgui/file_dialog.hpp>
#include <fmt/chrono.h>

#include "lodclusters.hpp"

namespace lodclusters {

#define MEMORY_WITH_BINARY_PREFIXES 1

std::string formatMemorySize(size_t sizeInBytes)
{
#if MEMORY_WITH_BINARY_PREFIXES
  static const std::string units[]     = {"B", "KiB", "MiB", "GiB"};
  static const size_t      unitSizes[] = {1, 1024, 1024 * 1024, 1024 * 1024 * 1024};
#else
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};
#endif

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

std::string formatMetric(size_t size)
{
  static const std::string units[]     = {"", "K", "M", "G"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(size < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float fsize = float(size) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", fsize, units[currentUnit]);
}

static const ImColor   kStreamGeometryFillColor(0.07f, 0.9f, 0.06f, 1.0f);
static const ImColor   kStreamGeometryLineColor(0.45f, 1.0f, 0.35f, 1.0f);
static const ImColor   kStreamClasFillColor(0.2f, 0.55f, 1.0f, 1.0f);
static const ImColor   kStreamClasLineColor(0.55f, 0.85f, 1.0f, 1.0f);
static constexpr float kMemoryPlotFillAlpha  = 0.50f;
static constexpr float kMemoryPlotLineWeight = 1.0f;

template <typename T, typename Tcont>
void uiPlot(const std::string& plotName, const std::string& tooltipFormat, const Tcont& data, const T& maxValue, int offset = 0, size_t sizeOverride = 0)
{
  ImVec2 plotSize = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2);
  size_t size     = sizeOverride ? sizeOverride : data.size();

  // Ensure minimum height to avoid overly squished graphics
  plotSize.y = std::max(plotSize.y, ImGui::GetTextLineHeight() * 20);

  const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
  const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;

  if(ImPlot::BeginPlot(plotName.c_str(), plotSize, plotFlags))
  {
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
    ImPlot::SetupAxes(nullptr, "Count", axesFlags, axesFlags);
    ImPlot::SetupAxesLimits(0, double(size), 0, static_cast<double>(maxValue), ImPlotCond_Always);

    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
    ImPlot::PlotLine("", data.data(), (int)size, 1.0, 0.0,
                     ImPlotSpec(ImPlotProp_LineColor, (ImU32)kStreamGeometryLineColor, ImPlotProp_LineWeight,
                                kMemoryPlotLineWeight, ImPlotProp_FillColor, (ImU32)kStreamGeometryFillColor, ImPlotProp_FillAlpha,
                                kMemoryPlotFillAlpha, ImPlotProp_Offset, offset, ImPlotProp_Flags, ImPlotLineFlags_Shaded));

    if(ImPlot::IsPlotHovered())
    {
      ImPlotPoint mouse       = ImPlot::GetPlotMousePos();
      int         mouseOffset = (int(mouse.x)) % (int)size;
      ImGui::BeginTooltip();
      ImGui::Text(tooltipFormat.c_str(), mouseOffset, data[mouseOffset]);
      ImGui::EndTooltip();
    }

    ImPlot::EndPlot();
  }
}

// clasData contains geometry + CLAS accumulated totals for stacked display
void uiPlotStreamingMemory(const std::string&           yAxisLabel,
                           const std::vector<uint32_t>& geometryData,
                           const std::vector<uint32_t>* clasData,
                           uint32_t                     maxValue,
                           int                          offset)
{
  ImVec2 plotSize = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
  size_t size     = geometryData.size();

  plotSize.y = std::max(plotSize.y, ImGui::GetTextLineHeight() * 20);

  const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
  const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;

  if(ImPlot::BeginPlot("Streaming Memory", plotSize, plotFlags))
  {
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
    ImPlot::SetupAxes(nullptr, yAxisLabel.c_str(), axesFlags, axesFlags);
    ImPlot::SetupAxesLimits(0, double(size), 0, static_cast<double>(maxValue), ImPlotCond_Always);

    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);

    // For stacked display: plot CLAS accumulated total first (background), then Geometry on top.
    // CLAS fill covers the range from Geometry to the combined total.
    if(clasData && !clasData->empty())
    {
      ImPlot::PlotLine("CLAS", clasData->data(), (int)size, 1.0, 0.0,
                       ImPlotSpec(ImPlotProp_LineColor, (ImU32)kStreamClasLineColor, ImPlotProp_LineWeight,
                                  kMemoryPlotLineWeight, ImPlotProp_FillColor, (ImU32)kStreamClasFillColor, ImPlotProp_FillAlpha,
                                  1.0f, ImPlotProp_Offset, offset, ImPlotProp_Flags, ImPlotLineFlags_Shaded));
    }

    ImPlot::PlotLine("Geometry", geometryData.data(), (int)size, 1.0, 0.0,
                     ImPlotSpec(ImPlotProp_LineColor, (ImU32)kStreamGeometryLineColor, ImPlotProp_LineWeight,
                                kMemoryPlotLineWeight, ImPlotProp_FillColor, (ImU32)kStreamGeometryFillColor, ImPlotProp_FillAlpha,
                                1.0f, ImPlotProp_Offset, offset, ImPlotProp_Flags, ImPlotLineFlags_Shaded));

    if(ImPlot::IsPlotHovered())
    {
      ImPlotPoint mouse       = ImPlot::GetPlotMousePos();
      int         mouseOffset = (int(mouse.x)) % (int)size;
      ImGui::BeginTooltip();
      ImGui::Text("Geometry: %u %s", geometryData[mouseOffset], yAxisLabel.c_str());
      if(clasData && !clasData->empty())
      {
        uint32_t clasMB =
            (*clasData)[mouseOffset] >= geometryData[mouseOffset] ? (*clasData)[mouseOffset] - geometryData[mouseOffset] : 0;
        ImGui::Text("CLAS: %u %s", clasMB, yAxisLabel.c_str());
      }
      ImGui::EndTooltip();
    }

    ImPlot::EndPlot();
  }
}

static uint32_t getUsagePct(uint64_t requested, uint64_t reserved)
{
  bool     exceeds = requested > reserved;
  uint32_t pct     = uint32_t(double(requested) * 100.0 / double(std::max(reserved, uint64_t(1))));
  // artificially raise pct over 100 to trigger warning
  if(exceeds && pct < 101)
    pct = 101;
  return pct;
}

struct UsagePercentages
{
  uint32_t pctClusters  = 0;
  uint32_t pctTasks     = 0;
  uint32_t pctResident  = 0;
  uint32_t pctBlas      = 0;
  uint32_t pctClasLeft  = 100;
  uint32_t pctGeoMemory = 0;

  void setupPercentages(shaderio::Readback& readback, uint64_t maxRenderClusters, uint64_t maxTraversalTasks, uint64_t maxBlasBuilds)
  {
    pctClusters = getUsagePct(std::max(std::max(readback.numRenderClusters, readback.numRenderClustersSW),
                                       std::max(readback.numRenderClustersAlpha, readback.numRenderClustersAlphaSW)),
                              maxRenderClusters);
    pctTasks    = getUsagePct(readback.numTraversalTasks, maxTraversalTasks);
    pctBlas     = getUsagePct(readback.numBlasBuilds, maxBlasBuilds);
  }

  void setupPercentages(StreamingStats& stats, const StreamingConfig& streamingConfig)
  {
    pctResident = uint32_t(double(stats.residentGroups) * 100.0 / double(stats.maxGroups));
    pctClasLeft = stats.maxClasBytes ? uint32_t(double(stats.maxSizedLeft) * 100.0 / double(stats.maxSizedReserved)) : 100;
    pctGeoMemory = uint32_t(double(stats.usedDataBytes) * 100.0 / double(stats.maxDataBytes));
  }

  const char* getWarning()
  {
    if(pctClusters > 100)
      return "WARNING: Scene: Render clusters limit exceeded";
    if(pctTasks > 100)
      return "WARNING: Scene: Traversal task limit exceeded";
    if(pctResident == 100)
      return "WARNING: Streaming: No resident groups left";
    if(pctClasLeft <= 1)
      return "WARNING: Streaming: Few CLAS groups left";
    if(pctGeoMemory >= 99)
      return "WARNING: Streaming: Little geometry memory left";
    return nullptr;
  }
};

void LodClusters::viewportUI(ImVec2 corner)
{
  ImVec2     mouseAbsPos = ImGui::GetMousePos();
  glm::uvec2 mousePos    = {mouseAbsPos.x - corner.x, mouseAbsPos.y - corner.y};

  m_frameConfig.frameConstants.mousePosition = glm::uvec2(glm::vec2(mousePos) * m_resources.getFramebufferWindow2RenderScale());

  if(m_renderer)
  {
    shaderio::Readback readback;
    m_resources.getReadbackData(readback);

    UsagePercentages pct;
    pct.setupPercentages(readback, m_renderer->getMaxRenderClusters(), m_renderer->getMaxTraversalTasks(),
                         m_renderer->getMaxBlasBuilds());

    if(m_renderScene->useStreaming)
    {
      StreamingStats streamingStats;
      m_renderScene->sceneStreaming.getStats(streamingStats);
      pct.setupPercentages(streamingStats, m_streamingConfig);
    }


    const char* warning = pct.getWarning();

    if(warning)
    {
      ImVec4 warn_color = {0.75f, 0.2f, 0.2f, 1};
      ImVec4 hi_color   = {0.85f, 0.3f, 0.3f, 1};
      ImVec4 lo_color   = {0, 0, 0, 1};

      ImGui::SetWindowFontScale(2.0);

      // poor man's outline
      ImGui::SetCursorPos({7, 7});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 9});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 7});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({7, 9});
      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({8, 8});
      ImGui::TextColored(hi_color, "%s", warning);

      ImGui::SetWindowFontScale(1.0);
    }
  }
}


void LodClusters::loadingUI() {}

void LodClusters::cameraPathUI()
{
  namespace PE = nvgui::PropertyEditor;

  // File: Save / Load the per-scene paths file stored next to the executable
  const std::filesystem::path pathsFile = sceneCameraPathsFile(m_sceneFilePath);
  ImGui::TextUnformatted("File:");
  ImGui::SameLine();
  ImGui::BeginDisabled(pathsFile.empty());
  if(ImGui::SmallButton("Save"))
  {
    if(saveCameraPaths(pathsFile, m_cameraPaths))
      LOGI("Saved %zu camera path(s) to %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(pathsFile).c_str());
    else
      LOGE("Could not write camera paths to %s\n", nvutils::utf8FromPath(pathsFile).c_str());
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Load") && loadCameraPaths(pathsFile, m_cameraPaths))
  {
    m_cameraPathActive = m_cameraPaths.empty() ? -1 : std::min(std::max(m_cameraPathActive, 0), int(m_cameraPaths.size()) - 1);
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::EndDisabled();
  if(pathsFile.empty())
    ImGui::TextDisabled("(load a scene to save/load its paths file)");
  else
    ImGui::TextDisabled("%s", nvutils::utf8FromPath(pathsFile.filename()).c_str());

  ImGui::Spacing();

  // Path: New / Delete / Copy / Paste (Copy/Paste exchange the path string via the clipboard)
  ImGui::TextUnformatted("Path:");
  ImGui::SameLine();
  if(ImGui::SmallButton("New##path"))
  {
    m_cameraPaths.emplace_back();
    m_cameraPathActive   = int(m_cameraPaths.size()) - 1;
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()));
  if(ImGui::SmallButton("Delete##path"))
  {
    m_cameraPaths.erase(m_cameraPaths.begin() + m_cameraPathActive);
    m_cameraPathActive   = m_cameraPaths.empty() ? -1 : std::min(m_cameraPathActive, int(m_cameraPaths.size()) - 1);
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Copy") && m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size()))
    ImGui::SetClipboardText(m_cameraPaths[m_cameraPathActive].getString().c_str());
  ImGui::EndDisabled();
  ImGui::SameLine();
  if(ImGui::SmallButton("Paste"))
  {
    const char* clip = ImGui::GetClipboardText();
    CameraPath  pasted;
    if(clip && pasted.setFromString(clip))
    {
      m_cameraPaths.push_back(std::move(pasted));
      m_cameraPathActive   = int(m_cameraPaths.size()) - 1;
      m_cameraPathEditKey  = -1;
      m_cameraPathPlayback = CAMERA_PATH_STOP;
    }
  }

  // active path selector
  {
    std::string preview = (m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size())) ?
                              fmt::format("Path {} ({} keys)", m_cameraPathActive, m_cameraPaths[m_cameraPathActive].size()) :
                              std::string("<none>");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::BeginCombo("##camerapathsel", preview.c_str()))
    {
      for(int i = 0; i < int(m_cameraPaths.size()); i++)
      {
        if(ImGui::Selectable(fmt::format("Path {} ({} keys)", i, m_cameraPaths[i].size()).c_str(), i == m_cameraPathActive))
        {
          m_cameraPathActive   = i;
          m_cameraPathEditKey  = -1;
          m_cameraPathPlayback = CAMERA_PATH_STOP;
          m_cameraPathU        = 0.0;
        }
      }
      ImGui::EndCombo();
    }
  }

  const bool hasPath = m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size());
  const nvutils::CameraManipulator::Camera& cam = m_info.cameraManipulator->getCamera();

  // Key: New / Update / Delete (captured from the current camera)
  ImGui::TextUnformatted("Key:");
  ImGui::SameLine();
  ImGui::BeginDisabled(!hasPath);
  if(ImGui::SmallButton("New##key") && hasPath)
  {
    CameraPath& keyPath = m_cameraPaths[m_cameraPathActive];
    int insertAt = (m_cameraPathEditKey >= 0 && m_cameraPathEditKey < int(keyPath.keys.size())) ? m_cameraPathEditKey + 1 :
                                                                                                  int(keyPath.keys.size());
    keyPath.keys.insert(keyPath.keys.begin() + insertAt, CameraPathKey{cam.eye, cam.ctr, cam.up, cam.fov});
    m_cameraPathEditKey = insertAt;
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Update##key") && hasPath && m_cameraPathEditKey >= 0
     && m_cameraPathEditKey < int(m_cameraPaths[m_cameraPathActive].keys.size()))
  {
    m_cameraPaths[m_cameraPathActive].keys[m_cameraPathEditKey] = CameraPathKey{cam.eye, cam.ctr, cam.up, cam.fov};
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Delete##key") && hasPath && m_cameraPathEditKey >= 0
     && m_cameraPathEditKey < int(m_cameraPaths[m_cameraPathActive].keys.size()))
  {
    CameraPath& keyPath = m_cameraPaths[m_cameraPathActive];
    keyPath.keys.erase(keyPath.keys.begin() + m_cameraPathEditKey);
    m_cameraPathEditKey = std::min(m_cameraPathEditKey, int(keyPath.keys.size()) - 1);
  }
  ImGui::EndDisabled();

  // keyframe list (always shown, even when empty; selecting previews the camera)
  if(ImGui::BeginListBox("##keys", ImVec2(-FLT_MIN, ImGui::GetTextLineHeightWithSpacing() * 4.5f)))
  {
    if(hasPath)
    {
      const CameraPath& selPath = m_cameraPaths[m_cameraPathActive];
      for(int i = 0; i < int(selPath.keys.size()); i++)
      {
        const CameraPathKey& k = selPath.keys[i];
        if(ImGui::Selectable(
               fmt::format("{:2d}: eye {:.1f} {:.1f} {:.1f}  fov {:.0f}", i, k.eye.x, k.eye.y, k.eye.z, k.fov).c_str(),
               i == m_cameraPathEditKey))
        {
          m_cameraPathEditKey                  = i;
          m_cameraPathPlayback                 = CAMERA_PATH_STOP;
          nvutils::CameraManipulator::Camera c = cam;
          c.eye = k.eye, c.ctr = k.ctr, c.up = k.up, c.fov = k.fov;
          m_info.cameraManipulator->setCamera(c, false);  // animate for a nicer preview
        }
      }
    }
    ImGui::EndListBox();
  }

  if(!hasPath)
    return;

  CameraPath& path = m_cameraPaths[m_cameraPathActive];

  PE::begin("camerapath", ImGuiTableFlags_Resizable);
  PE::Checkbox("Smooth", &path.smooth, "Catmull-Rom interpolation, otherwise piecewise linear");
  PE::Checkbox("Loop", &path.loop, "Real-time playback wraps around");
  {
    float dur = float(path.duration);
    if(PE::InputFloat("Duration (s)", &dur, 0.1f, 1.0f, "%.2f", 0, "Seconds for a full real-time playback"))
      path.duration = std::max(double(dur), 0.0);
  }
  PE::end();

  // real-time playback + scrub
  const bool active = m_cameraPathPlayback != CAMERA_PATH_STOP;
  if(ImGui::Button(active ? "Stop" : "Play"))
  {
    if(active)
    {
      m_cameraPathPlayback = CAMERA_PATH_STOP;
    }
    else
    {
      m_cameraPathPlayback = CAMERA_PATH_REALTIME;
      m_cameraPathStarted  = false;
      if(m_cameraPathU >= 1.0)
        m_cameraPathU = 0.0;
    }
  }
  ImGui::SameLine();
  if(ImGui::Button("Restart"))
  {
    m_cameraPathU       = 0.0;
    m_cameraPathStarted = false;
  }
  ImGui::SameLine();
  {
    float u = float(m_cameraPathU);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::SliderFloat("##scrub", &u, 0.0f, 1.0f, "t = %.3f"))
    {
      m_cameraPathU        = u;
      m_cameraPathPlayback = CAMERA_PATH_STOP;
      applyCameraPathSample(m_cameraPathU);
    }
  }

  // deterministic (fixed-step) playback, same behaviour as --runcamerapath
  PE::begin("camerapathfixed", ImGuiTableFlags_Resizable);
  PE::InputInt("Fixed frames", &m_cameraPathFrames, 1, 16, 0,
               "Number of frames for a full fixed-step (deterministic) traversal, as used by --runcamerapath");
  PE::end();
  m_cameraPathFrames = std::max(m_cameraPathFrames, 1);
  if(ImGui::Button("Run fixed"))
  {
    m_cameraPathFrame    = 0;
    m_cameraPathPlayback = CAMERA_PATH_FIXED;
  }
  ImGui::SameLine();
  if(m_cameraPathPlayback == CAMERA_PATH_FIXED)
    ImGui::Text("frame %d / %d", m_cameraPathFrame, m_cameraPathFrames);
  else
    ImGui::TextDisabled("cmd: --runcamerapath %d %d", m_cameraPathActive, m_cameraPathFrames);
}

void LodClusters::onUIRender()
{
  ImGuiWindow* viewport = ImGui::FindWindowByName("Viewport");

  bool requestCameraRecenter = false;
  bool requestMirrorBox      = false;

  if(m_sceneLoading)
  {
    // Display a modal window when loading assets or other long operation on separated thread
    ImGui::OpenPopup("Busy Info");

    // Position in the center of the main window when appearing
    const ImVec2 win_size(300, 100);
    ImGui::SetNextWindowSize(win_size);
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));

    // Window without any decoration
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
    if(ImGui::BeginPopupModal("Busy Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration))
    {
      // Center text in window
      ImGui::TextDisabled("Please wait ...");
      ImGui::NewLine();
      ImGui::ProgressBar(float(m_sceneProgress) / 100.0f, ImVec2(-1.0f, 0.0f), "Loading Scene");
      ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  if(viewport)
  {
    if(nvgui::isWindowHovered(viewport))
    {
      if(ImGui::IsKeyPressed(ImGuiKey_R, false))
      {
        m_reloadShaders = true;
      }
      if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
      {
        requestCameraRecenter = true;
      }
      if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right) || ImGui::IsKeyPressed(ImGuiKey_M))
      {
        requestMirrorBox = true;
      }

      bool screenShotJpg = ImGui::IsKeyPressed(ImGuiKey_F11, false);
      bool screenShotPng = ImGui::IsKeyPressed(ImGuiKey_F12, false);
      if(screenShotJpg || screenShotPng)
      {
        auto        now      = std::chrono::system_clock::now();
        std::string filename = fmt::format("screenshot_{:%Y_%m_%d_%H_%M_%S}.{}", now, screenShotJpg ? "jpg" : "png");

        VkExtent2D extent = {m_resources.m_frameBuffer.imgColor.extent.width, m_resources.m_frameBuffer.imgColor.extent.height};

        m_app->saveImageToFile(m_resources.m_frameBuffer.imgColor.image, extent, filename, screenShotJpg ? 90 : 100,
                               m_resources.m_frameBuffer.useResolved ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
                                                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      }
    }
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  bool       pickingValid = isPickingValid(readback);
  glm::dvec3 hitPos       = {};
  bool       hitPosValid  = false;
  if(pickingValid)
  {
    float d = decodePickingDepth(readback);
    // reject far plane and beyond (reversed-Z: 0 = far)
    if(d > 0.0f)
    {
      glm::uvec2 mousePos = {m_frameConfig.frameConstants.mousePosition.x, m_frameConfig.frameConstants.mousePosition.y};

      const glm::dmat4 view = m_info.cameraManipulator->getViewMatrix();
      const glm::dmat4 proj = m_frameConfig.frameConstants.projMatrix;

      glm::dvec4 win_norm = {0, 0, m_frameConfig.frameConstants.viewport.x, m_frameConfig.frameConstants.viewport.y};
      hitPosValid         = true;
      hitPos              = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);
    }
  }

  // camera control, recenter
  if((requestCameraRecenter || requestMirrorBox) && pickingValid)
  {
    if(hitPosValid)
    {
      glm::dvec3 eye, center, up;
      m_info.cameraManipulator->getLookat(eye, center, up);

      if(requestCameraRecenter)
      {
        // Set the interest position
        m_info.cameraManipulator->setLookat(eye, hitPos, up, false);
        m_info.cameraManipulator->setSpeed(glm::length(eye - hitPos) * m_tweak.clickSpeedScale);
      }

      if(requestMirrorBox)
      {
        m_frameConfig.frameConstants.useMirrorBox = 1;
        m_frameConfig.frameConstants.wMirrorBox = glm::vec4(hitPos, glm::distance(eye, hitPos) * m_tweak.mirrorBoxScale);
      }
    }
    else
    {
      if(requestMirrorBox)
      {
        m_frameConfig.frameConstants.useMirrorBox = 0;
      }
    }
  }
  else if(requestMirrorBox && !pickingValid)
  {
    m_frameConfig.frameConstants.useMirrorBox = 0;
  }

  ImVec4 text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  ImVec4 warn_color = text_color;
  warn_color.y *= 0.5f;
  warn_color.z *= 0.5f;

  // for emphasized parameter we want to recommend to the user
  const ImVec4 recommendedColor = ImVec4(0.0, 1.0, 0.0, 1.0);
  const ImVec4 changesColor     = ImVec4(1.0f, 1.0f, 0.1f, 1.0f);

  UsagePercentages pct = {};
  if(m_renderer)
  {
    pct.setupPercentages(readback, m_renderer->getMaxRenderClusters(), m_renderer->getMaxTraversalTasks(),
                         m_renderer->getMaxBlasBuilds());
  }

  StreamingStats stats = {};
  if(m_renderScene && m_renderScene->useStreaming)
  {
    m_renderScene->sceneStreaming.getStats(stats);
  }

  namespace PE = nvgui::PropertyEditor;

  if(ImGui::Begin("Settings"))
  {
    ImGui::PushItemWidth(170 * ImGui::GetWindowDpiScale());

    if(ImGui::CollapsingHeader("Scene Modifiers"))  //, nullptr, ImGuiTreeNodeFlags_DefaultOpen ))
    {
      PE::begin("##Scene Complexity", ImGuiTableFlags_Resizable);
      PE::Checkbox("Allow textured materials", &m_sceneLoaderConfig.enableTexturedMaterials);
      PE::InputIntClamped("Max texture MiB", (int*)&m_texturesConfig.maxBudgetMiB, 0, 1024 * 48, 128, 128,
                          ImGuiInputTextFlags_EnterReturnsTrue,
                          "VRAM budget for material textures. 0 disables the limit. Textures reload when changed.");
      PE::Checkbox("Flip faces winding", &m_rendererConfig.flipWinding);
      PE::Checkbox("Disable back-face culling", &m_rendererConfig.forceTwoSided);

      if(PE::treeNode("Render grid settings", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanFullWidth))
      {
        PE::InputInt("Copies", (int*)&m_sceneGridConfig.numCopies, 1, 16, ImGuiInputTextFlags_EnterReturnsTrue,
                     "Instances the entire scene on a grid");
        PE::entry("Position Axis", [&] {
          for(uint32_t i = 0; i < 3; i++)
          {
            ImGui::PushID(i);
            bool used = (m_sceneGridConfig.gridBits & (1 << i)) != 0;

            ImGui::Checkbox("##hidden", &used);
            if(i < 2)
              ImGui::SameLine();
            if(used)
              m_sceneGridConfig.gridBits |= (1 << i);
            else
              m_sceneGridConfig.gridBits &= ~(1 << i);
            ImGui::PopID();
          }
          return false;
        });

        PE::entry("Rotation Axis", [&] {
          for(uint32_t i = 3; i < 6; i++)
          {
            ImGui::PushID(i);
            bool used = (m_sceneGridConfig.gridBits & (1 << i)) != 0;

            ImGui::Checkbox("##hidden", &used);
            if((i % 3) < 2)
              ImGui::SameLine();
            if(used)
              m_sceneGridConfig.gridBits |= (1 << i);
            else
              m_sceneGridConfig.gridBits &= ~(1 << i);
            ImGui::PopID();
          }
          return false;
        });

        PE::Checkbox("Unique geometries", &m_sceneGridConfig.uniqueGeometriesForCopies,
                     "New Instances of the grid also get their own set of geometries, stresses streaming & memory consumption");

        PE::InputFloat("X gap", &m_sceneGridConfig.refShift.x, 0.1f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
        PE::InputFloat("Y gap", &m_sceneGridConfig.refShift.y, 0.1f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
        PE::InputFloat("Z gap", &m_sceneGridConfig.refShift.z, 0.1f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "Instance grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
        PE::InputFloat("Snap angle", &m_sceneGridConfig.snapAngle, 5.0f, 10.f, "%.3f",
                       ImGuiInputTextFlags_EnterReturnsTrue, "If rotation is active snaps angle");
        PE::InputFloat("Min scale", &m_sceneGridConfig.minScale, 0.1f, 1.f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue, "Scale object");
        PE::InputFloat("Max scale", &m_sceneGridConfig.maxScale, 0.1f, 1.f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue, "Scale object");
        PE::treePop();
      }
      PE::end();
    }

    if(ImGui::CollapsingHeader("Rendering", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {

      PE::begin("##Rendering", ImGuiTableFlags_Resizable);
      PE::entry("Renderer", [&]() { return m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer); });
      PE::entry("Super Resolution",
                [&]() { return m_ui.enumCombobox(GUI_SUPERSAMPLE, "sampling", &m_tweak.supersample); });
#if USE_DLSS
      bool        dlssAvailable = false;
      const char* dlssLabel     = "DLSS";
      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        dlssAvailable = m_resources.m_frameBuffer.dlssDenoiser.isAvailable();
        dlssLabel     = "DLSS - RR";
      }
      else if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)
      {
        dlssAvailable = m_resources.m_frameBuffer.dlssUpscaler.isAvailable();
        dlssLabel     = "DLSS - SR";
      }
      ImGui::BeginDisabled(!dlssAvailable);
      {
        PE::entry(dlssLabel, [&]() {
          bool changed = ImGui::Checkbox("Enabled", &m_rendererConfig.useDlss);
          ImGui::SameLine();

          const char* labels[] = {"DLAA", "Quality", "Balanced", "Performance", "Ultra Performance"};
          const NVSDK_NGX_PerfQuality_Value values[] = {
              NVSDK_NGX_PerfQuality_Value_DLAA,
              NVSDK_NGX_PerfQuality_Value_MaxQuality,
              NVSDK_NGX_PerfQuality_Value_Balanced,
              NVSDK_NGX_PerfQuality_Value_MaxPerf,
              NVSDK_NGX_PerfQuality_Value_UltraPerformance,
          };

          int current = 1;
          for(int i = 0; i < IM_ARRAYSIZE(values); i++)
          {
            if(m_rendererConfig.dlssQuality == values[i])
            {
              current = i;
              break;
            }
          }
          if(ImGui::Combo("Quality", &current, labels, IM_ARRAYSIZE(labels)))
          {
            m_rendererConfig.dlssQuality = values[current];
            changed                      = true;
          }
          return changed;
        });
      }
      ImGui::EndDisabled();
#endif
      PE::Text("Render Resolution:", "%d x %d", m_resources.m_frameBuffer.renderSize.width,
               m_resources.m_frameBuffer.renderSize.height);

      ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
      PE::entry("Visualize", [&]() {
        ImGui::PopStyleColor();  // pop text color here so it only applies to the label
        return m_ui.enumCombobox(GUI_VISUALIZE, "visualize", &m_frameConfig.visualize);
      });

      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        PE::Checkbox("Path tracing", &m_rendererConfig.usePathtrace,
                     "Basic path tracer: shade in the ray-generation shader with multi-bounce GI under the "
                     "physical sky (1 spp). Best with DLSS Ray Reconstruction; noisy otherwise.");

        if(m_rendererConfig.usePathtrace)
        {
          PE::SliderInt("Bounces", &m_frameConfig.frameConstants.pathtraceNumBounces, 1, 8);
          PE::SliderFloat("Firefly clamp", &m_frameConfig.frameConstants.pathtraceFireflyClamp, 0.0f, 100.0f, "%.1f");
          PE::Checkbox("Auto-exposure", (bool*)&m_frameConfig.frameConstants.pathtraceAutoExposure);
          PE::SliderFloat("Exposure (EV)", &m_frameConfig.frameConstants.pathtraceExposureBias, -6.0f, 6.0f, "%.2f");
          PE::entry("Tonemap", [&]() {
            const char* items[] = {"Filmic", "ACES", "Uncharted2", "Clip (no curve)"};
            return ImGui::Combo("##pttonemap", &m_frameConfig.frameConstants.pathtraceTonemap, items, IM_ARRAYSIZE(items));
          });
        }
        else
        {
          PE::Checkbox("Cast shadow rays", (bool*)&m_frameConfig.frameConstants.doShadow);
          PE::Checkbox("Ambient occlusion", &m_tweak.hbaoActive);

          if(!m_tweak.hbaoActive)
          {
            m_frameConfig.frameConstants.ambientOcclusionSamples = 0;
          }
          else
          {
            m_frameConfig.frameConstants.ambientOcclusionSamples = m_lastAmbientOcclusionSamples;
          }
        }
      }
      if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)
      {
        PE::Text("HBAO", "");
        PE::Checkbox("Ambient occlusion", &m_tweak.hbaoActive);
      }
      if(PE::treeNode("AO settings"))
      {  // conditional UI, declutters the UI, prevents presenting many sections in disabled state
        if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
        {
          PE::SliderFloat("Radius", &m_frameConfig.frameConstants.ambientOcclusionRadius, 0.001f, 1.f, "%.7f");
          if(PE::SliderInt("Rays", &m_frameConfig.frameConstants.ambientOcclusionSamples, 1, 64))
          {
            if(m_frameConfig.frameConstants.ambientOcclusionSamples)
            {
              m_tweak.hbaoActive = true;
            }
          }
          if(m_frameConfig.frameConstants.ambientOcclusionSamples)
          {
            m_lastAmbientOcclusionSamples = m_frameConfig.frameConstants.ambientOcclusionSamples;
          }
        }
        if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)
        {
          PE::Checkbox("Blur", &m_frameConfig.hbaoSettings.blur);
          PE::InputFloat("Radius", &m_tweak.hbaoRadius, 0.01f, 0, "%.6f");
          PE::InputFloat("Sharpness", &m_frameConfig.hbaoSettings.powerExponent, 1.0f);
          PE::InputFloat("Intensity", &m_frameConfig.hbaoSettings.intensity, 0.1f);
          PE::InputFloat("Bias", &m_frameConfig.hbaoSettings.bias, 0.01f);
        }
        PE::treePop();
      }

      if(PE::treeNode("Other settings"))
      {
        if(m_scene && m_scene->m_hasVertexNormals)
        {
          PE::Checkbox("Facet shading", &m_tweak.facetShading);
        }
        if(m_resources.m_supportsBarycentrics || m_resources.m_supportsClusterRaytracing)
        {
          PE::Checkbox("Wireframe", (bool*)&m_frameConfig.frameConstants.doWireframe);
        }
        PE::Checkbox("Instance BBoxes", &m_frameConfig.showInstanceBboxes);
        PE::Checkbox("Cluster BBoxes", &m_frameConfig.showClusterBboxes);
        PE::Checkbox("Reflective box", (bool*)&m_frameConfig.frameConstants.useMirrorBox);
        PE::treePop();
      }

      PE::end();
    }
    if(ImGui::CollapsingHeader("Traversal", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("##TraversalSpecifics", ImGuiTableFlags_Resizable);
      PE::InputIntClamped("Max tasks (bits)", (int*)&m_rendererConfig.numTraversalTaskBits, 8, 25, 1, 1,
                          ImGuiInputTextFlags_EnterReturnsTrue);
      PE::InputIntClamped("Max clusters (bits)", (int*)&m_rendererConfig.numRenderClusterBits, 8, 25, 1, 1,
                          ImGuiInputTextFlags_EnterReturnsTrue,
                          "Maximum clusters that can be enqueued per-frame in bits. For raster this equals rendered clusters, for ray tracing its BLAS input.");
      PE::InputFloat("LoD pixel error", &m_frameConfig.lodPixelError, 0.25f, 0.25f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
      PE::Checkbox("Adaptive error", &m_frameConfig.adaptiveError, "alters pixel error based on streaming load");
      if(m_frameConfig.adaptiveError && m_renderer)
      {
        PE::Text("LoD pixel error (used)", fmt::format("{}", m_renderer->getLodError()));
      }

      m_frameConfig.lodPixelError = std::max(0.001f, m_frameConfig.lodPixelError);

      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        ImGui::BeginDisabled(!m_rendererConfig.useCulling);
        PE::InputFloat("Culled error scale", &m_frameConfig.culledErrorScale, 1.f, 1.f, "%.3f",
                       ImGuiInputTextFlags_EnterReturnsTrue, "scale the pixel error for occluded objects in ray tracing");
        ImGui::EndDisabled();

        m_frameConfig.culledErrorScale = std::max(1.0f, m_frameConfig.culledErrorScale);

        PE::Checkbox("Blas Sharing", &m_rendererConfig.useBlasSharing, "shares blas for instances further away that can use it safely");

        if(PE::treeNode("Blas Sharing settings"))
        {
          ImGui::BeginDisabled(!m_rendererConfig.useBlasSharing);
          PE::Checkbox("Blas Caching", &m_rendererConfig.useBlasCaching,
                       "(only when streaming) builds a cached blas depending on highest fully resident lod level.");
          PE::Checkbox("Blas Merging", &m_rendererConfig.useBlasMerging,
                       "(only when streaming) builds a merged blas for closer instances. Guarantees only 2 dynamic blas per geometry.");
          PE::Checkbox("Push culled lod", &m_frameConfig.sharingPushCulled, "culled instances artificially pushed by one lod level");
          PE::InputIntClamped("Shared tail levels", (int*)&m_frameConfig.sharingEnabledLevels, 0, 32, 1, 1,
                              ImGuiInputTextFlags_EnterReturnsTrue,
                              "Sharing may be used in the last N levels of the instance geometry");
          PE::InputIntClamped("Tolerant tail levels", (int*)&m_frameConfig.sharingTolerantLevels, 0, 32, 1, 1,
                              ImGuiInputTextFlags_EnterReturnsTrue,
                              "Share BLAS despite a lod level mismatch in the last N levels of the instance geometry");
          PE::InputIntClamped("Cached tail levels", (int*)&m_frameConfig.cachingEnabledLevels, 0, 32, 1, 1,
                              ImGuiInputTextFlags_EnterReturnsTrue,
                              "Caching may be used in the last N levels of the instance geometry");
          ImGui::EndDisabled();
          PE::treePop();
        }
      }
      if(PE::treeNode("Other settings"))
      {
        PE::Checkbox("Persistent Traversal Kernel", &m_rendererConfig.usePersistentTraversal);
        PE::Checkbox("Instance Sorting", &m_rendererConfig.useSorting);
        PE::Checkbox("Enqueued Statistics", &m_rendererConfig.useRenderStats,
                     "Adds additional atomic counters for statistics, impacts performance");
        PE::Checkbox("Culling (Occlusion & Frustum)", &m_rendererConfig.useCulling);
        ImGui::BeginDisabled(!m_rendererConfig.useCulling);
        PE::Checkbox("Freeze Culling", &m_frameConfig.freezeCulling);
        PE::Checkbox("Freeze LoD", &m_frameConfig.freezeLoD);
        if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)
        {
          PE::Checkbox("Use TwoPass Culling", (bool*)&m_rendererConfig.useTwoPassCulling,
                       "Use two pass culling in rasterization, otherwise uses only last frame's hiz");
          ImGui::EndDisabled();
          ImGui::BeginDisabled(!(!m_rendererConfig.useEXTmeshShader && m_rendererConfig.useCulling && m_resources.m_supportsMeshShaderNV));
          PE::Checkbox("Use Primitive Culling", (bool*)&m_rendererConfig.usePrimitiveCulling, "Use primitive culling in NV mesh shader");
          ImGui::EndDisabled();
          ImGui::BeginDisabled(!((m_frameConfig.visualize == VISUALIZE_VIS_BUFFER || m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY)
                                 && m_rendererConfig.useCulling));
          PE::Checkbox("Allow SW-Raster", (bool*)&m_rendererConfig.useComputeRaster,
                       "Allows use of compute-shader based rasterization (if visualize == visibility buffer / depth only)");
          PE::InputFloat("SW-Raster threshold", &m_frameConfig.swRasterThreshold, 1.0f, 1.0f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                         "cluster uses SW-Raster if its longest edge has less than the specified projected pixels");
          ImGui::EndDisabled();
        }
        else
        {
          PE::Checkbox("Force Invisible Culling", (bool*)&m_rendererConfig.useForcedInvisibleCulling,
                       "Even ray tracing will cull based on primary visibility alone. Warning BLAS Sharing techniques may cause artifacts.");
          ImGui::EndDisabled();
        }
        PE::treePop();
      }
      PE::end();

      ImGui::Separator();

      if(m_rendererConfig.useRenderStats)
      {
        const bool hasAlphaMask = m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD && m_scene && m_scene->m_hasAlphaMask;
        const bool hasSW = m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD && m_rendererConfig.useComputeRaster;

        struct RenderStatTrack
        {
          const char* header;
          uint32_t    clusters;
          uint64_t    tris;
          uint64_t    raster;
        };
        RenderStatTrack tracks[4];
        int             trackCount = 0;
        tracks[trackCount++] = {"Default", readback.numRenderedClusters, readback.numRenderedTriangles, readback.numRasteredTriangles};
        if(hasAlphaMask)
        {
          tracks[trackCount++] = {"Alpha", readback.numRenderedClustersAlpha, readback.numRenderedTrianglesAlpha,
                                  readback.numRasteredTrianglesAlpha};
        }
        if(hasSW)
        {
          tracks[trackCount++] = {"SW", readback.numRenderedClustersSW, readback.numRenderedTrianglesSW, readback.numRasteredTrianglesSW};
        }
        if(hasAlphaMask && hasSW)
        {
          tracks[trackCount++] = {"Alpha SW", readback.numRenderedClustersAlphaSW, readback.numRenderedTrianglesAlphaSW,
                                  readback.numRasteredTrianglesAlphaSW};
        }

        if(ImGui::BeginTable("##Render stats", 1 + trackCount, ImGuiTableFlags_RowBg))
        {
          ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 80.0f * ImGui::GetWindowDpiScale());
          for(int i = 0; i < trackCount; i++)
          {
            ImGui::TableSetupColumn(tracks[i].header, ImGuiTableColumnFlags_WidthStretch);
          }
          ImGui::TableHeadersRow();

          static const char* kNoTrack = "\xe2\x80\x94";

          uint64_t sumClusters = 0;
          uint64_t sumTris     = 0;
          uint64_t sumRaster   = 0;
          for(int i = 0; i < trackCount; i++)
          {
            sumClusters += uint64_t(tracks[i].clusters);
            sumTris += tracks[i].tris;
            sumRaster += tracks[i].raster;
          }

          const auto startMetricRow = [](const char* label) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(label);
          };
          const auto startPctRow = []() {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("");
          };
          const auto nextCellText = [](const char* s) {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(s);
          };
          const auto nextCellFmtMetric = [](size_t v) {
            ImGui::TableNextColumn();
            const std::string t = formatMetric(v);
            ImGui::TextUnformatted(t.c_str());
          };
          const auto nextCellPct = [](uint64_t sum, uint64_t numer, const char* emptyCell) {
            ImGui::TableNextColumn();
            if(sum == 0)
            {
              ImGui::TextUnformatted(emptyCell);
            }
            else
            {
              ImGui::Text("%.0f %%", 100.0 * double(numer) / double(sum));
            }
          };

          startMetricRow("Tasks");
          for(int i = 0; i < trackCount; i++)
          {
            if(i == 0)
            {
              nextCellFmtMetric(size_t(readback.numTraversedTasks));
            }
            else
            {
              nextCellText(kNoTrack);
            }
          }

          startMetricRow("Clusters");
          for(int i = 0; i < trackCount; i++)
          {
            nextCellFmtMetric(size_t(tracks[i].clusters));
          }
          startPctRow();
          for(int i = 0; i < trackCount; i++)
          {
            nextCellPct(sumClusters, uint64_t(tracks[i].clusters), kNoTrack);
          }

          startMetricRow("Tri / Cluster");
          for(int i = 0; i < trackCount; i++)
          {
            ImGui::TableNextColumn();
            const uint32_t c = tracks[i].clusters;
            const uint64_t t = tracks[i].tris;
            if(c > 0)
            {
              ImGui::Text("%.1f", double(t) / double(c));
            }
            else
            {
              ImGui::TextUnformatted("N/A");
            }
          }

          startMetricRow("Triangles");
          for(int i = 0; i < trackCount; i++)
          {
            nextCellFmtMetric(size_t(tracks[i].tris));
          }
          startPctRow();
          for(int i = 0; i < trackCount; i++)
          {
            nextCellPct(sumTris, tracks[i].tris, kNoTrack);
          }

          startMetricRow("Rastered Tri");
          for(int i = 0; i < trackCount; i++)
          {
            nextCellFmtMetric(size_t(tracks[i].raster));
          }
          startPctRow();
          for(int i = 0; i < trackCount; i++)
          {
            nextCellPct(sumRaster, tracks[i].raster, kNoTrack);
          }

          ImGui::EndTable();
        }
      }
    }

    if(ImGui::CollapsingHeader("Clusters & LoDs generation", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::Text("Applying changes can take significant time");

      PE::begin("##Clusters", ImGuiTableFlags_Resizable);
      if(PE::entry("Cluster/meshlet size",
                   [&]() { return m_ui.enumCombobox(GUI_MESHLET, "##cluster", &m_tweak.clusterConfig); }))
      {
        setFromClusterConfig(m_sceneConfigEdit, m_tweak.clusterConfig);
      }

      if(PE::treeNode("Compression settings"))
      {
        PE::Checkbox("Enable compression", &m_sceneConfigEdit.useCompressedData, "Lowers cache file size, can speed up streaming");
        PE::InputIntClamped("POS Mantissa drop bits", (int*)&m_sceneConfigEdit.compressionPosDropBits, 0, 22, 1, 1,
                            ImGuiInputTextFlags_EnterReturnsTrue,
                            "position number of mantissa bits to drop (zeroed) to improve compression");
        PE::InputIntClamped("TC Mantissa drop bits", (int*)&m_sceneConfigEdit.compressionTexDropBits, 0, 22, 1, 1,
                            ImGuiInputTextFlags_EnterReturnsTrue,
                            "texcoord number of mantissa bits to drop (zeroed) to improve compression");
        PE::treePop();
      }

      if(PE::treeNode("Other settings"))
      {
        PE::InputIntClamped("LoD group size", (int*)&m_sceneConfigEdit.clusterGroupSize, 8, SHADERIO_MAX_GROUP_CLUSTERS,
                            1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                            "number of clusters that make a lod group. Their triangles are decimated together and they share a common error property");
        PE::InputIntClamped("Preferred node width", (int*)&m_sceneConfigEdit.preferredNodeWidth, 4,
                            SHADERIO_MAX_NODE_CHILDREN, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                            "number of children a lod node should have (max is always 32).");
        PE::Checkbox("Prefer ray tracing (RT)", &m_sceneConfigEdit.meshoptPreferRayTracing,
                     "Configures meshoptimizer's lod cluster builder to prefer ray tracing over rasterization.");
        PE::InputFloat("RT fill weight", &m_sceneConfigEdit.meshoptFillWeight, 0, 0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "If ray tracing is preferred, influences weight between SAH optimized (towards zero), or filling clusters (higher value).");
        PE::InputFloat("RA split factor", &m_sceneConfigEdit.meshoptSplitFactor, 0, 0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "If raster is preferred, influences the maximum size of a cluster prior splitting it up.");
        PE::Checkbox("Mesh Multi-Materials", &m_sceneConfigEdit.enableMultiMaterials, "Allows a mesh to have multiple materials.");
        PE::entry("Enabled Attributes", [&] {
          for(uint32_t i = 0; i < 4; i++)
          {
            ImGui::PushID(i);
            uint32_t bit  = (1 << i);
            bool     used = (m_sceneConfigEdit.enabledAttributes & bit) != 0;

            const char* what = "error";

            switch(bit)
            {
              case shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL:
                what = "NRM";
                break;
              case shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT:
                what = "TAN";
                break;
              case shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0:
                what = "TEX 0";
                break;
              case shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1:
                what = "TEX 1";
                break;
            }

            ImGui::Checkbox(what, &used);
            if((i % 2) < 1)
              ImGui::SameLine();
            if(used)
              m_sceneConfigEdit.enabledAttributes |= bit;
            else
              m_sceneConfigEdit.enabledAttributes &= ~bit;
            ImGui::PopID();
          }
          return false;
        });

        PE::treePop();
      }

      if(PE::treeNode("Mesh error settings"))
      {
        PE::InputFloat("Error merge previous", &m_sceneConfigEdit.lodErrorMergePrevious, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "Mesh error propagation: scales previous lod error before combining it with the current error to compute the group error as max(previous_error * factor, error).");
        PE::InputFloat("Error merge additive", &m_sceneConfigEdit.lodErrorMergeAdditive, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "Mesh error propagation: adds scaled current error to the group error after the maximum computation.");
        PE::InputFloat("Error edge limit", &m_sceneConfigEdit.lodErrorEdgeLimit, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "Mesh error: limit error by edge length, aiming to remove subpixel triangles even if the attribute error is high");
        PE::InputFloat("Normal weight", &m_sceneConfigEdit.simplifyNormalWeight, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "How much to weight this attribute for the error metric. 0 Disables");
        PE::InputFloat("TexCoord weight", &m_sceneConfigEdit.simplifyTexCoordWeight, 0, 0, "%.3f",
                       ImGuiInputTextFlags_EnterReturnsTrue, "How much to weight this attribute for the error metric. 0 Disables");
        PE::InputFloat("Tangent weight", &m_sceneConfigEdit.simplifyTangentWeight, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "How much to weight this attribute for the error metric. 0 Disables");
        PE::InputFloat("BiTangent Sign weight", &m_sceneConfigEdit.simplifyTangentSignWeight, 0, 0, "%.3f",
                       ImGuiInputTextFlags_EnterReturnsTrue, "How much to weight this attribute for the error metric. 0 Disables");
        PE::InputFloat("Material weight", &m_sceneConfigEdit.simplifyMaterialWeight, 0, 0, "%.3f",
                       ImGuiInputTextFlags_EnterReturnsTrue, "How much to weight this attribute for the error metric. 0 Disables");
        m_sceneConfigEdit.lodErrorMergePrevious = std::max(1.0f, m_sceneConfigEdit.lodErrorMergePrevious);
        m_sceneConfigEdit.lodErrorMergeAdditive = std::max(0.0f, m_sceneConfigEdit.lodErrorMergeAdditive);
        PE::treePop();
      }

      bool hasChanges = memcmp(&m_sceneConfigEdit, &m_sceneConfig, sizeof(m_sceneConfigEdit)) != 0;

      ImGui::BeginDisabled(!hasChanges);
      if(hasChanges)
      {
        ImGui::PushStyleColor(ImGuiCol_Text, changesColor);
      }

      ImVec2 buttonSize = {100.0f * ImGui::GetWindowDpiScale(), 20 * ImGui::GetWindowDpiScale()};
      if(PE::entry("Operations", [&] { return ImGui::Button("Apply Changes", buttonSize); }, "Applying changes triggers reload and processing of the scene"))
      {
        m_sceneConfig = m_sceneConfigEdit;
      }
      if(hasChanges)
      {
        ImGui::PopStyleColor();
      }
      if(PE::entry("", [&] { return ImGui::Button("Reset Changes", buttonSize); }, "Resets the current edits"))
      {
        m_sceneConfigEdit     = m_sceneConfig;
        m_tweak.clusterConfig = findSceneClusterConfig(m_sceneConfig);
      }
      ImGui::EndDisabled();

      PE::end();


      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        ImGui::Separator();
        PE::begin("##CLAS", ImGuiTableFlags_Resizable);
        PE::InputIntClamped("CLAS Mantissa drop bits", (int*)&m_streamingConfig.clasPositionTruncateBits, 0, 22, 1, 1,
                            ImGuiInputTextFlags_EnterReturnsTrue,
                            "number of mantissa bits to drop (zeroed) to reduce memory consumption");
        PE::entry("CLAS build mode",
                  [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##clasbuild", &m_streamingConfig.clasBuildFlags); });
        PE::end();
      }
    }


    if(m_renderScene && ImGui::CollapsingHeader("Streaming", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("##Streaming", ImGuiTableFlags_Resizable);
      if(m_renderSceneCanPreload)
      {
        ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
        PE::Checkbox("Enable", &m_tweak.useStreaming);
        ImGui::PopStyleColor();
      }

      ImGui::BeginDisabled(m_renderScene == nullptr);
      if(PE::entry("Streaming state", [&] { return ImGui::Button("Reset"); }, "resets the streaming state"))
      {
        m_renderScene->streamingReset();
      }
      ImGui::EndDisabled();


      PE::InputIntClamped("Max Resident Groups", (int*)&m_streamingConfig.maxGroups,
                          uint32_t(m_scene ? m_scene->getActiveGeometryCount() : 1024 * 1024), 1024 * 1024, 128, 128,
                          ImGuiInputTextFlags_EnterReturnsTrue);

      PE::InputIntClamped("Max Geometry MiB", (int*)&m_streamingConfig.maxGeometryMegaBytes, 128, 1024 * 48, 128, 128,
                          ImGuiInputTextFlags_EnterReturnsTrue);
      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        PE::InputIntClamped("Max CLAS MiB", (int*)&m_streamingConfig.maxClasMegaBytes, 256, 1024 * 48, 128, 128,
                            ImGuiInputTextFlags_EnterReturnsTrue);
        PE::InputIntClamped("Start CLAS MiB", (int*)&m_streamingConfig.startClasMegaBytes, 128, 1024 * 48, 128, 128,
                            ImGuiInputTextFlags_EnterReturnsTrue);
        PE::InputIntClamped("CLAS grow MiB", (int*)&m_streamingConfig.clasGrowMegaBytes, 128, 1024 * 48, 128, 128,
                            ImGuiInputTextFlags_EnterReturnsTrue);

        PE::Checkbox("Persistent CLAS Allocator", &m_streamingConfig.usePersistentClasAllocator,
                     "Use persistent allocation on the device for CLAS memory, otherwise move based compaction");
        if(PE::treeNode("Allocator settings"))
        {
          ImGui::BeginDisabled(!m_streamingConfig.usePersistentClasAllocator);
          PE::InputIntClamped("Granularity shift bits", (int*)&m_streamingConfig.clasAllocatorGranularityShift, 0, 8, 1,
                              1, ImGuiInputTextFlags_EnterReturnsTrue,
                              "CLAS Allocation byte granularity: (CLAS alignment value) < shift");
          PE::InputIntClamped("Sector shift bits", (int*)&m_streamingConfig.clasAllocatorSectorSizeShift, 5, 16, 1, 1,
                              ImGuiInputTextFlags_EnterReturnsTrue,
                              "CLAS Allocation is scanning for free gaps using unused bits is done per sector of (1 << shift) of 32-bit values");
          PE::Text("Sector size", "%d", 1 << m_streamingConfig.clasAllocatorSectorSizeShift);
          ImGui::EndDisabled();
          PE::treePop();
        }
      }

      if(PE::treeNode("Frame settings"))
      {
        PE::InputIntClamped("Unload frame delay", (int*)&m_frameConfig.streamingAgeThreshold, 2, 1024, 1, 1,
                            ImGuiInputTextFlags_EnterReturnsTrue);

        PE::InputIntClamped("Max Group Loads", (int*)&m_streamingConfig.maxPerFrameLoadRequests, 1, 16 * 1024 * 1024,
                            128, 128, ImGuiInputTextFlags_EnterReturnsTrue);
        PE::InputIntClamped("Max Group Unloads", (int*)&m_streamingConfig.maxPerFrameUnloadRequests, 1,
                            16 * 1024 * 1024, 128, 128, ImGuiInputTextFlags_EnterReturnsTrue);

        PE::InputIntClamped("Max Transfer MiB", (int*)&m_streamingConfig.maxTransferMegaBytes, 1, 1024, 1, 2,
                            ImGuiInputTextFlags_EnterReturnsTrue);

        PE::Checkbox("Async transfer", &m_streamingConfig.useAsyncTransfer, "Use asynchronous transfer queue for uploads");
        ImGui::BeginDisabled(!m_streamingConfig.useAsyncTransfer);
        PE::Checkbox("Decoupled transfer", &m_streamingConfig.useDecoupledAsyncTransfer,
                     "Allow asynchronous transfers to take multiple frames");
        ImGui::EndDisabled();
        PE::treePop();
      }

      PE::end();

      ImGui::Separator();

      if(ImGui::BeginTable("Streaming stats", 3, ImGuiTableFlags_RowBg))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 155.0f * ImGui::GetWindowDpiScale());
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Geometry");
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotStore ? warn_color : text_color, "%s", formatMemorySize(stats.usedDataBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotStore ? warn_color : text_color, "%d %%",
                           getUsagePct(stats.usedDataBytes, stats.maxDataBytes));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
        {
          ImGui::Text("CLAS memory");
          ImGui::TableNextColumn();
          ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%s",
                             formatMemorySize(stats.usedClasBytes).c_str());
          ImGui::TableNextColumn();
          ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%d %%",
                             getUsagePct(stats.usedClasBytes, stats.maxClasBytes));
          ImGui::TableNextRow();
          ImGui::TableNextColumn();

          ImGui::Text("CLAS waste");
          ImGui::TableNextColumn();
          ImGui::TextColored(text_color, "%s", formatMemorySize(stats.wastedClasBytes).c_str());
          ImGui::TableNextColumn();
          ImGui::TextColored(text_color, "%d %%", getUsagePct(stats.wastedClasBytes, stats.usedClasBytes));
          ImGui::TableNextRow();
          ImGui::TableNextColumn();

          ImGui::Text("CLAS groups left");
          ImGui::TableNextColumn();
          ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%s",
                             formatMetric(stats.maxSizedLeft).c_str());
          ImGui::TableNextColumn();
          ImGui::TextColored(stats.couldNotAllocateClas ? warn_color : text_color, "%d %%",
                             getUsagePct(stats.maxSizedLeft, stats.maxSizedReserved));
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
        }
        ImGui::Text("Resident groups");
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotAllocateGroup ? warn_color : text_color, "%s",
                           formatMetric(stats.residentGroups).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotAllocateGroup ? warn_color : text_color, "%d %%",
                           getUsagePct(stats.residentGroups, stats.maxGroups));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGui::Text("Resident clusters");
        uint32_t pctClusters = getUsagePct(stats.residentClusters, stats.maxClusters);
        ImGui::TableNextColumn();
        ImGui::TextColored(pctClusters > 99 ? warn_color : text_color, "%s", formatMetric(stats.residentClusters).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(pctClusters > 99 ? warn_color : text_color, "%d %%", pctClusters);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGui::Text("Resident Triangles");
        ImGui::TableNextColumn();
        ImGui::TextColored(text_color, "%s", formatMetric(stats.residentTriangles).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(text_color, "%d %%", pctClusters);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGui::Text("Last Completed Transfer");
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotTransfer ? warn_color : text_color, "%s", formatMemorySize(stats.transferBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.couldNotTransfer ? warn_color : text_color, "%d %%",
                           getUsagePct(stats.transferBytes, stats.maxTransferBytes));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
#if 0
      ImGui::Text("Last Completed Transfers");
      ImGui::TableNextColumn();
      ImGui::TextColored(text_color, "%s", formatMetric(stats.transferCount).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(text_color, "-");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
#endif

        uint32_t pctLoad =
            stats.loadCount == m_streamingConfig.maxPerFrameLoadRequests ?
                100 :
                std::min(99u, uint32_t(float(stats.loadCount) * 100.0f / float(m_streamingConfig.maxPerFrameLoadRequests)));

        ImGui::Text("Last Completed Loads");
        ImGui::TableNextColumn();
        ImGui::TextColored(pctLoad == 100 ? warn_color : text_color, "%s", formatMetric(stats.loadCount).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(pctLoad == 100 ? warn_color : text_color, "%d %%", pctLoad);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        uint32_t pctUnLoad =
            stats.unloadCount == m_streamingConfig.maxPerFrameUnloadRequests ?
                100 :
                std::min(99u, uint32_t(float(stats.unloadCount) * 100.0f / float(m_streamingConfig.maxPerFrameUnloadRequests)));

        ImGui::Text("Last Completed Unloads");
        ImGui::TableNextColumn();
        ImGui::TextColored(pctUnLoad == 100 ? warn_color : text_color, "%s", formatMetric(stats.unloadCount).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(pctUnLoad == 100 ? warn_color : text_color, "%d %%", pctUnLoad);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        uint32_t pctUncompleted =
            stats.uncompletedLoadCount == m_streamingConfig.maxPerFrameLoadRequests ?
                100 :
                std::min(99u, uint32_t(float(stats.unloadCount) * 100.0f / float(m_streamingConfig.maxPerFrameUnloadRequests)));

        ImGui::Text("Last Uncompleted Loads");
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.uncompletedLoadCount ? warn_color : text_color, "%s",
                           formatMetric(stats.uncompletedLoadCount).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(stats.uncompletedLoadCount ? warn_color : text_color, "%d %%", pctUncompleted);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGui::EndTable();
      }
    }
  }
  ImGui::End();

  Renderer::ResourceUsageInfo resourceActual = m_renderer ? m_renderer->getResourceUsage(false) : Renderer::ResourceUsageInfo();
  Renderer::ResourceUsageInfo resourceReserved = m_renderer ? m_renderer->getResourceUsage(true) : Renderer::ResourceUsageInfo();

  if(ImGui::Begin("Streaming memory"))
  {
    const uint32_t maxSlots = 512;
    if(m_streamGeometryHistogram.empty() == m_tweak.useStreaming)
    {
      m_streamHistogramMax    = 0;
      m_streamHistogramOffset = 0;
      m_streamGeometryHistogram.resize(m_tweak.useStreaming ? maxSlots : 0, 0);
      m_streamClasHistogram.resize(m_tweak.useStreaming ? maxSlots : 0, 0);
    }

    if(m_renderScene && !m_streamGeometryHistogram.empty())
    {
      m_renderScene->sceneStreaming.getStats(stats);

#if MEMORY_WITH_BINARY_PREFIXES
      size_t divisor = 1024 * 1024;
#define MEMORY_MB "MiB"
#else
      size_t divisor = 1000000;
#define MEMORY_MB "MB"
#endif

      uint32_t mbGeometry = uint32_t((stats.usedDataBytes + divisor - 1) / divisor);
      uint32_t mbClas     = uint32_t((stats.usedClasBytes + divisor - 1) / divisor);
      uint32_t mbCombined = mbGeometry;

      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        mbCombined = mbGeometry + mbClas;
      }

      m_streamHistogramMax = std::max(uint32_t(double(mbCombined) * 1.1), m_streamHistogramMax);

      m_streamHistogramOffset = (m_streamHistogramOffset + 1) % maxSlots;
      m_streamGeometryHistogram[(m_streamHistogramOffset + maxSlots - 1) % maxSlots] = mbGeometry;
      m_streamClasHistogram[(m_streamHistogramOffset + maxSlots - 1) % maxSlots]     = mbCombined;

      const std::vector<uint32_t>* clasHistogram =
          (m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD) ? &m_streamClasHistogram : nullptr;
      uiPlotStreamingMemory(std::string(MEMORY_MB), m_streamGeometryHistogram, clasHistogram, m_streamHistogramMax,
                            m_streamHistogramOffset);
    }
  }
  ImGui::End();

  if(ImGui::Begin("Statistics"))
  {
    if(m_scene && ImGui::CollapsingHeader("Scene", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::BeginTable("Scene stats", 3, ImGuiTableFlags_None))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Scene", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Triangles");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiTrianglesCountInstanced * m_sceneGridConfig.numCopies).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiTrianglesCount).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Clusters");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiClustersCountInstanced * m_sceneGridConfig.numCopies).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiClustersCount).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Instances");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_instances.size()).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_originalInstanceCount).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Geometries");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->getActiveGeometryCount()).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_originalGeometryCount).c_str());
        ImGui::EndTable();
      }
    }
    if(m_renderer && ImGui::CollapsingHeader("Traversal", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::BeginTable("Traversal stats", 3, ImGuiTableFlags_RowBg))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Requested", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Tasks");
        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctTasks > 100 ? warn_color : text_color, "%s", formatMetric(readback.numTraversalTasks).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctTasks > 100 ? warn_color : text_color, "%d %%", pct.pctTasks);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Clusters");
        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctClusters > 100 ? warn_color : text_color, "%s",
                           formatMetric(readback.numRenderClusters).c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctClusters > 100 ? warn_color : text_color, "%d %%", pct.pctClusters);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("BLAS Builds");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numBlasBuilds).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%d %%", pct.pctBlas);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::EndTable();
      }
    }
    if(m_renderer && ImGui::CollapsingHeader("Memory", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      const bool   hasTextures     = m_renderScene && m_renderScene->sceneTextures.hasTextures();
      const size_t textureMemBytes = hasTextures ? m_renderScene->sceneTextures.getTextureMemBytes() : 0;

      if(ImGui::BeginTable("Memory stats", 3, ImGuiTableFlags_RowBg))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Actual", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if(hasTextures)
        {
          ImGui::Text("Textures");
          ImGui::TableNextColumn();
          ImGui::Text("%s", formatMemorySize(textureMemBytes).c_str());
          ImGui::TableNextColumn();
          ImGui::Text("==");
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
        }
        ImGui::Text("Geometry");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.geometryMemBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceReserved.geometryMemBytes).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("CLAS");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.rtClasMemBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceReserved.rtClasMemBytes).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("BLAS");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.rtBlasMemBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceReserved.rtBlasMemBytes).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("TLAS");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.rtTlasMemBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("==");
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Operations");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.operationsMemBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("==");
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Total");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.getTotalSum() + textureMemBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceReserved.getTotalSum() + textureMemBytes).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::EndTable();
      }
    }

    if(m_scene && ImGui::CollapsingHeader("Model Cluster Stats"))  //, nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::Text("Cluster max triangles: %d", m_scene->m_maxClusterTriangles);
      ImGui::Text("Cluster max vertices: %d", m_scene->m_maxClusterVertices);
      ImGui::Text("Cluster count: %" PRIu64, m_scene->m_totalClustersCount);
      ImGui::Text("Clusters with config (%u) triangles: %u (%.1f%%)", m_scene->m_config.clusterTriangles,
                  m_scene->m_histograms.clusterTriangles[m_scene->m_config.clusterTriangles],
                  float(m_scene->m_histograms.clusterTriangles[m_scene->m_config.clusterTriangles]) * 100.f
                      / float(m_scene->m_totalClustersCount));
      ImGui::Text("Geometry max lod levels: %d", m_scene->m_maxLodLevelsCount);

      uiPlot(std::string("Cluster Triangles Histogram"), std::string("Cluster count with %d triangles: %u"),
             m_scene->m_histograms.clusterTriangles, m_scene->m_histograms.clusterTrianglesMax, 0,
             m_scene->m_config.clusterTriangles + 1);

      uiPlot(std::string("Cluster Vertices Histogram"), std::string("Cluster count with %d vertices: %u"),
             m_scene->m_histograms.clusterVertices, m_scene->m_histograms.clusterVerticesMax, 0,
             m_scene->m_config.clusterVertices + 1);

      uiPlot(std::string("Group Clusters Histogram"), std::string("Group count with %d clusters: %u"),
             m_scene->m_histograms.groupClusters, m_scene->m_histograms.groupClustersMax, 0,
             m_scene->m_config.clusterGroupSize + 1);

      uiPlot(std::string("Node Children Histogram"), std::string("Node count with %d children: %u"),
             m_scene->m_histograms.nodeChildren, m_scene->m_histograms.nodeChildrenMax);

      uiPlot(std::string("LOD Levels Histogram"), std::string("Mesh count with %d LOD levels: %u"),
             m_scene->m_histograms.lodLevels, m_scene->m_histograms.lodLevelsMax, 0, m_scene->m_maxLodLevelsCount + 1);
    }
  }
  ImGui::End();

  if(ImGui::Begin("Misc Settings"))
  {
    if(ImGui::CollapsingHeader("Camera", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      nvgui::CameraWidget(m_info.cameraManipulator, false);
      namespace PE = nvgui::PropertyEditor;
      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::InputFloat("Speed distance factor", &m_tweak.clickSpeedScale, 0, 0, "%.2f", 0,
                     "double click causes speed to be based on this percentage of the distance to hit point");
      PE::end();
    }
    if(ImGui::CollapsingHeader("Camera Paths", nullptr))
    {
      cameraPathUI();
    }

    if(ImGui::CollapsingHeader("Lighting", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      namespace PE = nvgui::PropertyEditor;
      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                      "Raster / ray-trace: mix between flashlight and sun light.\n"
                      "Path tracer: camera flashlight brightness added on top of the sky (1 = sky brightness).");
      PE::end();
      // Path tracing lights from the physical sky; raster / ray tracing use the simple analytic sky.
      // Show the editor for whichever is active. The sun direction is shared across both models (synced
      // in LodClusters::onRender), so switching renderers keeps the sun in place.
      const bool pathtraceActive = m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD && m_rendererConfig.usePathtrace;
      ImGui::Text(pathtraceActive ? "Sun & Sky (physical)" : "Sun & Sky (simple)");
      if(pathtraceActive)
      {
        nvgui::skyPhysicalParameterUI(m_frameConfig.frameConstants.skyPhysical);
      }
      else
      {
        nvgui::skySimpleParametersUI(m_frameConfig.frameConstants.skyParams, "misc");
      }
    }

    if(ImGui::CollapsingHeader("Mirror Box", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      namespace PE = nvgui::PropertyEditor;
      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::Checkbox("Use", (bool*)&m_frameConfig.frameConstants.useMirrorBox);
      PE::InputFloat3("Position", &m_frameConfig.frameConstants.wMirrorBox.x);
      PE::InputFloat("Size", &m_frameConfig.frameConstants.wMirrorBox.w);
      PE::InputFloat("Size distance factor", &m_tweak.mirrorBoxScale, 0, 0, "%.3f", 0,
                     "When event is used, determine mirror box size based on `distance * factor`");
      PE::end();
    }

    if(ImGui::CollapsingHeader("Hit Info", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      uint32_t    instanceID          = -1;
      uint32_t    geometryID          = -1;
      uint32_t    materialID          = -1;
      uint32_t    clusterID           = -1;
      uint32_t    triangleID          = -1;
      const char* materialName        = nullptr;
      const char* geometryName        = nullptr;
      bool        materialAlphaMasked = false;
      bool        materialTwoSided    = false;

      if(m_scene && pickingValid)
      {
        instanceID = readback.instanceId;
        geometryID = m_scene->m_instances[readback.instanceId].geometryID;

        if(m_renderer)
        {
          materialID = m_renderer->getOriginalMaterialID(readback.materialId);
          if(materialID != ~0)
          {
            materialName        = m_scene->m_materialNames[materialID].c_str();
            materialAlphaMasked = m_scene->m_materials[materialID].alphaMasked;
            materialTwoSided    = m_scene->m_materials[materialID].twoSided;
          }
        }

        clusterID    = readback.clusterTriangleId >> 8;
        triangleID   = readback.clusterTriangleId & 0xFF;
        geometryName = m_scene->m_geometryNames[geometryID].c_str();
      }

      ImGui::Text("Instance ID:  %d", instanceID);
      ImGui::Text("Geometry ID:  %d \"%s\"", geometryID, geometryName ? geometryName : "");
      ImGui::Text("Material ID:  %d \"%s\"%s %s", materialID, materialName ? materialName : "",
                  materialAlphaMasked ? " alpha-masked" : "", materialTwoSided ? " two-sided" : "");
      ImGui::Text("Cluster  ID:  %d", clusterID);
      ImGui::Text("Triangle ID:  %d", triangleID);
      ImGui::Text("Position:  [%f, %f, %f]", hitPos.x, hitPos.y, hitPos.z);
    }

    if(ImGui::CollapsingHeader("Advanced", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::SliderFloat("Texture Gradient Scale", &m_frameConfig.frameConstants.texGradScale, 0.0f, 1.0f, "%.3f", 0,
                      "Influence texture gradient in ray tracing and compute rasterization");
      PE::entry(
          "Ray Cone Texture LOD",
          [&]() {
            const char* items[] = {"Gradient (textureGrad)", "Explicit LOD (textureLod)", "Mip 0 (baseline)"};
            return ImGui::Combo("##texlodmode", &m_rendererConfig.textureLodMode, items, IM_ARRAYSIZE(items));
          },
          "Ray/path tracer: how material textures pick their mip level from the ray-cone footprint "
          "(recompiles shaders; raster always uses hardware derivatives)");
      PE::InputIntClamped("Persistent Traversal Threads", (int*)&m_frameConfig.traversalPersistentThreads, 32,
                          256 * 1024, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
      PE::InputInt("Colorize xor", (int*)&m_frameConfig.frameConstants.colorXor);
      PE::Checkbox("Auto reset timer", &m_tweak.autoResetTimers);
      if(m_resources.m_supportsMeshShaderNV)
      {
        PE::Checkbox("Use EXT Mesh shader", &m_rendererConfig.useEXTmeshShader);
      }
      PE::end();
    }
  }
  ImGui::End();

  if(m_showDebugUI)
  {
    if(ImGui::Begin("Debug"))
    {
      if(ImGui::CollapsingHeader("Debug Shader Values", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
      {
        PE::begin("##HiddenID");
        PE::InputInt("dbgInt", (int*)&m_frameConfig.frameConstants.dbgUint, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue);
        PE::InputFloat("dbgFloat", &m_frameConfig.frameConstants.dbgFloat, 0.1f, 1.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        PE::end();

        ImGui::Text(" debugI :  %10d", readback.debugI);
        ImGui::Text(" debugUI:  %10u", readback.debugUI);
        ImGui::Text(" debugU64:  %" PRIX64, readback.debugU64);
        static bool debugFloat = false;
        static bool debugHex   = false;
        static bool debugAll   = false;
        ImGui::Checkbox(" as float", &debugFloat);
        ImGui::SameLine();
        ImGui::Checkbox("hex", &debugHex);
        ImGui::SameLine();
        ImGui::Checkbox("all", &debugAll);
        //ImGui::SameLine();
        //bool     doPrint = ImGui::Button("print");
        uint32_t count = debugAll ? 64 : 32;

        if(ImGui::BeginTable("##Debug", 4, ImGuiTableFlags_BordersOuter))
        {
          ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 32);
          ImGui::TableSetupColumn("A", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("B", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("C", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableHeadersRow();
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          for(uint32_t i = 0; i < count; i++)
          {
            ImGui::Text("%2d", i);
            if(debugFloat)
            {
              ImGui::TableNextColumn();
              ImGui::Text("%f", *(float*)&readback.debugA[i]);
              ImGui::TableNextColumn();
              ImGui::Text("%f", *(float*)&readback.debugB[i]);
              ImGui::TableNextColumn();
              ImGui::Text("%f", *(float*)&readback.debugC[i]);
            }
            else if(debugHex)
            {
              ImGui::TableNextColumn();
              ImGui::Text("%X", readback.debugA[i]);
              ImGui::TableNextColumn();
              ImGui::Text("%X", readback.debugB[i]);
              ImGui::TableNextColumn();
              ImGui::Text("%X", readback.debugC[i]);
            }
            else
            {
              ImGui::TableNextColumn();
              ImGui::Text("%d", readback.debugA[i]);
              ImGui::TableNextColumn();
              ImGui::Text("%d", readback.debugB[i]);
              ImGui::TableNextColumn();
              ImGui::Text("%d", readback.debugC[i]);
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
          }

          ImGui::EndTable();
        }
      }
    }
    ImGui::End();
  }

  handleChanges();

  // Rendered image displayed fully in 'Viewport' window
  if(ImGui::Begin("Viewport"))
  {
    ImVec2 corner = ImGui::GetCursorScreenPos();  // Corner of the viewport
    ImGui::Image((ImTextureID)m_imguiTexture, ImGui::GetContentRegionAvail());
    viewportUI(corner);
  }
  ImGui::End();
}

void LodClusters::onUIMenu()
{
  bool vsync = m_app->isVsync();

  bool doOpenFile        = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O);
  bool doSaveCacheFile   = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S);
  bool doReloadFile      = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_R);
  bool doDeleteCacheFile = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D);
  bool doCloseApp        = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Q);
  bool doToggleVsync     = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_V);

  bool hasCache = m_scene && !m_scene->isMemoryMappedCache() && std::filesystem::exists(m_scene->getCacheFilePath());


  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN "Open", "Ctrl+O"))
    {
      doOpenFile = true;
    }
    if(m_scene)
    {
      if(ImGui::MenuItem(ICON_MS_REFRESH "Reload File", "Ctrl+R"))
      {
        doReloadFile = true;
      }

      if(!m_scene->m_loadedFromCache)
      {
        if(ImGui::MenuItem(ICON_MS_FILE_SAVE "Save Cache", "Ctrl+S"))
        {
          doSaveCacheFile = true;
        }
      }

      if(hasCache)
      {
        if(ImGui::MenuItem(ICON_MS_DELETE "Delete Cache", "Ctrl+D"))
        {
          doDeleteCacheFile = true;
        }
      }
    }
    if(ImGui::MenuItem(ICON_MS_DIRECTORY_SYNC "Reload Shaders", "R"))
    {
      m_reloadShaders = true;
    }
    if(ImGui::MenuItem(ICON_MS_POWER_SETTINGS_NEW "Exit", "Ctrl+Q"))
    {
      doCloseApp = true;
    }

    ImGui::EndMenu();
  }

  if(ImGui::BeginMenu("View"))
  {
    if(ImGui::MenuItem(ICON_MS_BOTTOM_PANEL_OPEN "V-Sync", "Ctrl+Shift+V", &vsync))
    {
      doToggleVsync = true;
    }

    ImGui::EndMenu();
  }

  if(doToggleVsync)
  {
    vsync = !vsync;
    m_app->setVsync(vsync);
  }

  if(doOpenFile)
  {
    std::filesystem::path filePath =
        nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load supported",
                                    "Supported Files|*.gltf;*.glb;*.cfg|glTF(.gltf, .glb)|*.gltf;*.glb|config file(.cfg)|*.cfg");
    if(!filePath.empty())
    {
      onFileDrop(filePath);
    }
  }

  if(m_scene && doReloadFile)
  {
    std::filesystem::path filePath = m_sceneFilePathDropLast;
    onFileDrop(filePath);
  }

  if(m_scene)
  {
    if(!m_scene->m_loadedFromCache && doSaveCacheFile)
    {
      saveCacheFile();
    }

    if(hasCache && doDeleteCacheFile)
    {
      try
      {
        if(std::filesystem::remove(m_scene->getCacheFilePath()))
        {
          LOGI("Cache file deleted successfully.\n");
        }
      }
      catch(const std::filesystem::filesystem_error& e)
      {
        LOGW("Problem deleting cache file: %s\n", e.what());
      }
    }
  }


  if(doCloseApp)
  {
    m_app->close();
  }
}

}  // namespace lodclusters
