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

#include <cinttypes>
#include <filesystem>
#include <chrono>
#include <thread>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvgui/fonts.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/window.hpp>
#include <nvgui/file_dialog.hpp>

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

template <typename T, typename Tcont>
void uiPlot(const std::string& plotName, const std::string& tooltipFormat, const Tcont& data, const T& maxValue, int offset = 0, size_t sizeOverride = 0)
{
  ImVec2 plotSize = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2);
  size_t size     = sizeOverride ? sizeOverride : data.size();

  // Ensure minimum height to avoid overly squished graphics
  plotSize.y = std::max(plotSize.y, ImGui::GetTextLineHeight() * 20);

  const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
  const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;
  const ImColor         plotColor = ImColor(0.07f, 0.9f, 0.06f, 1.0f);

  if(ImPlot::BeginPlot(plotName.c_str(), plotSize, plotFlags))
  {
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
    ImPlot::SetupAxes(nullptr, "Count", axesFlags, axesFlags);
    ImPlot::SetupAxesLimits(0, double(size), 0, static_cast<double>(maxValue), ImPlotCond_Always);

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
    ImPlot::SetNextFillStyle(plotColor);
    ImPlot::PlotShaded("", data.data(), (int)size, -INFINITY, 1.0, 0.0, 0, offset);
    ImPlot::PopStyleVar();

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
    pctClusters = getUsagePct(std::max(readback.numRenderClusters, readback.numRenderClustersSW), maxRenderClusters);
    pctTasks    = getUsagePct(readback.numTraversalTasks, maxTraversalTasks);
    pctBlas     = getUsagePct(readback.numBlasBuilds, maxBlasBuilds);
  }

  void setupPercentages(StreamingStats& stats, const StreamingConfig& streamingConfig)
  {
    pctResident = uint32_t(double(stats.residentGroups) * 100.0 / double(stats.maxGroups));
    pctClasLeft = stats.reservedClasBytes ? uint32_t(double(stats.maxSizedLeft) * 100.0 / double(stats.maxSizedReserved)) : 100;
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
      if(ImGui::IsKeyDown(ImGuiKey_R))
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
    }
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  bool pickingValid = isPickingValid(readback);
  // camera control, recenter
  if((requestCameraRecenter || requestMirrorBox) && pickingValid)
  {
    glm::uvec2 mousePos = {m_frameConfig.frameConstants.mousePosition.x, m_frameConfig.frameConstants.mousePosition.y};

    const glm::mat4 view = m_info.cameraManipulator->getViewMatrix();
    const glm::mat4 proj = m_frameConfig.frameConstants.projMatrix;

    float d = decodePickingDepth(readback);

    if(d < 1.0F)  // Ignore infinite
    {
      glm::vec4 win_norm     = {0, 0, m_frameConfig.frameConstants.viewport.x, m_frameConfig.frameConstants.viewport.y};
      const glm::vec3 hitPos = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);

      glm::vec3 eye, center, up;
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
      ImGui::BeginDisabled(
          !(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD && m_resources.m_frameBuffer.dlssDenoiser.isAvailable()));
      {
        PE::entry("DLSS - Denoiser", [&]() {
          ImGui::Checkbox("Enabled", &m_rendererConfig.useDlss);
          ImGui::SameLine();
          bool dlaa = m_rendererConfig.dlssQuality == NVSDK_NGX_PerfQuality_Value_DLAA;
          ImGui::Checkbox("DLAA", &dlaa);
          m_rendererConfig.dlssQuality = dlaa ? NVSDK_NGX_PerfQuality_Value_DLAA : NVSDK_NGX_PerfQuality_Value_MaxQuality;
          return false;
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
          PE::InputFloat("Radius", &m_tweak.hbaoRadius, 0.01f, 0, "%.6f");
          PE::Checkbox("Full resolution", &m_tweak.hbaoFullRes);
          PE::InputFloat("Blur sharpness", &m_frameConfig.hbaoSettings.blurSharpness, 1.0f);
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
                       "builds a cached blas depending on highest fully resident lod level.");
          PE::Checkbox("Blas Merging", &m_rendererConfig.useBlasMerging,
                       "builds a merged blas for closer instances. Guarantees only 2 dynamic blas per geometry.");
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
        PE::Checkbox("Separate Groups Kernel", &m_rendererConfig.useSeparateGroups,
                     "optimization that splits traversal into two separate kernels");
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
                                 && m_rendererConfig.useCulling && m_rendererConfig.useSeparateGroups));
          PE::Checkbox("Allow SW-Raster", (bool*)&m_rendererConfig.useComputeRaster,
                       "Allows use of compute-shader based rasterization (if visualize == visibility buffer / depth only)");
          PE::InputFloat("SW-Raster threshold", &m_frameConfig.swRasterThreshold, 1.0f, 1.0f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                         "cluster uses SW-Raster if its longest edge has less than the specified projected pixels");
          ImGui::EndDisabled();
        }
        else
        {
          ImGui::EndDisabled();
        }
        PE::treePop();
      }
      PE::end();

      ImGui::Separator();

      if(m_rendererConfig.useRenderStats && ImGui::BeginTable("##Render stats", 3, ImGuiTableFlags_RowBg))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 140.0f * ImGui::GetWindowDpiScale());
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Enqueued Tasks");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numTraversedTasks).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Enqueued Clusters");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numRenderedClusters).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Enqueued Triangles");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numRenderedTriangles).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Enqueued Tri/Cluster");
        ImGui::TableNextColumn();
        if(readback.numRenderedClusters > 0)
        {
          ImGui::Text("%.1f", float(readback.numRenderedTriangles) / float(readback.numRenderedClusters));
        }
        else
        {
          ImGui::Text("N/A");
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Rastered Triangles");
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numRasteredTriangles).c_str());

        if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD && m_rendererConfig.useComputeRaster)
        {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("Enqueued Clusters SW");
          ImGui::TableNextColumn();
          ImGui::Text("%s", formatMetric(readback.numRenderedClustersSW).c_str());
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("Enqueued Triangles SW");
          ImGui::TableNextColumn();
          ImGui::Text("%s", formatMetric(readback.numRenderedTrianglesSW).c_str());
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("Enqueued Tri/Cluster SW");
          ImGui::TableNextColumn();
          if(readback.numRenderedClustersSW > 0)
          {
            ImGui::Text("%.1f", float(readback.numRenderedTrianglesSW) / float(readback.numRenderedClustersSW));
          }
          else
          {
            ImGui::Text("N/A");
          }
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("Rastered Triangles SW");
          ImGui::TableNextColumn();
          ImGui::Text("%s", formatMetric(readback.numRasteredTrianglesSW).c_str());
        }
        ImGui::EndTable();
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
        PE::InputIntClamped("LoD group size", (int*)&m_sceneConfigEdit.clusterGroupSize, 8, 128, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                            "number of clusters that make a lod group. Their triangles are decimated together and they share a common error property");
        PE::InputIntClamped("Preferred node width", (int*)&m_sceneConfigEdit.preferredNodeWidth, 4, 32, 1, 1,
                            ImGuiInputTextFlags_EnterReturnsTrue,
                            "number of children a lod node should have (max is always 32). Currently _not_ implemented for nv_cluster_lod_builder.");
        PE::Checkbox("Prefer ray tracing (RT)", &m_sceneConfigEdit.meshoptPreferRayTracing,
                     "Configures meshoptimizer's lod cluster builder to prefer ray tracing over rasterization.");
        PE::InputFloat("RT fill weight", &m_sceneConfigEdit.meshoptFillWeight, 0, 0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "If ray tracing is preferred, influences weight between SAH optimized (towards zero), or filling clusters (higher value).");
        PE::InputFloat("RA split factor", &m_sceneConfigEdit.meshoptSplitFactor, 0, 0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "If raster is preferred, influences the maximum size of a cluster prior splitting it up.");

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
        PE::InputFloat("Normal weight", &m_sceneConfigEdit.simplifyNormalWeight, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "How much to weight this attribute for the error metric. 0 Disables");
        PE::InputFloat("TexCoord weight", &m_sceneConfigEdit.simplifyTexCoordWeight, 0, 0, "%.3f",
                       ImGuiInputTextFlags_EnterReturnsTrue, "How much to weight this attribute for the error metric. 0 Disables");
        PE::InputFloat("Tangent weight", &m_sceneConfigEdit.simplifyTangentWeight, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                       "How much to weight this attribute for the error metric. 0 Disables");
        PE::InputFloat("BiTangent Sign weight", &m_sceneConfigEdit.simplifyTangentSignWeight, 0, 0, "%.3f",
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
        PE::InputIntClamped("Max CLAS MiB", (int*)&m_streamingConfig.maxClasMegaBytes, 128, 1024 * 48, 16, 16,
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
                             getUsagePct(stats.usedClasBytes, stats.reservedClasBytes));
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
      m_streamGeometryHistogramMax = 0;
      m_streamHistogramOffset      = 0;
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


      uint32_t mbGeometry          = uint32_t((stats.usedDataBytes + divisor - 1) / divisor);
      uint32_t mbClas              = uint32_t((stats.usedClasBytes + divisor - 1) / divisor);
      m_streamGeometryHistogramMax = std::max(m_streamGeometryHistogramMax, mbGeometry);
      m_streamClasHistogramMax     = std::max(m_streamClasHistogramMax, mbClas);
      {
        m_streamHistogramOffset = (m_streamHistogramOffset + 1) % maxSlots;
        m_streamGeometryHistogram[(m_streamHistogramOffset + maxSlots - 1) % maxSlots] = mbGeometry;
        m_streamClasHistogram[(m_streamHistogramOffset + maxSlots - 1) % maxSlots]     = mbClas;
      }

      uiPlot(std::string("Streaming Geometry Memory (" MEMORY_MB ")"), std::string("past %d " MEMORY_MB " %d"),
             m_streamGeometryHistogram, m_streamGeometryHistogramMax, m_streamHistogramOffset);
      if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS_LOD)
      {
        uiPlot(std::string("Streaming CLAS Memory (" MEMORY_MB ")"), std::string("past % d " MEMORY_MB " % d"),
               m_streamClasHistogram, m_streamClasHistogramMax, m_streamHistogramOffset);
      }
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
      if(ImGui::BeginTable("Memory stats", 3, ImGuiTableFlags_RowBg))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Actual", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
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
        ImGui::Text("%s", formatMemorySize(resourceActual.getTotalSum()).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceReserved.getTotalSum()).c_str());
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

    if(ImGui::CollapsingHeader("Lighting", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      namespace PE = nvgui::PropertyEditor;
      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                      "Mix between flashlight and sun light");
      PE::end();
      ImGui::Text("Sun & Sky");
      nvgui::skySimpleParametersUI(m_frameConfig.frameConstants.skyParams, "misc", ImGuiTableFlags_Resizable);
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

    if(ImGui::CollapsingHeader("Advanced", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::InputIntClamped("Persistent traversal threads", (int*)&m_frameConfig.traversalPersistentThreads, 32,
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
    std::filesystem::path filePath = m_sceneFilePath;
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
