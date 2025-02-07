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

#include <nvp/nvpsystem.hpp>
#include <imgui/backends/imgui_vk_extra.h>
#include <imgui/imgui_camera_widget.h>
#include <nvvk/samplers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvkhl/application.hpp>
#include <nvvkhl/element_benchmark_parameters.hpp>
#include <nvvkhl/element_profiler.hpp>
#include <nvvkhl/element_nvml.hpp>
#include <nvvkhl/element_logger.hpp>
#include <nvvkhl/element_camera.hpp>
#include <nvvkhl/element_gui.hpp>

#include "lodclusters.hpp"

namespace lodclusters {

nvvkhl::SampleAppLog                   g_logger;
std::shared_ptr<nvvkhl::ElementCamera> g_elementCamera;


//////////////////////////////////////////////////////////////////////////

class LodClustersElement : public nvvkhl::IAppElement
{
public:
  nvvkhl::Application* m_app;

  LodClusters::CallBacks   m_callbacks;
  LodClusters::EventStates m_eventStates;
  LodClusters              m_core;
  uint32_t                 m_width;
  uint32_t                 m_height;
  VkDescriptorSet          m_imguiImage   = nullptr;
  VkSampler                m_imguiSampler = nullptr;

  std::shared_ptr<nvvk::ProfilerVK> m_profilerVK;
  nvh::ParameterList*               m_parameterList = nullptr;

  bool m_useUI = true;

  LodClustersElement()           = default;
  ~LodClustersElement() override = default;

private:
  nvvk::Context* m_context{};

public:
  void setContext(nvvk::Context* context) { m_context = context; }

  void windowTitle()
  {
    // Window Title
    static float dirty_timer = 0.0F;
    dirty_timer += ImGui::GetIO().DeltaTime;
    if(dirty_timer > 1.0F)  // Refresh every seconds
    {
      const auto&           size = m_app->getViewportSize();
      std::array<char, 256> buf{};
      const int             ret = snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms", PROJECT_NAME,
                                           static_cast<int>(size.width), static_cast<int>(size.height),
                                           static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
      glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
      dirty_timer = 0;
    }
  }

  void cmdImageTransition(VkCommandBuffer    cmd,
                          VkImage            img,
                          VkImageAspectFlags aspects,
                          VkAccessFlags      src,
                          VkAccessFlags      dst,
                          VkImageLayout      oldLayout,
                          VkImageLayout      newLayout) const
  {
    VkPipelineStageFlags srcPipe = nvvk::makeAccessMaskPipelineStageFlags(src, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkPipelineStageFlags dstPipe = nvvk::makeAccessMaskPipelineStageFlags(dst, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    VkImageSubresourceRange range;
    memset(&range, 0, sizeof(range));
    range.aspectMask     = aspects;
    range.baseMipLevel   = 0;
    range.levelCount     = VK_REMAINING_MIP_LEVELS;
    range.baseArrayLayer = 0;
    range.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    memBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memBarrier.dstAccessMask        = dst;
    memBarrier.srcAccessMask        = src;
    memBarrier.oldLayout            = oldLayout;
    memBarrier.newLayout            = newLayout;
    memBarrier.image                = img;
    memBarrier.subresourceRange     = range;

    vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, nullptr, 0, nullptr, 1, &memBarrier);
  }

  void updateImage()
  {
    if(m_imguiImage)
    {
      ImGui_ImplVulkan_RemoveTexture(m_imguiImage);
      m_imguiImage = nullptr;
    }

    m_imguiImage = ImGui_ImplVulkan_AddTexture(m_imguiSampler, m_core.m_targetImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void setParameterList(nvh::ParameterList& paramaterList)
  {
    m_parameterList = &paramaterList;
    m_core.setupConfigParameters(paramaterList);
  }

  void onAttach(nvvkhl::Application* app) override
  {
    m_app = app;

    if(!m_core.initCore(*m_context, 128, 128, NVPSystem::exePath()))
    {
      exit(-1);
    }

    {
      const VkSamplerCreateInfo sampler_info = nvvk::makeSamplerCreateInfo();
      vkCreateSampler(m_context->m_device, &sampler_info, nullptr, &m_imguiSampler);
    }

    m_callbacks.openFile = [&](const char* msg, const char* exts) {
      return NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), msg, exts);
    };
    m_callbacks.parameterList = m_parameterList;

    updateImage();
  }

  void onDetach() override
  {
    m_core.deinit(*m_context);
    ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)m_imguiImage);
    vkDestroySampler(m_context->m_device, m_imguiSampler, nullptr);
  }

  // Return true if the current window is active
  bool isWindowHovered(ImGuiWindow* ref_window, ImGuiHoveredFlags flags)
  {
    ImGuiContext& g = *ImGui::GetCurrentContext();
    if(g.HoveredWindow != ref_window)
      return false;
    if(!ImGui::IsWindowContentHoverable(ref_window, ImGuiFocusedFlags_RootWindow))
      return false;
    if(g.ActiveId != 0 && !g.ActiveIdAllowOverlap && g.ActiveId != ref_window->MoveId)
      return false;

    // Cancel if over the title bar
    {
      if(g.IO.ConfigWindowsMoveFromTitleBarOnly)
        if(!(ref_window->Flags & ImGuiWindowFlags_NoTitleBar) || ref_window->DockIsActive)
          if(ref_window->TitleBarRect().Contains(g.IO.MousePos))
            return false;
    }

    return true;
  }

  void onUIRender() override
  {
    ImGuiWindow* viewportWindow = ImGui::FindWindowByName("Viewport");
    if(viewportWindow)
    {
      // If the mouse cursor is over the "Viewport", check for all inputs that can manipulate
      // the camera.
      if(isWindowHovered(viewportWindow, ImGuiFocusedFlags_RootWindow))
      {
        m_eventStates.winSize          = {m_width, m_height};
        m_eventStates.mouseButtonFlags = 0;
        m_eventStates.mouseButtonFlags |= ImGui::IsMouseDown(ImGuiMouseButton_Left) ? 1 : 0;
        m_eventStates.mouseButtonFlags |= ImGui::IsMouseDown(ImGuiMouseButton_Right) ? 2 : 0;
        m_eventStates.mouseButtonFlags |= ImGui::IsMouseDown(ImGuiMouseButton_Middle) ? 4 : 0;
        ImVec2 mousePos     = ImGui::GetMousePos();
        m_eventStates.mouse = {mousePos.x, mousePos.y};
        m_eventStates.mouseWheel += ImGui::GetIO().MouseWheel * 50.0f;
      }
      else
      {
        m_eventStates.mouseButtonFlags = 0;
      }
    }

    m_eventStates.alignView     = ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_A, false);
    m_eventStates.reloadShaders = ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_R, false);
    m_eventStates.saveView      = ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_C, false);

    m_core.processUI(glfwGetTime(), *m_profilerVK, m_callbacks);
    LodClusters::ChangeStates changes = m_core.handleChanges(m_width, m_height, m_eventStates);
    if(changes.targetImage)
    {
      updateImage();
    }
    if(changes.timerReset)
    {
      m_profilerVK->reset();
    }

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImVec2 corner = ImGui::GetCursorScreenPos();  // Corner of the viewport
    ImGui::Image(m_imguiImage, ImGui::GetContentRegionAvail());
    m_core.viewportUI(corner);
    ImGui::End();
  }


  void onRender(VkCommandBuffer cmd) override
  {
    m_core.renderFrame(cmd, m_width, m_height, glfwGetTime(), *m_profilerVK, m_app->getFrameCycleIndex(), m_app);
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    m_width  = width;
    m_height = height;
    m_core.initFramebuffers(m_width, m_height);
    updateImage();
  }


  // Called if showMenu is true
  void onUIMenu() override
  {
    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Open"))
      {
        std::string fileNameLo = m_callbacks.openFile("Pick model file",
                                                      "Supported (glTF 2.0)|*.gltf;*.glb;"
                                                      "|All|*.*");
        m_core.loadFile(fileNameLo);
      }
      if(ImGui::MenuItem("Save Cache"))
      {
        m_core.saveCacheFile();
      }
      ImGui::Separator();
      ImGui::EndMenu();
    }
    windowTitle();
  }
  void onFileDrop(const char* filename) override { m_core.loadFile(filename); }
};

}  // namespace lodclusters

using namespace lodclusters;


#include <thread>

int main(int argc, char** argv)
{
  try
  {
    NVPSystem sys(PROJECT_NAME);

    // This is not absolutely required, but having this early, loads the Vulkan DLL, which delays
    // the window to show up by ~1.5 seconds, but on the other hands, reduce the time the window
    // displays a white background.
    int glfw_valid = GLFW_TRUE;
    glfw_valid &= glfwInit();
    glfw_valid &= glfwVulkanSupported();
    if(!glfw_valid)
    {
      std::string err_message("Vulkan is not supported on this computer.");
#if _WIN32
      MessageBox(nullptr, err_message.c_str(), "Vulkan error", MB_OK);
#endif
      LOGE("%s", err_message.c_str());
      return EXIT_FAILURE;
    }

    nvvkhl::ApplicationCreateInfo appInfo;
    appInfo.name    = PROJECT_NAME;
    appInfo.useMenu = true;
    appInfo.vSync   = false;
    appInfo.imguiConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;  // keep single window

    // for now always set to false, given extension isn't supported
    bool     validationLayer       = false;
    bool     wantedVSync           = true;
    uint32_t compatibleDeviceIndex = 0;

    for(int a = 0; a < argc; a++)
    {
      if(strcmp(argv[a], "-device") == 0 && a + 1 < argc)
      {
        compatibleDeviceIndex = atoi(argv[a + 1]);
        a += 1;
      }
      else if(strcmp(argv[a], "-novalidation") == 0)
      {
        validationLayer = false;
      }
      else if(strcmp(argv[a], "-validation") == 0)
      {
        validationLayer = true;
      }
      else if(strcmp(argv[a], "-vsync") == 0 && a + 1 < argc)
      {
        wantedVSync = atoi(argv[a + 1]);
        a += 1;
      }
    }

    nvvk::Context           context{};
    nvvk::ContextCreateInfo contextCreateInfo(validationLayer);
    contextCreateInfo.setVersion(1, 3);
    contextCreateInfo.compatibleDeviceIndex = compatibleDeviceIndex;

    nvvkhl::addSurfaceExtensions(contextCreateInfo.instanceExtensions);
    contextCreateInfo.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    LodClusters::setupContextInfo(contextCreateInfo);

    if(!context.init(contextCreateInfo))
    {
      LOGE("Vulkan context init failed\n");
      return EXIT_FAILURE;
    }

    context.ignoreDebugMessage(0x715035dd);  // 16 bit storage for mesh shaders
    context.ignoreDebugMessage(0x6e224e9);   // 16 bit storage for mesh shaders

    // Setting up the layout of the application
    appInfo.dockSetup = [](ImGuiID viewportID) {
#ifdef _DEBUG
      // left side panel container
      ImGuiID debugID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.15F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Debug", debugID);
#endif
      // right side panel container
      ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Settings", settingID);
      ImGui::DockBuilderDockWindow("Misc Settings", settingID);

      // bottom panel container
      ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Log", loggerID);
      ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.75F, nullptr, &loggerID);
      ImGui::DockBuilderDockWindow("Profiler", profilerID);
      ImGuiID streamingID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Right, 0.66F, nullptr, &profilerID);
      ImGui::DockBuilderDockWindow("Streaming memory", streamingID);
      ImGuiID statisticsID = ImGui::DockBuilderSplitNode(streamingID, ImGuiDir_Right, 0.5F, nullptr, &streamingID);
      ImGui::DockBuilderDockWindow("Statistics", statisticsID);
    };

    appInfo.instance       = context.m_instance;
    appInfo.device         = context.m_device;
    appInfo.physicalDevice = context.m_physicalDevice;
    appInfo.queues.resize(1);
    appInfo.queues[0].queue       = context.m_queueGCT.queue;
    appInfo.queues[0].familyIndex = context.m_queueGCT.familyIndex;
    appInfo.queues[0].queueIndex  = context.m_queueGCT.queueIndex;

    {

      // Create the application
      auto app = std::make_unique<nvvkhl::Application>(appInfo);

      // create the different elements that constitutes the application
      auto elementAnimatedClusters = std::make_shared<LodClustersElement>();
      auto elementBenchmark        = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
      auto elementProfiler         = std::make_shared<nvvkhl::ElementProfiler>(true);
      g_elementCamera              = std::make_shared<nvvkhl::ElementCamera>();

      nvprintSetCallback([](int level, const char* fmt) { g_logger.addLog(level, "%s", fmt); });
      g_logger.setLogLevel(LOGBITS_INFO);
      bool hideLogOnStart = true;
#ifdef _DEBUG
      hideLogOnStart = false;
#endif
      auto elementLogger = std::make_shared<nvvkhl::ElementLogger>(&g_logger, !hideLogOnStart);
      elementBenchmark->setProfiler(elementProfiler);

      elementAnimatedClusters->m_profilerVK = elementProfiler;
      elementAnimatedClusters->setParameterList(elementBenchmark->parameterLists());
      elementAnimatedClusters->setContext(&context);

      app->addElement(elementBenchmark);
      app->addElement(elementAnimatedClusters);
      app->addElement(elementProfiler);
      app->addElement(elementLogger);
      app->addElement(g_elementCamera);
      // Set the actual vsync after initialization, works around an issue with window creation
      // FIXME see with MKL for a fix
      app->setVsync(wantedVSync);
      app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
      app->run();
    }

    context.deinit();

    return EXIT_SUCCESS;
  }
  catch(const std::exception& e)
  {
    LOGE("Uncaught exception: %s\n", e.what());
    assert(!"We should never reach here under normal operation, but this "
      "prints a nicer error message in the event we do.");
    return EXIT_FAILURE;
  }
}
