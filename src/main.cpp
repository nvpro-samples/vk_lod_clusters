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

#ifdef _DEBUG
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, (format), __VA_ARGS__);                                                                            \
    fprintf(stderr, "\n");                                                                                             \
  } while(false)
#endif

#define VMA_IMPLEMENTATION

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif

#include <imgui/imgui.h>

#include <nvvk/validation_settings.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/parameter_parser.hpp>

#include "lodclusters.hpp"

using namespace lodclusters;

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name    = TARGET_NAME;
  appInfo.useMenu = true;

  VkPhysicalDeviceMeshShaderFeaturesNV  meshNV  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
  VkPhysicalDeviceMeshShaderFeaturesEXT meshEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rayPosKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clustersNV = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};
  VkPhysicalDeviceShaderClockFeaturesKHR clockKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
      .queues             = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
  };
  vkSetup.deviceExtensions.push_back({VK_NV_MESH_SHADER_EXTENSION_NAME, &meshNV, false});
  vkSetup.deviceExtensions.push_back({VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshEXT});
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accKHR});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rayKHR});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &rayPosKHR});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayQueryKHR});
  // set to version 2 compatibility instead of VK_NV_CLUSTER_ACCELERATION_STRUCTURE_SPEC_VERSION to cover more drivers
  vkSetup.deviceExtensions.push_back({VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, &clustersNV, false, 2});
  vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockKHR});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures});
  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &shadingRateFeatures});
  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &barycentricFeatures});

  nvutils::ProfilerManager                    profilerManager;
  std::shared_ptr<nvutils::CameraManipulator> cameraManipulator = std::make_shared<nvutils::CameraManipulator>();

  nvutils::ParameterRegistry parameterRegistry;
  nvutils::ParameterParser   parameterParser;

  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);

  LodClusters::Info sampleInfo;
  sampleInfo.cameraManipulator               = cameraManipulator;
  sampleInfo.profilerManager                 = &profilerManager;
  sampleInfo.parameterRegistry               = &parameterRegistry;
  sampleInfo.parameterParser                 = &parameterParser;
  std::shared_ptr<LodClusters> sampleElement = std::make_shared<LodClusters>(sampleInfo);

  parameterParser.add(parameterRegistry);
  parameterParser.parse(argc, argv);

  // can skip vulkan
  if(sampleElement->isProcessingOnly())
  {
    sampleElement->doProcessingOnly();
    return 0;
  }

  nvvk::ValidationSettings validationSettings;
  if(vkSetup.enableValidationLayers)
  {
    validationSettings.message_id_filter = {"VUID-RuntimeSpirv-storageInputOutput16-06334", "VUID-VkShaderModuleCreateInfo-pCode-08740"};

    vkSetup.instanceCreateInfoExt = validationSettings.buildPNextChain();
  }

  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  nvvk::Context vkContext;

  // Initialize the Vulkan loader
  NVVK_CHECK(volkInitialize());

  {
    nvutils::ScopedTimer st("Creating Vulkan Context");


#if USE_DLSS
    // Adding the DLSS extensions to the instance
    static std::vector<VkExtensionProperties> extraInstanceExtensions;
    DlssRayReconstruction::getRequiredInstanceExtensions({}, extraInstanceExtensions);
    for(auto& ext : extraInstanceExtensions)
    {
      vkSetup.instanceExtensions.emplace_back(ext.extensionName);
    }
#endif
    VkResult result{};

    vkContext.contextInfo = vkSetup;

    result = vkContext.createInstance();
    result = vkContext.selectPhysicalDevice();

#if USE_DLSS
    // Adding the extra device extensions required by DLSS
    static std::vector<VkExtensionProperties> extraDeviceExtensions;
    DlssRayReconstruction::getRequiredDeviceExtensions({}, vkContext.getInstance(), vkContext.getPhysicalDevice(), extraDeviceExtensions);
    for(auto& ext : extraDeviceExtensions)
    {
      vkContext.contextInfo.deviceExtensions.push_back({.extensionName = ext.extensionName, .specVersion = ext.specVersion});
    }
#endif

    result = vkContext.createDevice();
    NVVK_CHECK(result);

    nvvk::DebugUtil::getInstance().init(vkContext.getDevice());


    if(vkContext.contextInfo.verbose)
    {
      NVVK_CHECK(nvvk::Context::printVulkanVersion());
      NVVK_CHECK(nvvk::Context::printInstanceLayers());
      NVVK_CHECK(nvvk::Context::printInstanceExtensions(vkContext.contextInfo.instanceExtensions));
      NVVK_CHECK(nvvk::Context::printDeviceExtensions(vkContext.getPhysicalDevice(), vkContext.contextInfo.deviceExtensions));
      NVVK_CHECK(nvvk::Context::printGpus(vkContext.getInstance(), vkContext.getPhysicalDevice()));
      LOGI("_________________________________________________\n");
    }
  }

  sampleElement->setSupportsClusters(vkContext.hasExtensionEnabled(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME));
  sampleElement->setSupportsMeshShaderEXT(vkContext.hasExtensionEnabled(VK_EXT_MESH_SHADER_EXTENSION_NAME));
  sampleElement->setSupportsMeshShaderNV(vkContext.hasExtensionEnabled(VK_NV_MESH_SHADER_EXTENSION_NAME));

  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  bool hasDebugUI = sampleElement->getShowDebugUI();

  // Setting up the layout of the application
  appInfo.dockSetup = [&hasDebugUI](ImGuiID viewportID) {
    if(hasDebugUI)
    {
      // left side panel container
      ImGuiID debugID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.15F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Debug", debugID);
    }

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

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  auto                  logger      = std::make_shared<nvapp::ElementLogger>();
  nvapp::ElementLogger* loggerDeref = logger.get();
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    loggerDeref->addLog(logLevel, "%s", text.c_str());
  });

  auto profilerUiSettings          = std::make_shared<nvapp::ElementProfiler::ViewSettings>();
  profilerUiSettings->table.levels = 1u;

  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());
  app.addElement(sampleElement);
  app.addElement(logger);
  app.addElement(std::make_shared<nvapp::ElementCamera>(cameraManipulator));
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager, profilerUiSettings));
  app.run();

  nvutils::Logger::getInstance().setLogCallback(nullptr);

  // Cleanup in reverse order
  app.deinit();
  vkContext.deinit();

  return 0;
}
