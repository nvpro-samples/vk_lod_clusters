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

#include "vk_nv_cluster_acc.h"
#include <nvvk/extensions_vk.hpp>

static PFN_vkGetClusterAccelerationStructureBuildSizesNV s_vkGetClusterAccelerationStructureBuildSizesNV = nullptr;
static PFN_vkCmdBuildClusterAccelerationStructureIndirectNV s_vkCmdBuildClusterAccelerationStructureIndirectNV = nullptr;

#ifndef NVVK_HAS_VK_NV_cluster_acceleration_structure
VKAPI_ATTR void VKAPI_CALL vkGetClusterAccelerationStructureBuildSizesNV(VkDevice device,
                                                                         const VkClusterAccelerationStructureInputInfoNV* input,
                                                                         VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo)
{
  s_vkGetClusterAccelerationStructureBuildSizesNV(device, input, pSizeInfo);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBuildClusterAccelerationStructureIndirectNV(VkCommandBuffer commandBuffer,
                                                                            const VkClusterAccelerationStructureCommandsInfoNV* cmdInfo)
{
  s_vkCmdBuildClusterAccelerationStructureIndirectNV(commandBuffer, cmdInfo);
}
#endif

VkBool32 load_VK_NV_cluster_accleration_structure(VkInstance instance, VkDevice device)
{
  s_vkGetClusterAccelerationStructureBuildSizesNV    = nullptr;
  s_vkCmdBuildClusterAccelerationStructureIndirectNV = nullptr;

  s_vkGetClusterAccelerationStructureBuildSizesNV =
      (PFN_vkGetClusterAccelerationStructureBuildSizesNV)vkGetDeviceProcAddr(device, "vkGetClusterAccelerationStructureBuildSizesNV");
  s_vkCmdBuildClusterAccelerationStructureIndirectNV =
      (PFN_vkCmdBuildClusterAccelerationStructureIndirectNV)vkGetDeviceProcAddr(device, "vkCmdBuildClusterAccelerationStructureIndirectNV");

  return s_vkGetClusterAccelerationStructureBuildSizesNV && s_vkCmdBuildClusterAccelerationStructureIndirectNV;
}
