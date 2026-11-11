/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio_core.h"

#ifndef _SHADERIO_PTLAS_H_
#define _SHADERIO_PTLAS_H_

// Structures of `VK_NV_partitioned_acceleration_structure` (PTLAS).
// Layouts are verified against the Vulkan structs in `partitioned_acceleration_structures.hpp`.

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#endif

// VkPartitionedAccelerationStructureOpTypeNV
#define PTLAS_OP_WRITE_INSTANCE 0
#define PTLAS_OP_UPDATE_INSTANCE 1
#define PTLAS_OP_WRITE_PARTITION_TRANSLATION 2

// instances not tied to a spatial partition
#define PTLAS_PARTITION_GLOBAL (~0u)

// strides of the operation arguments below, the shaders have no sizeof
#define PTLAS_INSTANCE_WRITE_SIZE 104
#define PTLAS_INSTANCE_UPDATE_SIZE 16

// VkPartitionedAccelerationStructureInstanceFlagBitsNV
#define PtlasInstanceFlag_CULL_DISABLE_BIT (1 << 0)
#define PtlasInstanceFlag_FLIP_FACING_BIT (1 << 1)
#define PtlasInstanceFlag_FORCE_OPAQUE_BIT (1 << 2)
#define PtlasInstanceFlag_FORCE_NO_OPAQUE_BIT (1 << 3)
#define PtlasInstanceFlag_EXPLICIT_BBOX_BIT (1 << 4)

// Full instance description, argument of PTLAS_OP_WRITE_INSTANCE.
// The same instanceIndex must not appear twice within one operation.
struct PtlasInstanceWrite
{
  mat3x4 worldMatrix;
  // only used with PtlasInstanceFlag_EXPLICIT_BBOX_BIT
  vec3     bboxLo;
  vec3     bboxHi;
  uint32_t instanceCustomIndex;
  uint32_t instanceMask;
  uint32_t instanceSbtOffset;
  uint32_t instanceFlags;
  // slot within the PTLAS, and the partition that owns it
  uint32_t instanceIndex;
  uint32_t partitionIndex;
  uint64_t blasReference;
};

// Argument of PTLAS_OP_UPDATE_INSTANCE, changes the blas of an instance that
// was written before. Must also be used when the blas content changed.
struct PtlasInstanceUpdate
{
  uint32_t instanceIndex;
  uint32_t instanceSbtOffset;
  uint64_t blasReference;
};
BUFFER_REF_DECLARE_ARRAY(PtlasInstanceUpdates_inout, PtlasInstanceUpdate, , 8);

// One operation of a PTLAS build, `argCount` arguments of the type that
// `opType` implies, sourced from `argAddress`.
struct PtlasOperation
{
  uint32_t opType;
  uint32_t argCount;
  uint64_t argAddress;
  uint64_t argStride;
};
BUFFER_REF_DECLARE_ARRAY(PtlasOperations_inout, PtlasOperation, , 8);

#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_PTLAS_H_
