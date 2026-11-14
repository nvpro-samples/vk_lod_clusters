/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio_core.h"

#ifndef _SHADERIO_FRAME_CAPTURE_H_
#define _SHADERIO_FRAME_CAPTURE_H_

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#endif

// a thread converts 4x2 pixels, so its luma rows and its chroma row are whole words
#define FRAME_CAPTURE_NV12_WORKGROUP_X 16
#define FRAME_CAPTURE_NV12_WORKGROUP_Y 8

struct FrameCaptureNv12Constants
{
  // tightly packed rgba8 pixels of `srcWidth * srcHeight`
  uint64_t srcAddress;
  // nv12: `width * height` luma bytes, then `height / 2` rows of interleaved cb cr
  uint64_t dstAddress;
  // a multiple of 4, height a multiple of 2
  uint width;
  uint height;
  // the source is fit into the frame, pixel x of the frame reads (x + 0.5 - offset) * invScale
  uint  srcWidth;
  uint  srcHeight;
  vec2  offset;
  float invScale;
  // the source is bgra8, as a swapchain image typically is
  uint swapRedBlue;
  // FRAME_CAPTURE_READ_IMAGE: the source is the image itself, not srcAddress
  uint readImage;
};

#ifdef __cplusplus
}
#endif

#endif
