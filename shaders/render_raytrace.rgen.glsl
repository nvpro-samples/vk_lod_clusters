/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"
#if USE_DLSS
#include "dlss_util.h"
#endif

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_RENDERMATERIALS_SSBO, set = 0) buffer renderMaterialsBuffer
{
  RenderMaterial materials[];
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};

layout(set = 0, binding = BINDINGS_RAYTRACING_DEPTH, r32f) uniform image2D imgRaytracingDepth;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba16f) uniform image2D imgColor;
#if USE_DLSS
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssAlbedo, rgba8)             uniform image2D imgDlssAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecAlbedo, rgba16f)       uniform image2D imgDlssSpecAlbedo;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssNormalRoughness, rgba16f)  uniform image2D imgDlssNormalRoughness;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssMotion, rg16f)             uniform image2D imgDlssMotion;
layout(set = 0, binding = BINDINGS_RENDER_TARGET + SHADERIO_eDlssSpecHitDist, r16f)         uniform image2D imgDlssSpecHitDist;
#endif

//////////////////////////////////////////////////////////////

#if DEBUG_VISUALIZATION && ALLOW_SHADING
// This ray-gen declares no bindless texture array; it only needs texturing.glsl's computeViewRayDirection
// for the wireframe-overlay differentials. Opt out of sampleBindless (which references that array).
#define TEXTURING_SKIP_BINDLESS 1
#include "texturing.glsl"
#endif

//////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT RayPayload rayHit;

#if ALLOW_SHADING
#include "sky_fog.glsl"
#endif

//////////////////////////////////////////////////////////////

#include "render_raytrace_raygen.glsl"
