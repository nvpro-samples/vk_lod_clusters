/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace lodclusters {

// A single camera keyframe. Uses the same eye/center/up/fov model as
// nvutils::CameraManipulator::Camera so keyframes can be captured from, and
// applied back to, the manipulator without conversion.
struct CameraPathKey
{
  glm::dvec3 eye = glm::dvec3(0, 0, 0);
  glm::dvec3 ctr = glm::dvec3(0, 0, 0);
  glm::dvec3 up  = glm::dvec3(0, 1, 0);
  double     fov = 60.0;
};

// CameraPath is a self-contained keyframe based fly-through that is fully
// defined within this sample (it does not rely on nvpro_core camera plugins).
//
// It can be serialized to / parsed from a compact, copy/paste friendly string
// so a path can be provided on the command line just like the camera settings.
//
// String format (single line, whitespace tolerant, `#` comments allowed inside
// config files). A header with global settings, followed by `;` separated
// keyframes, each in the same braced style as the camera string:
//
//   smooth 1 loop 0 dur 10 ; {ex, ey, ez}, {cx, cy, cz}, {ux, uy, uz}, {fov} ; ...
//
// The header is optional; when the first `;` separated chunk already contains a
// `{` it is treated as a keyframe and global defaults are used. Per keyframe the
// fov is optional (defaults to 60).
class CameraPath
{
public:
  std::vector<CameraPathKey> keys;

  bool   smooth   = true;   // Catmull-Rom interpolation (vs. piecewise linear)
  bool   loop     = false;  // whether real-time playback wraps around
  double duration = 10.0;   // seconds for a full real-time playback (ignored by fixed-step playback)

  bool   empty() const { return keys.empty(); }
  size_t size() const { return keys.size(); }

  // Evaluate the path at the normalized position `u` in [0,1] (0 == first key,
  // 1 == last key). Returns false when there are no keys.
  bool sample(double u, CameraPathKey& out) const;

  // Serialization for copy/paste and the `--addcamerapath` command line option.
  std::string getString() const;
  bool        setFromString(const std::string& text);
};

// Load/save a list of camera paths as a simple text file, one path per line
// (in the getString() format). Blank lines and `#` comments are ignored.
// loadCameraPaths overwrites `paths` and returns false only if the file cannot
// be opened (a readable but empty file results in an empty list).
bool loadCameraPaths(const std::filesystem::path& filename, std::vector<CameraPath>& paths);
bool saveCameraPaths(const std::filesystem::path& filename, const std::vector<CameraPath>& paths);

}  // namespace lodclusters
