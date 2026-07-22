/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <fmt/format.h>

#include "camera_path.hpp"

namespace lodclusters {

// Catmull-Rom spline through p1 and p2 using neighbours p0 and p3, t in [0,1].
static glm::dvec3 catmullRom(const glm::dvec3& p0, const glm::dvec3& p1, const glm::dvec3& p2, const glm::dvec3& p3, double t)
{
  double t2 = t * t;
  double t3 = t2 * t;
  return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

// Two keys are considered matching (a closed loop) when eye, center and up
// coincide within a small relative tolerance.
static bool keysMatch(const CameraPathKey& a, const CameraPathKey& b)
{
  auto approxEqual = [](const glm::dvec3& u, const glm::dvec3& v) {
    double scale = std::max({1.0, glm::length(u), glm::length(v)});
    return glm::distance(u, v) <= 1e-4 * scale;
  };
  return approxEqual(a.eye, b.eye) && approxEqual(a.ctr, b.ctr) && approxEqual(a.up, b.up);
}

bool CameraPath::sample(double u, CameraPathKey& out) const
{
  size_t n = keys.size();
  if(n == 0)
    return false;
  if(n == 1)
  {
    out = keys[0];
    return true;
  }

  u = std::clamp(u, 0.0, 1.0);

  // map to segment [i, i+1] with local parameter t
  double segF = u * double(n - 1);
  size_t i    = size_t(segF);
  if(i >= n - 1)
    i = n - 2;
  double t = segF - double(i);

  const CameraPathKey& k1 = keys[i];
  const CameraPathKey& k2 = keys[i + 1];

  if(smooth)
  {
    // For a closed loop (looping with a matching first/last key) wrap the boundary
    // neighbours across the join so the spline is smooth there; otherwise duplicate
    // the end points.
    bool wrap = loop && n >= 3 && keysMatch(keys.front(), keys.back());

    const CameraPathKey& k0 = (i > 0) ? keys[i - 1] : (wrap ? keys[n - 2] : keys[i]);
    const CameraPathKey& k3 = (i + 2 < n) ? keys[i + 2] : (wrap ? keys[1] : keys[n - 1]);
    out.eye                 = catmullRom(k0.eye, k1.eye, k2.eye, k3.eye, t);
    out.ctr                 = catmullRom(k0.ctr, k1.ctr, k2.ctr, k3.ctr, t);
  }
  else
  {
    out.eye = glm::mix(k1.eye, k2.eye, t);
    out.ctr = glm::mix(k1.ctr, k2.ctr, t);
  }

  glm::dvec3 up  = glm::mix(k1.up, k2.up, t);
  double     len = glm::length(up);
  out.up         = (len > 1e-9) ? (up / len) : k1.up;
  out.fov        = glm::mix(k1.fov, k2.fov, t);

  return true;
}

// Serialize the global settings as "smooth <0/1> loop <0/1> dur <sec>".
static std::string headerString(const CameraPath& path)
{
  return fmt::format("smooth {} loop {} dur {}", path.smooth ? 1 : 0, path.loop ? 1 : 0, path.duration);
}

// Serialize a keyframe as "{eye}, {center}, {up}, {fov}".
static std::string keyframeString(const CameraPathKey& k)
{
  return fmt::format("{{{}, {}, {}}}, {{{}, {}, {}}}, {{{}, {}, {}}}, {{{}}}",  //
                     k.eye.x, k.eye.y, k.eye.z,                                 //
                     k.ctr.x, k.ctr.y, k.ctr.z,                                 //
                     k.up.x, k.up.y, k.up.z,                                    //
                     k.fov);
}

std::string CameraPath::getString() const
{
  std::string text = headerString(*this);
  for(const CameraPathKey& k : keys)
    text += " ; " + keyframeString(k);
  return text;
}

// Extract all floating point numbers from a string, ignoring any other
// characters (braces, commas, whitespace, ...).
static std::vector<double> scanNumbers(const std::string& text)
{
  std::vector<double> numbers;
  const char*         p = text.c_str();
  while(*p)
  {
    char c = *p;
    if((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.')
    {
      char*  end = nullptr;
      double v   = std::strtod(p, &end);
      if(end != p)
      {
        numbers.push_back(v);
        p = end;
        continue;
      }
    }
    ++p;
  }
  return numbers;
}

bool CameraPath::setFromString(const std::string& text)
{
  if(text.empty())
    return false;

  // split into `;` separated chunks
  std::vector<std::string> chunks;
  {
    std::stringstream ss(text);
    std::string       chunk;
    while(std::getline(ss, chunk, ';'))
      chunks.push_back(chunk);
  }
  if(chunks.empty())
    return false;

  std::vector<CameraPathKey> newKeys;
  bool                       newSmooth   = true;
  bool                       newLoop     = false;
  double                     newDuration = 10.0;
  bool                       hasHeader   = false;

  size_t firstKeyChunk = 0;
  // when the first chunk has no `{` it is a header of `keyword value` pairs
  if(chunks[0].find('{') == std::string::npos)
  {
    std::istringstream header(chunks[0]);
    std::string        keyword;
    while(header >> keyword)
    {
      double value = 0;
      if(keyword == "smooth" && (header >> value))
      {
        newSmooth = value != 0.0;
        hasHeader = true;
      }
      else if(keyword == "loop" && (header >> value))
      {
        newLoop   = value != 0.0;
        hasHeader = true;
      }
      else if(keyword == "dur" && (header >> value))
      {
        newDuration = value;
        hasHeader   = true;
      }
    }
    firstKeyChunk = 1;
  }

  for(size_t c = firstKeyChunk; c < chunks.size(); c++)
  {
    std::vector<double> n = scanNumbers(chunks[c]);
    if(n.size() < 9)
      continue;  // skip incomplete/empty chunks (e.g. trailing separators)

    CameraPathKey key;
    key.eye = glm::dvec3(n[0], n[1], n[2]);
    key.ctr = glm::dvec3(n[3], n[4], n[5]);
    key.up  = glm::dvec3(n[6], n[7], n[8]);
    key.fov = (n.size() >= 10) ? n[9] : 60.0;
    newKeys.push_back(key);
  }

  // accept keyframe paths as well as header-only (empty) paths so that empty
  // paths created in the UI survive a save/reload round-trip; reject only when
  // nothing meaningful was parsed (e.g. a stray/garbage line)
  if(newKeys.empty() && !hasHeader)
    return false;

  keys     = std::move(newKeys);
  smooth   = newSmooth;
  loop     = newLoop;
  duration = newDuration;
  return true;
}

bool loadCameraPaths(const std::filesystem::path& filename, std::vector<CameraPath>& paths)
{
  std::ifstream in(filename);
  if(!in.is_open())
    return false;

  // A single path may span multiple lines. Each path begins with the `smooth`
  // header keyword (as written by getString()/saveCameraPaths()), so tokens are
  // accumulated until the next `smooth`. This also preserves header-only (empty)
  // paths. Anything from a `#` to the end of the line is a comment.
  std::vector<CameraPath> loaded;
  std::string             chunk;

  auto flush = [&]() {
    CameraPath path;
    if(!chunk.empty() && path.setFromString(chunk))
      loaded.push_back(std::move(path));
    chunk.clear();
  };

  std::string line;
  while(std::getline(in, line))
  {
    size_t hash = line.find('#');
    if(hash != std::string::npos)
      line.resize(hash);

    std::istringstream iss(line);
    std::string        token;
    while(iss >> token)
    {
      if(token == "smooth" && !chunk.empty())
        flush();  // this token starts the next path
      chunk += token;
      chunk += ' ';
    }
  }
  flush();

  paths = std::move(loaded);
  return true;
}

bool saveCameraPaths(const std::filesystem::path& filename, const std::vector<CameraPath>& paths)
{
  std::ofstream out(filename, std::ios::trunc);
  if(!out.is_open())
    return false;

  out << "# vk_lod_clusters camera paths\n";
  out << "# format: smooth <0/1> loop <0/1> dur <sec> ; {eye}, {center}, {up}, {fov} ; ...\n";
  out << "# a path may span multiple lines; it ends at the next header keyword\n";
  for(const CameraPath& path : paths)
  {
    // header on its own line, then one keyframe per line
    out << "\n" << headerString(path);
    for(const CameraPathKey& k : path.keys)
      out << " ;\n  " << keyframeString(k);
    out << "\n";
  }

  return true;
}

}  // namespace lodclusters
