/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fstream>

#include <json.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>

#include "scene.hpp"

namespace lodclusters {

// json key -> SceneConfig member. Keys match the equivalent command line options, so a
// setting is named the same everywhere. Only settings that affect simplification are listed,
// everything else about a geometry must stay global (cluster sizes and the like feed the
// storage layout and the renderer's worst-case buffers).
using SimplifyField = SimplifyOverrides::Field;

// append only, that keeps the `Entry::setMask` bits and therefore existing cache hashes stable
static const SimplifyOverrides::Field s_fields[] = {
    {"simplifynormalweight", SimplifyField::eFloat, offsetof(SceneConfig, simplifyNormalWeight)},
    {"simplifyuvweight", SimplifyField::eFloat, offsetof(SceneConfig, simplifyTexCoordWeight)},
    {"simplifytangentweight", SimplifyField::eFloat, offsetof(SceneConfig, simplifyTangentWeight)},
    {"simplifytangentsignweight", SimplifyField::eFloat, offsetof(SceneConfig, simplifyTangentSignWeight)},
    {"simplifymaterialweight", SimplifyField::eFloat, offsetof(SceneConfig, simplifyMaterialWeight)},
    {"loderrormergeprevious", SimplifyField::eFloat, offsetof(SceneConfig, lodErrorMergePrevious)},
    {"loderrormergeadditive", SimplifyField::eFloat, offsetof(SceneConfig, lodErrorMergeAdditive)},
    {"loderroredgelimit", SimplifyField::eFloat, offsetof(SceneConfig, lodErrorEdgeLimit)},
    {"simplifyerrorclamped", SimplifyField::eBool, offsetof(SceneConfig, simplifyErrorClamped)},
    {"simplifypreservefolds", SimplifyField::eBool, offsetof(SceneConfig, simplifyPreserveFolds)},
    {"simplifydilateall", SimplifyField::eBool, offsetof(SceneConfig, simplifyDilateBordersAll)},
    {"optimizeclusterslevel", SimplifyField::eUint, offsetof(SceneConfig, optimizeClustersLevel)},
    {"simplifydilatetwosided", SimplifyField::eBool, offsetof(SceneConfig, simplifyDilateBordersTwoSided)},
};

static size_t fieldSize(SimplifyField::Kind kind)
{
  switch(kind)
  {
    case SimplifyField::eBool:
      return sizeof(bool);
    case SimplifyField::eUint:
      return sizeof(uint32_t);
    default:
      return sizeof(float);
  }
}

static_assert(std::size(s_fields) <= 32, "SimplifyOverrides::Entry::setMask is 32 bits");

std::span<const SimplifyOverrides::Field> SimplifyOverrides::getFields()
{
  return std::span<const Field>(s_fields, std::size(s_fields));
}

static const uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
static const uint64_t FNV_PRIME        = 0x100000001b3ULL;

static uint64_t hashBytesFNV(uint64_t hash, const void* data, size_t size)
{
  for(size_t i = 0; i < size; i++)
  {
    hash = (hash ^ uint64_t(((const uint8_t*)data)[i])) * FNV_PRIME;
  }
  return hash;
}

bool SimplifyOverrides::load(const std::filesystem::path& filePath)
{
  entries.clear();

  std::string fileName = nvutils::utf8FromPath(filePath);

  std::ifstream stream(filePath);
  if(!stream.is_open())
  {
    LOGE("SimplifyOverrides: could not open \"%s\"\n", fileName.c_str());
    return false;
  }

  nlohmann::json root;
  try
  {
    stream >> root;
  }
  catch(const std::exception& e)
  {
    LOGE("SimplifyOverrides: \"%s\" is not valid json: %s\n", fileName.c_str(), e.what());
    return false;
  }

  if(!root.is_array())
  {
    LOGE("SimplifyOverrides: \"%s\" must contain an array of override entries\n", fileName.c_str());
    return false;
  }

  for(size_t i = 0; i < root.size(); i++)
  {
    const nlohmann::json& item = root[i];

    if(!item.is_object())
    {
      LOGE("SimplifyOverrides: \"%s\" entry %zu is not an object\n", fileName.c_str(), i);
      return false;
    }

    auto itMesh = item.find("mesh");
    if(itMesh == item.end() || !itMesh->is_string())
    {
      LOGE("SimplifyOverrides: \"%s\" entry %zu needs a \"mesh\" string with the name regular expression\n", fileName.c_str(), i);
      return false;
    }

    Entry entry;
    entry.pattern = itMesh->get<std::string>();

    try
    {
      entry.regex = std::regex(entry.pattern);
    }
    catch(const std::regex_error& e)
    {
      LOGE("SimplifyOverrides: \"%s\" entry %zu has an invalid \"mesh\" regular expression \"%s\": %s\n",
           fileName.c_str(), i, entry.pattern.c_str(), e.what());
      return false;
    }

    for(auto it = item.begin(); it != item.end(); ++it)
    {
      if(it.key() == "mesh")
        continue;

      uint32_t fieldIndex = ~0u;
      for(uint32_t f = 0; f < uint32_t(std::size(s_fields)); f++)
      {
        if(it.key() == s_fields[f].key)
        {
          fieldIndex = f;
          break;
        }
      }

      if(fieldIndex == ~0u)
      {
        LOGE("SimplifyOverrides: \"%s\" entry %zu has unknown setting \"%s\"\n", fileName.c_str(), i, it.key().c_str());
        return false;
      }

      const Field& field = s_fields[fieldIndex];

      uint8_t* dst = (uint8_t*)&entry.values + field.offset;

      if(field.kind == SimplifyField::eFloat)
      {
        if(!it->is_number())
        {
          LOGE("SimplifyOverrides: \"%s\" entry %zu setting \"%s\" must be a number\n", fileName.c_str(), i, field.key);
          return false;
        }
        *(float*)dst = it->get<float>();
      }
      else if(field.kind == SimplifyField::eUint)
      {
        if(!it->is_number_unsigned())
        {
          LOGE("SimplifyOverrides: \"%s\" entry %zu setting \"%s\" must be a non-negative integer\n", fileName.c_str(),
               i, field.key);
          return false;
        }
        *(uint32_t*)dst = it->get<uint32_t>();
      }
      else
      {
        if(!it->is_boolean())
        {
          LOGE("SimplifyOverrides: \"%s\" entry %zu setting \"%s\" must be true or false\n", fileName.c_str(), i, field.key);
          return false;
        }
        *(bool*)dst = it->get<bool>();
      }

      entry.setMask |= 1u << fieldIndex;
    }

    if(!entry.setMask)
    {
      LOGW("SimplifyOverrides: \"%s\" entry %zu (\"%s\") sets nothing\n", fileName.c_str(), i, entry.pattern.c_str());
    }

    entries.push_back(std::move(entry));
  }

  LOGI("SimplifyOverrides: %zu entries from\n  %s\n", entries.size(), fileName.c_str());

  return true;
}

uint64_t SimplifyOverrides::match(const std::string& meshName, SceneConfig* config) const
{
  uint64_t hash    = FNV_OFFSET_BASIS;
  bool     matched = false;

  for(const Entry& entry : entries)
  {
    if(!std::regex_match(meshName, entry.regex))
      continue;

    matched = true;

    // the hash must describe what was applied, not the resulting config, so that it is
    // independent of the global settings the overrides are layered on top of
    hash = hashBytesFNV(hash, entry.pattern.data(), entry.pattern.size());
    hash = hashBytesFNV(hash, &entry.setMask, sizeof(entry.setMask));

    for(uint32_t f = 0; f < uint32_t(std::size(s_fields)); f++)
    {
      if(!(entry.setMask & (1u << f)))
        continue;

      const Field& field = s_fields[f];
      const size_t size  = fieldSize(field.kind);
      const void*  src   = (const uint8_t*)&entry.values + field.offset;

      hash = hashBytesFNV(hash, src, size);

      if(config)
      {
        memcpy((uint8_t*)config + field.offset, src, size);
      }
    }
  }

  return matched ? hash : 0;
}

}  // namespace lodclusters
