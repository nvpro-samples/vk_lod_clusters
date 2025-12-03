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

#include <float.h>
#include <unordered_map>
#include <string>

#include <fmt/format.h>
#include <glm/gtc/type_ptr.hpp>
#include <cgltf.h>
#include <meshoptimizer.h>
#include <nvutils/logger.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>

#include "scene.hpp"

namespace {
class SpinLock
{
public:
  SpinLock(std::atomic_uint32_t& reference)
      : m_reference(reference)
  {
    while(m_reference.exchange(1, std::memory_order_acquire) == 1)
    {
      // Spin-wait with pause instruction
      while(m_reference.load(std::memory_order_relaxed) == 1)
      {
        std::this_thread::yield();
      }
    }
  }

  ~SpinLock() { m_reference.store(0, std::memory_order_release); }

private:
  std::atomic_uint32_t& m_reference;
};

struct FileMappingList
{
  struct Entry
  {
    nvutils::FileReadMapping mapping;
    int64_t                  refCount = 1;
  };
  std::unordered_map<std::string, Entry>       m_nameToMapping;
  std::unordered_map<const void*, std::string> m_dataToName;
#ifndef NDEBUG
  int64_t m_openBias = 0;
#endif

  bool open(const char* path, size_t* size, void** data)
  {
#ifndef NDEBUG
    m_openBias++;
#endif

    std::string pathStr(path);

    auto it = m_nameToMapping.find(pathStr);
    if(it != m_nameToMapping.end())
    {
      *data = const_cast<void*>(it->second.mapping.data());
      *size = it->second.mapping.size();
      it->second.refCount++;
      return true;
    }

    Entry entry;
    if(entry.mapping.open(path))
    {
      const void* mappingData = entry.mapping.data();
      *data                   = const_cast<void*>(mappingData);
      *size                   = entry.mapping.size();
      m_dataToName.insert({mappingData, pathStr});
      m_nameToMapping.insert({pathStr, std::move(entry)});
      return true;
    }

    return false;
  }

  void close(void* data)
  {
#ifndef NDEBUG
    m_openBias--;
#endif
    auto itName = m_dataToName.find(data);
    if(itName != m_dataToName.end())
    {
      auto itMapping = m_nameToMapping.find(itName->second);
      if(itMapping != m_nameToMapping.end())
      {
        itMapping->second.refCount--;

        if(!itMapping->second.refCount)
        {
          m_nameToMapping.erase(itMapping);
          m_dataToName.erase(itName);
        }
      }
    }
  }

  ~FileMappingList()
  {
#ifndef NDEBUG
    assert(m_openBias == 0 && "open/close bias wrong");
#endif
    assert(m_nameToMapping.empty() && m_dataToName.empty() && "not all opened files were closed");
  }
};

const uint8_t* cgltf_buffer_view_data(const cgltf_buffer_view* view)
{
  if(view->data)
    return (const uint8_t*)view->data;

  if(!view->buffer->data)
    return NULL;

  const uint8_t* result = (const uint8_t*)view->buffer->data;
  result += view->offset;
  return result;
}

cgltf_result cgltf_read(const struct cgltf_memory_options* memory_options,
                        const struct cgltf_file_options*   file_options,
                        const char*                        path,
                        cgltf_size*                        size,
                        void**                             data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  if(mappings->open(path, size, data))
  {
    return cgltf_result_success;
  }

  return cgltf_result_io_error;
}

void cgltf_release(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  mappings->close(data);
}

// Defines a unique_ptr that can be used for cgltf_data objects.
// Freeing a unique_cgltf_ptr calls cgltf_free, instead of delete.
// This can be constructed using unique_cgltf_ptr foo(..., &cgltf_free).
using unique_cgltf_ptr = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;


// based on meshopt_quantizeFloat
// https://github.com/zeux/meshoptimizer/blob/master/src/quantization.cpp
inline float quantizeFloat(float value, uint32_t dropBits)
{
  union
  {
    uint32_t u32;
    float    f32;
  } un;

  un.f32      = value;
  uint32_t ui = un.u32;

  const int32_t mask  = (1 << (dropBits)) - 1;
  const int32_t round = (1 << (dropBits)) >> 1;

  int32_t  e   = ui & 0x7f800000;
  uint32_t rui = (ui + round) & ~mask;

  // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
  ui = e == 0x7f800000 ? ui : rui;

  // flush denormals to zero
  ui = e == 0 ? 0 : ui;

  un.u32 = ui;
  return un.f32;
}

inline glm::vec2 quantizeFloat(const glm::vec2& vec, uint32_t dropBits)
{
  glm::vec2 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  return res;
}

inline glm::vec3 quantizeFloat(const glm::vec3& vec, uint32_t dropBits)
{
  glm::vec3 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  res.z = quantizeFloat(vec.z, dropBits);
  return res;
}

inline glm::vec4 quantizeFloat(const glm::vec4& vec, uint32_t dropBits)
{
  glm::vec4 res;
  res.x = quantizeFloat(vec.x, dropBits);
  res.y = quantizeFloat(vec.y, dropBits);
  res.z = quantizeFloat(vec.z, dropBits);
  res.w = quantizeFloat(vec.w, dropBits);
  return res;
}

// Traverses the glTF node and any of its children, adding a MeshInstance to
// the meshSet for each referenced glTF primitive.
void addInstancesFromNode(std::vector<lodclusters::Scene::Instance>&     instances,
                          std::vector<lodclusters::Scene::GeometryView>& geometryViews,
                          const std::vector<size_t>&                     meshToGeometry,
                          const cgltf_data*                              data,
                          const cgltf_node*                              node,
                          const glm::mat4                                parentObjToWorldTransform = glm::mat4(1))
{
  if(node == nullptr)
    return;

  // Compute this node's object-to-world transform.
  // See https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_004_ScenesNodes.md .
  // Note that this depends on glm::mat4 being column-major.
  // The documentation above also means that vectors are multiplied on the right.
  glm::mat4 localNodeTransform(1);
  cgltf_node_transform_local(node, glm::value_ptr(localNodeTransform));
  const glm::mat4 nodeObjToWorldTransform = parentObjToWorldTransform * localNodeTransform;

  // If this node has a mesh, add instances for its primitives.
  if(node->mesh != nullptr)
  {
    lodclusters::Scene::Instance instance{};
    const ptrdiff_t              meshIndex   = (node->mesh) - data->meshes;
    const cgltf_material*        material    = node->mesh->primitives[0].material;
    bool                         addInstance = true;

    if(material)
    {
      instance.materialID = uint32_t(material - data->materials);
      if(material->unlit || material->has_pbr_metallic_roughness)
      {
        instance.color.x = material->pbr_metallic_roughness.base_color_factor[0];
        instance.color.y = material->pbr_metallic_roughness.base_color_factor[1];
        instance.color.z = material->pbr_metallic_roughness.base_color_factor[2];
        instance.color.w = material->pbr_metallic_roughness.base_color_factor[3];
      }
      else if(material->has_pbr_specular_glossiness)
      {
        instance.color.x = material->pbr_specular_glossiness.diffuse_factor[0];
        instance.color.y = material->pbr_specular_glossiness.diffuse_factor[1];
        instance.color.z = material->pbr_specular_glossiness.diffuse_factor[2];
        instance.color.w = material->pbr_specular_glossiness.diffuse_factor[3];
      }

      if(material->alpha_mode == cgltf_alpha_mode_blend)
      {
        addInstance = false;
      }
    }

    if(addInstance)
    {
      instance.geometryID = uint32_t(meshToGeometry[meshIndex]);
      instance.matrix     = nodeObjToWorldTransform;

      geometryViews[instance.geometryID].instanceReferenceCount++;


      instances.push_back(instance);
    }
  }

  // Recurse over any children of this node.
  const size_t numChildren = node->children_count;
  for(size_t childIdx = 0; childIdx < numChildren; childIdx++)
  {
    addInstancesFromNode(instances, geometryViews, meshToGeometry, data, node->children[childIdx], nodeObjToWorldTransform);
  }
}

}  // namespace


namespace lodclusters {
Scene::Result Scene::loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath)
{
  std::string fileName = nvutils::utf8FromPath(filePath);

  // Parse the glTF file using cgltf
  cgltf_options options = {};

  FileMappingList mappings;
  options.file.read      = cgltf_read;
  options.file.release   = cgltf_release;
  options.file.user_data = &mappings;

  cgltf_result     cgltfResult;
  unique_cgltf_ptr gltf = unique_cgltf_ptr(nullptr, &cgltf_free);
  {
    // We have this local pointer followed by an ownership transfer here
    // because cgltf_parse_file takes a pointer to a pointer to cgltf_data.
    cgltf_data* rawData = nullptr;
    cgltfResult         = cgltf_parse_file(&options, fileName.c_str(), &rawData);
    gltf                = unique_cgltf_ptr(rawData, &cgltf_free);
  }
  // Check for errors; special message for legacy files
  if(cgltfResult == cgltf_result_legacy_gltf)
  {
    LOGE(
        "loadGLTF: This glTF file is an unsupported legacy file - probably glTF 1.0, while cgltf only supports glTF "
        "2.0 files. Please load a glTF 2.0 file instead.\n");
    return SCENE_RESULT_ERROR;
  }
  else if((cgltfResult != cgltf_result_success) || (gltf == nullptr))
  {
    LOGE("loadGLTF: cgltf_parse_file failed. Is this a valid glTF file? (cgltf result: %d)\n", cgltfResult);
    return SCENE_RESULT_ERROR;
  }

  // Perform additional validation.
  cgltfResult = cgltf_validate(gltf.get());
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file could be parsed, but cgltf_validate failed. Consider using the glTF Validator at "
        "https://github.khronos.org/glTF-Validator/ to see if the non-displacement parts of the glTF file are correct. "
        "(cgltf result: %d)\n",
        cgltfResult);
    return SCENE_RESULT_ERROR;
  }

  // if we are loading from a cache file, we don't need any of the raw buffers
  if(!m_cacheFileView.isValid())
  {
    // For now, also tell cgltf to go ahead and load all buffers.
    cgltfResult = cgltf_load_buffers(&options, gltf.get(), fileName.c_str());
    if(cgltfResult != cgltf_result_success)
    {
      LOGE(
          "loadGLTF: The glTF file was valid, but cgltf_load_buffers failed. Are the glTF file's referenced file paths "
          "valid? (cgltf result: %d)\n",
          cgltfResult);
      return SCENE_RESULT_ERROR;
    }
  }

  // glTF doesn't have trivial instancing of meshes with different materials.
  // We need to detect meshes with identical primitive/accessor setups first,
  // these become our unique geometries that we can then instance under different
  // materials as well.

  std::vector<size_t> geometryToMesh;
  std::vector<size_t> geometryTriangleCount;
  std::vector<size_t> taskToGeometry;
  std::vector<size_t> meshToGeometry(gltf->meshes_count, -1);

  uint64_t totalTriangleCount = 0;

  {
    size_t geometryMemoryEstimate = 0;

    std::unordered_map<std::string, size_t> mapMeshToGeometry;

    for(size_t meshIndex = 0; meshIndex < gltf->meshes_count; meshIndex++)
    {
      const cgltf_mesh gltfMesh = gltf->meshes[meshIndex];

      size_t      meshMemoryEstimate = 0;
      size_t      meshTriangleCount  = 0;
      std::string meshIdentifier;

      for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
      {
        cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

        if(gltfPrim->type != cgltf_primitive_type_triangles)
        {
          continue;
        }

        if(gltfPrim->attributes_count == 0)
        {
          continue;
        }

        struct MeshAccessors
        {
          const void* pos    = nullptr;
          const void* index  = nullptr;
          const void* tex    = nullptr;
          const void* normal = nullptr;
        } meshAccessors;

        for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
        {
          const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
          const cgltf_accessor*  accessor   = gltfAttrib.data;

          if(strcmp(gltfAttrib.name, "POSITION") == 0)
          {
            meshAccessors.pos = accessor;
            meshMemoryEstimate += sizeof(glm::vec4) * accessor->count;
          }
          else if(strcmp(gltfAttrib.name, "TEXCOORD_0") == 0)
          {
            meshAccessors.tex = accessor;
          }
          else if(strcmp(gltfAttrib.name, "NORMAL") == 0)
          {
            meshAccessors.normal = accessor;
          }
        }

        meshAccessors.index = gltfPrim->indices;
        meshMemoryEstimate += sizeof(uint32_t) * gltfPrim->indices->count;

        meshTriangleCount += gltfPrim->indices->count / 3;
        totalTriangleCount += gltfPrim->indices->count / 3;

        // just serialize the pointer values as identifier for the mesh
        meshIdentifier +=
            fmt::format("{},{},{},{},", meshAccessors.pos, meshAccessors.normal, meshAccessors.index, meshAccessors.tex);
      }

      // find canonical string in map
      auto pair = mapMeshToGeometry.try_emplace(meshIdentifier, geometryToMesh.size());
      if(pair.second)
      {
        size_t geometryIndex      = geometryToMesh.size();
        meshToGeometry[meshIndex] = geometryIndex;
        geometryToMesh.push_back(meshIndex);
        taskToGeometry.push_back(geometryIndex);
        geometryTriangleCount.push_back(meshTriangleCount);
        geometryMemoryEstimate += meshMemoryEstimate;
      }
      else
      {
        meshToGeometry[meshIndex] = pair.first->second;
      }
    }

    // if there is too much geometry memory in the scene and we are not in processing only mode, early out
    if(!m_cacheFileView.isValid() && !m_loaderConfig.processingOnly
       && (geometryMemoryEstimate > size_t(m_loaderConfig.forcePreprocessMiB) * 1024 * 1024))
    {
      return SCENE_RESULT_NEEDS_PREPROCESS;
    }
  }

  m_geometryStorages.resize(geometryToMesh.size());
  m_geometryViews.resize(geometryToMesh.size());

  beginProcessingOnly(geometryToMesh.size());

  if(!m_cacheFileView.isValid())
  {
    processingInfo.setupCompressedGltf(gltf->buffer_views_count);
  }

  // when we are resuming in processingOnly mode, we might have completed several geometries already,
  // which is passed to influence the decision about the parallelism mode.
  processingInfo.setupParallelism(geometryToMesh.size(), m_processingOnlyPartialCompleted, m_loaderConfig.processingMode);

  if(processingInfo.numOuterThreads > processingInfo.numInnerThreads)
  {
    // let's do the actual processing in a slightly different order (large meshes first).
    // This gives better work distribution across threads, avoids few long running threads
    // at the end. Thanks Arseny Kapoulkine for this suggestion.
    std::sort(taskToGeometry.begin(), taskToGeometry.end(),
              [&](size_t l, size_t r) { return geometryTriangleCount[l] > geometryTriangleCount[r]; });
  }

  auto fnLoadAndProcessGeometry = [&](uint64_t taskIndex, uint32_t threadOuterIdx) {
    uint64_t geometryIndex = taskToGeometry[taskIndex];
    // map back from unique geometry to gltf mesh
    size_t meshIndex = geometryToMesh[geometryIndex];

    loadGeometryGLTF(processingInfo, geometryIndex, meshIndex, gltf.get());
  };

  // for partial files we don't have the completed triangle information
  processingInfo.logBegin(m_processingOnlyPartialFile ? 0 : totalTriangleCount);
  if(m_loaderConfig.progressPct)
  {
    m_loaderConfig.progressPct->store(0);
  }

  nvutils::parallel_batches_pooled<1>(geometryToMesh.size(), fnLoadAndProcessGeometry, processingInfo.numOuterThreads);

  processingInfo.logEnd();
  if(m_loaderConfig.progressPct)
  {
    m_loaderConfig.progressPct->store(100);
  }

  bool notCompleted = processingInfo.progressGeometriesCompleted != geometryToMesh.size();
  if(notCompleted)
  {
    LOGW("Error in processing geometries, completed / required mismatch\nTry using `--processingonly 1`\n");
  }
  else
  {
    computeHistogramMaxs();
  }

  if(endProcessingOnly(notCompleted))
  {
    return notCompleted ? SCENE_RESULT_ERROR : SCENE_RESULT_PREPROCESS_COMPLETED;
  }

  if(notCompleted)
  {
    return m_cacheFileView.isValid() ? SCENE_RESULT_CACHE_INVALID : SCENE_RESULT_ERROR;
  }

  if(gltf->scenes_count > 0)
  {
    const cgltf_scene scene = (gltf->scene != nullptr) ? (*(gltf->scene)) : (gltf->scenes[0]);
    for(size_t nodeIdx = 0; nodeIdx < scene.nodes_count; nodeIdx++)
    {
      addInstancesFromNode(m_instances, m_geometryViews, meshToGeometry, gltf.get(), scene.nodes[nodeIdx]);
    }
  }
  else
  {
    for(size_t nodeIdx = 0; nodeIdx < gltf->nodes_count; nodeIdx++)
    {
      if(gltf->nodes[nodeIdx].parent == nullptr)
      {
        addInstancesFromNode(m_instances, m_geometryViews, meshToGeometry, gltf.get(), &(gltf->nodes[nodeIdx]));
      }
    }
  }


  if(gltf->cameras_count > 0)
  {
    for(size_t nodeIdx = 0; nodeIdx < gltf->nodes_count; nodeIdx++)
    {
      if(gltf->nodes[nodeIdx].camera != nullptr && gltf->nodes[nodeIdx].camera->type == cgltf_camera_type_perspective)
      {
        Camera cam{};
        cam.fovy = gltf->nodes[nodeIdx].camera->data.perspective.yfov;
        glm::mat4 worldNodeTransform(1);
        cgltf_node_transform_world(&gltf->nodes[nodeIdx], glm::value_ptr(cam.worldMatrix));
        cam.eye    = glm::vec3(cam.worldMatrix[3]);
        cam.center = (m_bbox.hi + m_bbox.lo) * 0.5f;
        cam.up     = {0, 1, 0};
        m_cameras.push_back(cam);
      }
    }
  }

  return SCENE_RESULT_SUCCESS;
}


bool Scene::loadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                    std::unordered_set<struct cgltf_buffer_view*>& compressedViews,
                                    const struct cgltf_data*                       gltf)
{
  static bool warned   = false;
  bool        hadError = false;

  for(cgltf_buffer_view* bufferView : compressedViews)
  {
    size_t bufferViewIndex = bufferView - gltf->buffer_views;

    SpinLock lock((std::atomic_uint32_t&)processingInfo.bufferViewLocks[bufferViewIndex]);

    uint32_t users = processingInfo.bufferViewUsers[bufferViewIndex];
    if(users == 0)
    {
      // this decoding logic was derived from `decompressMeshopt`
      // in https://github.com/zeux/meshoptimizer/blob/master/gltf/parsegltf.cpp

      cgltf_meshopt_compression* mc = &bufferView->meshopt_compression;

      const unsigned char* source = (const unsigned char*)mc->buffer->data;
      if(!source)
        return false;
      source += mc->offset;

      void* result = malloc(mc->count * mc->stride);
      if(!result)
        return false;

      int  rc   = -1;
      bool warn = false;

      switch(mc->mode)
      {
        case cgltf_meshopt_compression_mode_attributes:
          warn = meshopt_decodeVertexVersion(source, mc->size) != 0;
          rc   = meshopt_decodeVertexBuffer(result, mc->count, mc->stride, source, mc->size);
          break;

        case cgltf_meshopt_compression_mode_triangles:
          warn = meshopt_decodeIndexVersion(source, mc->size) != 1;
          rc   = meshopt_decodeIndexBuffer(result, mc->count, mc->stride, source, mc->size);
          break;

        case cgltf_meshopt_compression_mode_indices:
          warn = meshopt_decodeIndexVersion(source, mc->size) != 1;
          rc   = meshopt_decodeIndexSequence(result, mc->count, mc->stride, source, mc->size);
          break;
      }

      if(rc != 0)
      {
        free(result);
        return false;
      }

      bufferView->data = result;

      if(warn && !warned)
      {
        LOGW("Warning: EXT_meshopt_compression data uses versions outside of the glTF specification (vertex 0 / index 1 expected)\n");
        warned = true;
      }

      switch(mc->filter)
      {
        case cgltf_meshopt_compression_filter_octahedral:
          meshopt_decodeFilterOct(result, mc->count, mc->stride);
          break;

        case cgltf_meshopt_compression_filter_quaternion:
          meshopt_decodeFilterQuat(result, mc->count, mc->stride);
          break;

        case cgltf_meshopt_compression_filter_exponential:
          meshopt_decodeFilterExp(result, mc->count, mc->stride);
          break;

        default:
          break;
      }
    }
    processingInfo.bufferViewUsers[bufferViewIndex] = users + 1;
  }

  return true;
}

void Scene::unloadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                      std::unordered_set<struct cgltf_buffer_view*>& compressedViews,
                                      const struct cgltf_data*                       gltf)
{
  for(cgltf_buffer_view* bufferView : compressedViews)
  {
    size_t bufferViewIndex = bufferView - gltf->buffer_views;

    SpinLock lock((std::atomic_uint32_t&)processingInfo.bufferViewLocks[bufferViewIndex]);

    uint32_t users = processingInfo.bufferViewUsers[bufferViewIndex]--;

    if(users == 0)
    {
      free(bufferView->data);
    }
  }
}

template <class T, bool doQuantize, bool doBBox>
inline void readAttributesGLTF(const cgltf_accessor* accessor,
                               float*                writeAttributes,
                               size_t                attributeStride,
                               cgltf_type            expectedType,
                               uint32_t              dropBits = 0,
                               T*                    bboxMin  = nullptr,
                               T*                    bboxMax  = nullptr)
{
  if(accessor->component_type == cgltf_component_type_r_32f && accessor->type == expectedType && accessor->stride == sizeof(T))
  {
    const T* readAttributes = (const T*)(cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset);
    for(size_t i = 0; i < accessor->count; i++)
    {
      T tmp = readAttributes[i];

      if(doQuantize && dropBits)
      {
        tmp = quantizeFloat(tmp, dropBits);
      }

      if(doBBox)
      {
        *bboxMin = glm::min(*bboxMin, tmp);
        *bboxMax = glm::max(*bboxMax, tmp);
      }

      *(T*)&writeAttributes[i * attributeStride] = tmp;
    }
  }
  else
  {
    for(size_t i = 0; i < accessor->count; i++)
    {
      T tmp;
      cgltf_accessor_read_float(accessor, i, &tmp.x, sizeof(T) / sizeof(float));

      if(doQuantize && dropBits)
      {
        tmp = quantizeFloat(tmp, dropBits);
      }

      if(doBBox)
      {
        *bboxMin = glm::min(*bboxMin, tmp);
        *bboxMax = glm::max(*bboxMax, tmp);
      }

      *(T*)&writeAttributes[i * attributeStride] = tmp;
    }
  }
}

void Scene::loadGeometryGLTF(ProcessingInfo& processingInfo, uint64_t geometryIndex, size_t meshIndex, const struct cgltf_data* gltf)
{
  // when resuming a partial processing, early out if it was already processed
  // second entry is dataSize
  if(m_processingOnlyPartialFile && m_processingOnlyGeometryOffsets[geometryIndex * 2 + 1])
  {
    uint32_t percentage = processingInfo.logCompletedGeometry();
    if(m_loaderConfig.progressPct)
    {
      m_loaderConfig.progressPct->store(percentage);
    }

    return;
  }

  std::unordered_set<cgltf_buffer_view*> compressedViews;

  const cgltf_mesh& gltfMesh = gltf->meshes[meshIndex];
  GeometryStorage&  geometry = m_geometryStorages[geometryIndex];
  geometry.bbox              = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  // count triangle and vertices pass
  uint32_t triangleCount = 0;
  uint32_t verticesCount = 0;
  for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
  {
    cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

    if(gltfPrim->type != cgltf_primitive_type_triangles)
    {
      continue;
    }

    // If the mesh has no attributes, there's nothing we can do
    if(gltfPrim->attributes_count == 0)
    {
      continue;
    }

    for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
    {
      const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
      const cgltf_accessor*  accessor   = gltfAttrib.data;

      if(accessor->buffer_view->has_meshopt_compression)
        compressedViews.insert(accessor->buffer_view);

      if(strcmp(gltfAttrib.name, "POSITION") == 0)
      {
        verticesCount += (uint32_t)gltfAttrib.data->count;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL) && strcmp(gltfAttrib.name, "NORMAL") == 0)
      {
        m_hasVertexNormals = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT) && strcmp(gltfAttrib.name, "TANGENT") == 0)
      {
        m_hasVertexTangents = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0) && strcmp(gltfAttrib.name, "TEXCOORD_0") == 0)
      {
        m_hasVertexTexCoord0 = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1) && strcmp(gltfAttrib.name, "TEXCOORD_1") == 0)
      {
        m_hasVertexTexCoord1 = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      }
    }

    if(gltfPrim->indices->buffer_view->has_meshopt_compression)
      compressedViews.insert(gltfPrim->indices->buffer_view);

    triangleCount += (uint32_t)gltfPrim->indices->count / 3;
  }


  // use memset 0 to avoid issues with padding within struct
  memset(&geometry.lodInfo, 0, sizeof(geometry.lodInfo));
  geometry.lodInfo.inputTriangleCount = triangleCount;
  geometry.lodInfo.inputVertexCount   = verticesCount;

  // test if this mesh exists in the cache
  bool isCached = checkCache(geometry.lodInfo, geometryIndex);

  // invalid cache
  if(m_cacheFileView.isValid() && !isCached)
  {
    LOGW("geometry mismatches scene cache file\n");
    return;
  }

  // load vertices & index data
  if(!isCached)
  {
    // disable tangents if no TEXCOORDS or NORMALS are provided
    // might as well use automatic tangent space then
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    }
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    }

    // disable TEX_1 if no TEX_0
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
    }

    size_t attributeStride = (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL ? 3 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT ? 4 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 ? 2 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1 ? 2 : 0);
    uint32_t attributeStart = 0;
    uint32_t attributeEnd   = uint32_t(attributeStride);

    // all attributes with simplification weights must come first due to how
    // meshoptimizer works
    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      if(m_config.simplifyNormalWeight > 0)
      {
        geometry.attributeNormalOffset = attributeStart;
        attributeStart += 3;
      }
      else
      {
        geometry.attributeNormalOffset = attributeEnd - 3;
        attributeEnd -= 3;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      if(m_config.simplifyTexCoordWeight > 0)
      {
        geometry.attributeTex0offset = attributeStart;
        attributeStart += 2;
      }
      else
      {
        geometry.attributeTex0offset = attributeEnd - 2;
        attributeEnd -= 2;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
    {
      if(m_config.simplifyTexCoordWeight > 0)
      {
        geometry.attributeTex1offset = attributeStart;
        attributeStart += 2;
      }
      else
      {
        geometry.attributeTex1offset = attributeEnd - 2;
        attributeEnd -= 2;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
    {
      if(m_config.simplifyTangentSignWeight > 0 && m_config.simplifyTangentSignWeight > 0)
      {
        geometry.attributeTangentOffset = attributeStart;
        attributeStart += 4;
      }
      else
      {
        geometry.attributeTangentOffset = attributeEnd - 4;
        attributeEnd -= 4;
      }
    }

    assert(attributeStart == attributeEnd);

    geometry.attributesWithWeights = attributeStart;
    geometry.vertexPositions.resize(verticesCount);
    geometry.vertexAttributes.resize(verticesCount * attributeStride, 0);
    geometry.triangles.resize(triangleCount);

    // decompress views
    if(!compressedViews.empty())
    {
      if(!loadCompressedViewsGLTF(processingInfo, compressedViews, gltf))
      {
        LOGW("Error decompressing GLTF\n");
        return;
      }
    }

    // fill pass
    uint32_t offsetVertices  = 0;
    uint32_t offsetTriangles = 0;

    for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
    {
      cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

      if(gltfPrim->type != cgltf_primitive_type_triangles)
      {
        continue;
      }

      // If the mesh has no attributes, there's nothing we can do
      if(gltfPrim->attributes_count == 0)
      {
        continue;
      }

      uint32_t numVertices = 0;

      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
        const cgltf_accessor*  accessor   = gltfAttrib.data;

        if(strcmp(gltfAttrib.name, "POSITION") == 0)
        {
          glm::vec3* writeVertices = geometry.vertexPositions.data() + offsetVertices;

          readAttributesGLTF<glm::vec3, true, true>(accessor, (float*)writeVertices, 3, cgltf_type_vec3,
                                                    m_config.useCompressedData ? m_config.compressionPosDropBits : 0,
                                                    &geometry.bbox.lo, &geometry.bbox.hi);

          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "NORMAL") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeNormalOffset;

          readAttributesGLTF<glm::vec3, false, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec3);

          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "TANGENT") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeTangentOffset;

          readAttributesGLTF<glm::vec4, false, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec4);
          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "TEXCOORD_0") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeTex0offset;

          readAttributesGLTF<glm::vec2, true, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec2,
                                                     m_config.useCompressedData ? m_config.compressionTexDropBits : 0);
          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "TEXCOORD_1") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeTex1offset;

          readAttributesGLTF<glm::vec2, true, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec2,
                                                     m_config.useCompressedData ? m_config.compressionTexDropBits : 0);
          numVertices = (uint32_t)accessor->count;
        }
      }

      // indices
      {
        const cgltf_accessor* accessor = gltfPrim->indices;

        uint32_t* writeIndices = (uint32_t*)(geometry.triangles.data() + offsetTriangles);

        if(offsetVertices == 0 && accessor->component_type == cgltf_component_type_r_32u
           && accessor->type == cgltf_type_scalar && accessor->stride == sizeof(uint32_t))
        {
          memcpy(writeIndices, cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset,
                 sizeof(uint32_t) * accessor->count);
        }
        else
        {
          for(size_t i = 0; i < accessor->count; i++)
          {
            writeIndices[i] = (uint32_t)cgltf_accessor_read_index(gltfPrim->indices, i) + offsetVertices;
          }
        }

        offsetTriangles += (uint32_t)accessor->count / 3;
      }

      offsetVertices += numVertices;
    }
  }

  processGeometry(processingInfo, geometryIndex, isCached);

  if(!compressedViews.empty())
  {
    unloadCompressedViewsGLTF(processingInfo, compressedViews, gltf);
  }

  uint32_t percentage = processingInfo.logCompletedGeometry(triangleCount);
  if(m_loaderConfig.progressPct)
  {
    m_loaderConfig.progressPct->store(percentage);
  }
}
}  // namespace lodclusters
