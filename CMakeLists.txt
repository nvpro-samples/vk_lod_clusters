cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES C CXX)

option(NVPRO2_ENABLE_nvgl OFF)
option(NVPRO2_ENABLE_nvgpu_monitor OFF)
option(NVPRO2_ENABLE_nvslang OFF)
option(NVPRO2_ENABLE_nvvkgltf OFF)


# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

#####################################################################################
# Download the default scene
download_files(FILENAMES bunny_v2.zip EXTRACT)

#####################################################################################
# extra libraries

if(NOT TARGET nv_cluster_lod_builder)
  add_subdirectory(external/nv_cluster_lod_builder)
endif()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.*)
file(GLOB SHADER_FILES shaders/*.glsl shaders/*.h)
file(GLOB VK_RADIX_SORT_FILES thirdparty/vulkan_radix_sort/src/vk_radix_sort.cc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vulkan_radix_sort/include)

#####################################################################################
# Executable
#

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${SHADER_FILES} ${VK_RADIX_SORT_FILES})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvvkglsl
  nv_cluster_lod_builder
  meshoptimizer
  cgltf
)

#####################################################################################
# source groups


source_group("Shader Files" FILES ${SHADER_FILES})
source_group("Source Files" FILES ${SOURCE_FILES})

#####################################################################################
# finalize project
#
add_project_definitions(${PROJECT_NAME})
copy_to_runtime_and_install( ${PROJECT_NAME}
    NVSHADERS_FILES "${NVSHADERS_DIR}/nvshaders/slang_types.h"
                    "${NVSHADERS_DIR}/nvshaders/sky_io.h.slang"
                    "${NVSHADERS_DIR}/nvshaders/sky_functions.h.slang"
                    "${NVSHADERS_DIR}/nvshaders/constants.h.slang"
    LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
    AUTO
)

