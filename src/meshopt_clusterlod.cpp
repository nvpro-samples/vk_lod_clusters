/**
* clusterlod - a small "library"/example built on top of meshoptimizer to generate cluster LOD hierarchies
* This is intended to either be used as is, or as a reference for implementing similar functionality in your engine.
*
* To use this code, you need to have one source file which includes meshoptimizer.h and defines CLUSTERLOD_IMPLEMENTATION
* before including this file. Other source files in your project can just include this file and use the provided functions.
*
* Copyright (C) 2025, by Arseny Kapoulkine (arseny.kapoulkine@gmail.com)
* This code is distributed under the MIT License. See notice at the end of this file.
*/

// Copyright (c) 2025, NVIDIA CORPORATION:
// This file originated from
// https://github.com/zeux/meshoptimizer/blob/8c6bb4f32ca8cfad31886745321a0ebabc3f0453/demo/clusterlod.h
// and was modified by NVIDIA

/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <meshoptimizer.h>

#define CLUSTERLOD_IMPLEMENTATION 1
#include "meshopt_clusterlod.h"
