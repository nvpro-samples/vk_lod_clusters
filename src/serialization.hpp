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

#pragma once

#include <cstdint>

namespace serialization {

static constexpr uint64_t ALIGNMENT  = 16ULL;
static constexpr uint64_t ALIGN_MASK = ALIGNMENT - 1;
static_assert(ALIGNMENT >= sizeof(uint64_t));

template <typename T>
inline uint64_t getCachedSize(const std::span<T>& view)
{
  // use one extra ALIGNMENT to store count
  return ((view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK) + ALIGNMENT;
}

template <typename T>
inline void storeAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd, const std::span<const T>& view)
{
  assert(static_cast<uint64_t>(dataAddress) % ALIGNMENT == 0);

  if(isValid && dataAddress + getCachedSize(view) <= dataEnd)
  {
    union
    {
      uint64_t count;
      uint8_t  countData[ALIGNMENT];
    };
    memset(countData, 0, sizeof(countData));

    count = view.size();

    // store count first
    memcpy(reinterpret_cast<void*>(dataAddress), countData, ALIGNMENT);
    dataAddress += ALIGNMENT;

    if(view.size())
    {
      // then data
      memcpy(reinterpret_cast<void*>(dataAddress), view.data(), view.size_bytes());
      dataAddress += (view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK;
    }
  }
  else
  {
    isValid = false;
  }
}

template <typename T>
inline void loadAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd, std::span<const T>& view)
{
  union
  {
    const T* basePointer;
    uint64_t baseRaw;
  };
  baseRaw = dataAddress;

  assert(dataAddress % ALIGNMENT == 0);

  uint64_t count = *reinterpret_cast<const uint64_t*>(basePointer);
  baseRaw += ALIGNMENT;

  if(isValid && count && (baseRaw + (sizeof(T) * count) <= dataEnd))
  {
    // each array is 16 byte aligned
    view = std::span<const T>(basePointer, count);
  }
  else
  {
    view = {};
    // count of zero is valid, otherwise bail
    isValid = isValid && count == 0;
  }

  baseRaw += sizeof(T) * count;
  baseRaw = (baseRaw + ALIGN_MASK) & ~(ALIGN_MASK);

  dataAddress = baseRaw;
}

}  // namespace serialization
