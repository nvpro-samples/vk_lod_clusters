/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <span>

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
