#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct range_nfd {
  uint32_t vFirst;
  uint32_t vLast;
  uint32_t vNfd;
};

static const uint32_t cMaxCodepoints = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> cUnicodeRangesFlags;
extern const std::unordered_set<uint32_t>                         sUnicodeSetWhitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> cUnicodeMapLowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> cUnicodeMapUppercase;
extern const std::initializer_list<range_nfd>                     cUnicodeRangesNfd;
