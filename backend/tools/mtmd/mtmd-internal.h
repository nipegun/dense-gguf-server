#pragma once

#include "mtmd.h"

#include <string>
#include <vector>

// !!! Internal header, to be used by mtmd and its unit tests only !!!

#define MTMD_INTERNAL_HEADER

// bitmap is null for text parts
struct mtmd_internal_part {
  std::string         vText;
  const mtmd_bitmap * bitmap;
  // only used for text parts
  bool                vParseSpecial = false;
};

// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always maps to one part
std::vector<std::vector<const mtmd_bitmap *>> fMtmdGroupMergeableBitmaps(std::vector<mtmd_internal_part> & pParts,
                                                                         int                               pNMerge);
