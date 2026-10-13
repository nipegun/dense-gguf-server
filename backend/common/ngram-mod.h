#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

//
// common_ngram_mod
//

// basic n-gram hasher
struct common_ngram_mod {
  using entry_t = int32_t;

  static constexpr entry_t cEmpty = -1;

  common_ngram_mod(uint16_t n, size_t size);

  size_t  mIdx(const entry_t * pTokens) const;
  void    mAdd(const entry_t * pTokens);
  entry_t get(const entry_t * pTokens) const;  // return -1 if not found

  void reset();

  size_t mGetN() const;
  size_t mGetUsed() const;

  size_t size() const;
  size_t mSizeBytes() const;

private:
  size_t n;  // ngram size to hash

  size_t vUsed;

  std::vector<entry_t> lEntries;
};
