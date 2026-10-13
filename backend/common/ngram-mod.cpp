#include "ngram-mod.h"

#include <algorithm>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), vUsed(0) {
  lEntries.resize(size);

  reset();
}

size_t common_ngram_mod::mIdx(const entry_t * pTokens) const {
  size_t vRes = 0;

  for (size_t vI = 0; vI < n; ++vI) {
    vRes = vRes * 6364136223846793005ULL + pTokens[vI];
  }

  vRes = vRes % lEntries.size();

  return vRes;
}

void common_ngram_mod::mAdd(const entry_t * pTokens) {
  const size_t cI = mIdx(pTokens);

  if (lEntries[cI] == cEmpty) {
    vUsed++;
  }

  lEntries[cI] = pTokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * pTokens) const {
  const size_t cI = mIdx(pTokens);

  return lEntries[cI];
}

void common_ngram_mod::reset() {
  std::fill(lEntries.begin(), lEntries.end(), cEmpty);
  vUsed = 0;
}

size_t common_ngram_mod::mGetN() const {
  return n;
}

size_t common_ngram_mod::mGetUsed() const {
  return vUsed;
}

size_t common_ngram_mod::size() const {
  return lEntries.size();
}

size_t common_ngram_mod::mSizeBytes() const {
  return lEntries.size() * sizeof(lEntries[0]);
}
