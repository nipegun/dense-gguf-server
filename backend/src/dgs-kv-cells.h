#pragma once

#include "dgs-cparams.h"
#include "dgs.h"

#include <bitset>
#include <cassert>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

struct dgs_kv_cell_ext {
  // 2D spatial positions, typically used for M-RoPE
  dgs_pos vX = 0;
  dgs_pos vY = 0;

  // when tok = DGS_TOKEN_NULL when the cell is produced by embedding input (i.e. multimodal)
  // use case: n-gram embeddings hash
  dgs_token vTok = DGS_TOKEN_NULL;

  // return true if the current 2D spatial position is greater than other
  bool mIs2dGt(dgs_pos pOx, dgs_pos pOy) const { return (vY > pOy) || (vY == pOy && vX > pOx); }

  void reset() {
    static_assert(std::is_trivially_copyable_v<dgs_kv_cell_ext>);

    *this = dgs_kv_cell_ext{};
  }
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class dgs_kv_cells {
public:
  using seq_set_t = std::bitset<DGS_MAX_SEQ>;

  void reset() {
    for (uint32_t vI = 0; vI < lPos.size(); ++vI) {
      lPos[vI] = -1;
      lExt[vI].reset();
      lShift[vI] = 0;
      lSeq[vI].reset();
    }

    vHasShift = false;

    sUsed.clear();

    for (uint32_t vS = 0; vS < DGS_MAX_SEQ; ++vS) {
      sSeqPos[vS].clear();
    }
  }

  void mResetShift() {
    vHasShift = false;

    for (uint32_t vI = 0; vI < lShift.size(); ++vI) {
      lShift[vI] = 0;
    }
  }

  uint32_t size() const { return lPos.size(); }

  void resize(uint32_t n) {
    lPos.resize(n);
    lExt.resize(n);
    lShift.resize(n);
    lSeq.resize(n);

    reset();
  }

  bool mIsEmpty(uint32_t pI) const {
    assert(i < pos.size());
    assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

    return lPos[pI] == -1;
  }

  uint32_t mGetUsed() const { return sUsed.size(); }

  // the index of the first cell that is used
  // return 0 if no cells are used
  uint32_t mUsedMin() const { return sUsed.empty() ? 0 : *sUsed.begin(); }

  // the index of the last cell that is used + 1
  // return 0 if no cells are used
  uint32_t mUsedMaxP1() const { return sUsed.empty() ? 0 : *sUsed.rbegin() + 1; }

  bool mGetHasShift() const { return vHasShift; }

  // move cell isrc to idst (used during defrag)
  //void mv(uint32_t isrc, uint32_t idst) {
  //    assert(isrc < pos.size());
  //    assert(idst < pos.size());

  //    assert(pos[idst] == -1);
  //    assert(pos[isrc] != -1);

  //    pos  [idst] = pos  [isrc];
  //    shift[idst] = shift[isrc];
  //    seq  [idst] = seq  [isrc];

  //    pos  [isrc] = -1;
  //    shift[isrc] =  0;
  //    seq  [isrc].reset();

  //    used.erase (isrc);
  //    used.insert(idst);
  //}

  // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
  dgs_kv_cells mCp(uint32_t pI, uint32_t n) const {
    assert(i + n <= pos.size());

    dgs_kv_cells vRes;

    vRes.resize(n);

    for (uint32_t vJ = 0; vJ < n; ++vJ) {
      const auto cIdx = pI + vJ;

      vRes.lPos[vJ] = lPos[cIdx];
      vRes.lExt[vJ] = lExt[cIdx];
      vRes.lSeq[vJ] = lSeq[cIdx];

      assert(shift[idx] == 0);
    }

    return vRes;
  }

  // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
  dgs_kv_cells mCp(const std::vector<uint32_t> & pIdxs) const {
    dgs_kv_cells vRes;

    vRes.resize(pIdxs.size());

    for (uint32_t vJ = 0; vJ < pIdxs.size(); ++vJ) {
      const auto cIdx = pIdxs[vJ];

      vRes.lPos[vJ] = lPos[cIdx];
      vRes.lExt[vJ] = lExt[cIdx];
      vRes.lSeq[vJ] = lSeq[cIdx];

      assert(shift[idx] == 0);
    }

    return vRes;
  }

  // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
  void set(uint32_t pI, const dgs_kv_cells & pOther) {
    assert(i + other.pos.size() <= pos.size());

    for (uint32_t vJ = 0; vJ < pOther.lPos.size(); ++vJ) {
      const auto cIdx = pI + vJ;

      if (lPos[cIdx] == -1 && pOther.lPos[vJ] != -1) {
        sUsed.insert(pI + vJ);
      }

      if (lPos[cIdx] != -1 && pOther.lPos[vJ] == -1) {
        sUsed.erase(pI + vJ);
      }

      if (lPos[cIdx] != -1) {
        mSeqPosRm(pI + vJ);
      }

      lPos[cIdx] = pOther.lPos[vJ];
      lExt[cIdx] = pOther.lExt[vJ];
      lSeq[cIdx] = pOther.lSeq[vJ];

      if (lPos[cIdx] != -1) {
        mSeqPosAdd(pI + vJ);
      }

      assert(shift[idx] == 0);
    }
  }

  // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
  void set(const std::vector<uint32_t> & pIdxs, const dgs_kv_cells & pOther) {
    assert(idxs.size() == other.pos.size());

    for (uint32_t vJ = 0; vJ < pOther.lPos.size(); ++vJ) {
      const auto cIdx = pIdxs[vJ];

      if (lPos[cIdx] == -1 && pOther.lPos[vJ] != -1) {
        sUsed.insert(cIdx);
      }

      if (lPos[cIdx] != -1 && pOther.lPos[vJ] == -1) {
        sUsed.erase(cIdx);
      }

      if (lPos[cIdx] != -1) {
        mSeqPosRm(cIdx);
      }

      lPos[cIdx] = pOther.lPos[vJ];
      lExt[cIdx] = pOther.lExt[vJ];
      lSeq[cIdx] = pOther.lSeq[vJ];

      if (lPos[cIdx] != -1) {
        mSeqPosAdd(cIdx);
      }

      assert(shift[idx] == 0);
    }
  }

  // clear a non-empty cell
  void mRm(uint32_t pI) {
    assert(i < pos.size());
    assert(pos[i] != -1);

    mSeqPosRm(pI);
    lSeq[pI].reset();

    lPos[pI] = -1;
    lExt[pI].reset();
    lShift[pI] = 0;

    sUsed.erase(pI);
  }

  // note: call only if the cell has seq_id
  // return true if the cell becomes empty
  bool mSeqRm(uint32_t pI, dgs_seq_id pSeqId) {
    assert(i < pos.size());
    assert(seq[i].test(seq_id));
    assert(pos[i] != -1);
    assert(seq_id >= 0);

    lSeq[pI].reset(pSeqId);
    mSeqPosDec(pSeqId, pI);

    if (lSeq[pI].none()) {
      lPos[pI] = -1;
      lExt[pI].reset();
      lShift[pI] = 0;

      sUsed.erase(pI);

      return true;
    }

    return false;
  }

  // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
  bool mSeqKeep(uint32_t pI, dgs_seq_id pSeqId) {
    assert(i < pos.size());

    if (lSeq[pI].test(pSeqId)) {
      mSeqPosRm(pI);
      lSeq[pI].reset();

      lSeq[pI].set(pSeqId);
      mSeqPosInc(pSeqId, pI);

      return false;
    }

    if (lSeq[pI].any()) {
      mSeqPosRm(pI);
      lSeq[pI].reset();

      lPos[pI] = -1;
      lExt[pI].reset();
      lShift[pI] = 0;

      sUsed.erase(pI);

      return true;
    }

    assert(pos[i] == -1);

    return false;
  }

  // number of different sequences in the cell
  int mSeqCount(uint32_t pI) const {
    assert(i < pos.size());
    assert(pos[i] != -1);

    return lSeq[pI].count();
  }

  // the full set of sequences this cell is visible to
  const seq_set_t & mSeqGetAll(uint32_t pI) const {
    assert(i < pos.size());

    return lSeq[pI];
  }

  // check if the cell contains seq_id
  bool mSeqHas(uint32_t pI, dgs_seq_id pSeqId) const {
    assert(i < pos.size());
    assert(seq_id >= 0);

    return lSeq[pI].test(pSeqId);
  }

  // the (pos, cell) pairs of sequence seq_id, ordered by position
  const std::set<std::pair<dgs_pos, uint32_t>> & mSeqPosGet(dgs_seq_id pSeqId) const {
    assert(seq_id >= 0);
    assert(seq_id < DGS_MAX_SEQ);

    return sSeqPos[pSeqId];
  }

  // the token of the cell of sequence seq_id at the largest position <= p
  // when several cells share that position, the one with the highest index wins
  // return DGS_TOKEN_NULL if the sequence has no cell at or before p
  // note: used by n-gram input embeddings to recover the tokens preceding a ubatch
  dgs_token mSeqPosTokLe(dgs_seq_id pSeqId, dgs_pos pP) const {
    assert(seq_id >= 0);
    assert(seq_id < DGS_MAX_SEQ);

    const auto & sSp = sSeqPos[pSeqId];

    auto vIt = sSp.upper_bound({ pP, std::numeric_limits<uint32_t>::max() });
    if (vIt == sSp.begin()) {
      return DGS_TOKEN_NULL;
    }

    return lExt[(--vIt)->second].vTok;
  }

  // note: call only if the cell is not empty and the seq_id is not in the cell
  void mSeqAdd(uint32_t pI, dgs_seq_id pSeqId) {
    assert(i < pos.size());
    assert(pos[i] != -1);
    assert(!seq[i].test(seq_id));

    lSeq[pI].set(pSeqId);
    mSeqPosInc(pSeqId, pI);
  }

  // return the sequence id of this cell
  // note: call only for cells with exactly one sequence
  dgs_seq_id mSeqGet(uint32_t pI) const {
    assert(seq[i].count() == 1);

    for (int vS = 0; vS < DGS_MAX_SEQ; ++vS) {
      if (lSeq[pI].test(vS)) {
        return vS;
      }
    }

    return -1;
  }

  // the minimum position of sequence seq_id currently present in any of the cells
  // return -1 if the sequence is not present
  dgs_pos mSeqPosMin(dgs_seq_id pSeqId) const {
    assert(seq_id >= 0);
    assert(seq_id < DGS_MAX_SEQ);

    if (sSeqPos[pSeqId].empty()) {
      return -1;
    }

    return sSeqPos[pSeqId].begin()->first;
  }

  // the maximum position of sequence seq_id currently present in any of the cells
  // return -1 if the sequence is not present
  dgs_pos mSeqPosMax(dgs_seq_id pSeqId) const {
    assert(seq_id >= 0);
    assert(seq_id < DGS_MAX_SEQ);

    if (sSeqPos[pSeqId].empty()) {
      return -1;
    }

    return sSeqPos[pSeqId].rbegin()->first;
  }

  // note: call only if the cell is not empty
  dgs_pos mPosGet(uint32_t pI) const {
    assert(i < pos.size());
    assert(pos[i] != -1);

    return lPos[pI];
  }

  const dgs_kv_cell_ext & mExtGet(uint32_t pI) const {
    assert(i < pos.size());
    assert(pos[i] != -1);

    return lExt[pI];
  }

  // note: call only if the cell is not empty
  dgs_pos mGetShift(uint32_t pI) const {
    assert(i < pos.size());
    assert(pos[i] != -1);

    return lShift[pI];
  }

  // check if a cell is not empty and its position is within [p0, p1)
  bool mPosIn(uint32_t pI, dgs_pos pP0, dgs_pos pP1) const {
    assert(i < pos.size());

    return lPos[pI] >= pP0 && lPos[pI] < pP1;
  }

  // set the position of an empty cell
  // does not modify "has_shift"
  // note: call only if the cell is empty
  void mPosSet(uint32_t pI, dgs_pos pP) {
    assert(i < pos.size());
    assert(pos[i] == -1);
    assert(seq[i].none());

    lPos[pI] = pP;

    sUsed.insert(pI);
  }

  void mExtSet(uint32_t pI, dgs_kv_cell_ext pP) {
    assert(i < ext.size());
    lExt[pI] = pP;
  }

  // pos[i] = pos[i] + d
  // sets "has_shift" to true
  // note: call only if the cell is not empty
  bool mPosAdd(uint32_t pI, dgs_pos pD) {
    assert(i < pos.size());
    assert(pos[i] != -1);

    mSeqPosRm(pI);

    lPos[pI] += pD;
    lShift[pI] += pD;

    vHasShift = true;

    if (lPos[pI] < 0) {
      lSeq[pI].reset();
      lPos[pI]   = -1;
      lShift[pI] = 0;

      sUsed.erase(pI);

      return true;
    }

    mSeqPosAdd(pI);

    return false;
  }

  // pos[i] = pos[i] / d
  // sets "has_shift" to true
  // note: call only if the cell is not empty
  void mPosDiv(uint32_t pI, int pD) {
    assert(i < pos.size());
    assert(pos[i] != -1);

    const dgs_pos cPOld = lPos[pI];

    mSeqPosRm(pI);

    lPos[pI] /= pD;
    lShift[pI] += cPOld - lPos[pI];

    mSeqPosAdd(pI);

    vHasShift = true;
  }

private:
  bool vHasShift = false;

  // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
  std::set<uint32_t> sUsed;

  std::vector<dgs_pos> lPos;

  // stores extra info per cell
  std::vector<dgs_kv_cell_ext> lExt;

  // this array accumulates any applied shifts to the pos array since the last reset_shift() call
  // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
  //
  //   cells.pos_add(x, shift_x);
  //   cells.pos_div(y, shift_y);
  //   ...
  //
  //   if (cells.has_shift()) {
  //      for (int i = 0; i < n; ++i) {
  //          auto shift_i = cells.get_shift(i);
  //          ...
  //      }
  //      cells.reset_shift();
  //   }
  //
  std::vector<dgs_pos> lShift;

  // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
  std::vector<seq_set_t> lSeq;

  // the set seq_pos[s] holds one (pos, cell) pair per cell that carries sequence s, ordered by position
  // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
  // and upper_bound() on a position finds the nearest cell of the sequence in logarithmic time
  //
  // the cell index is part of the key because a position can occur more than once for the same seq:
  //  - during performing a cache reuse via (rm + add)
  //  - some vision models have input embeddings with repeating positions
  //
  std::set<std::pair<dgs_pos, uint32_t>> sSeqPos[DGS_MAX_SEQ];

  // helper functions for updating `seq_pos`, once cell at a time:

  void mSeqPosDec(dgs_seq_id pS, uint32_t pI) {
    const auto n = sSeqPos[pS].erase({ lPos[pI], pI });
    assert(n == 1);
    GGML_UNUSED(n);
  }

  void mSeqPosInc(dgs_seq_id pS, uint32_t pI) { sSeqPos[pS].insert({ lPos[pI], pI }); }

  // remove cell i
  void mSeqPosRm(uint32_t pI) {
    for (int vS = 0; vS < DGS_MAX_SEQ; ++vS) {
      if (lSeq[pI].test(vS)) {
        mSeqPosDec(vS, pI);
      }
    }
  }

  // add cell i
  void mSeqPosAdd(uint32_t pI) {
    for (int vS = 0; vS < DGS_MAX_SEQ; ++vS) {
      if (lSeq[pI].test(vS)) {
        mSeqPosInc(vS, pI);
      }
    }
  }
};

using dgs_kv_cells_vec = std::vector<dgs_kv_cells>;
