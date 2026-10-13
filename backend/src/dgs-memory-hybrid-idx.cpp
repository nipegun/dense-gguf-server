#include "dgs-memory-hybrid-idx.h"

#include "dgs-batch.h"
#include "dgs-impl.h"
#include "dgs-io.h"
#include "dgs-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <type_traits>

//
// dgs_memory_hybrid_idx
//

dgs_memory_hybrid_idx::dgs_memory_hybrid_idx(const dgs_model &       pModel,
                                             /* attn */
                                             ggml_type               pTypeK,
                                             ggml_type               pTypeV,
                                             bool                    pVTrans,
                                             uint32_t                pKvSize,
                                             uint32_t                pNPad,
                                             uint32_t                pNSwa,
                                             dgs_swa_type            pSwaType,
                                             /* recurrent */
                                             ggml_type               pTypeR,
                                             ggml_type               pTypeS,
                                             uint32_t                pRsSize,
                                             /* common */
                                             uint32_t                pNSeqMax,
                                             uint32_t                pNRsSeq,
                                             bool                    pOffload,
                                             bool                    pUnified,
                                             /* layer filters */
                                             const layer_filter_cb & pFilterAttn,
                                             const layer_filter_cb & pFilterRecr,
                                             const layer_filter_cb & pFilterIdx) :
  dgs_memory_hybrid(pModel,
                    pTypeK,
                    pTypeV,
                    pVTrans,
                    pKvSize,
                    pNPad,
                    pNSwa,
                    pSwaType,
                    pTypeR,
                    pTypeS,
                    pRsSize,
                    pNSeqMax,
                    pNRsSeq,
                    pOffload,
                    pUnified,
                    pFilterAttn,
                    pFilterRecr),
  vHparamsIdx(pModel.hparams),
  cMemIdx(pFilterIdx == nullptr ? nullptr : [&] {
    // MQA with a single key head of indexer_head_size, as dgs_kv_cache_dsa shapes its own
    std::fill(vHparamsIdx.aNHeadKvArr.begin(), vHparamsIdx.aNHeadKvArr.end(), 1);
    // a k-pool indexer caches its per-token rows and the pooled key side by side
    // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
    vHparamsIdx.vNEmbdHeadKFull =
      pModel.hparams.vIndexerHeadSize * (pModel.hparams.vIndexerKpool > 0 ? pModel.hparams.vIndexerKpoolRow : 1);

    // the cached indexer keys are raw, rotation happens after pooling at read time, so a
    // K-shift must not rotate them while the stream copies in the same update still apply
    vHparamsIdx.vRopeType = cDgsRopeTypeNone;

    // fool dgs_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
    vHparamsIdx.vNEmbdHeadKMlaImpl = pModel.hparams.vIndexerHeadSize;
    vHparamsIdx.vNEmbdHeadVMlaImpl = pModel.hparams.vIndexerHeadSize;

    DGS_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, pKvSize);

    return new dgs_kv_cache(pModel, vHparamsIdx, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize, pNSeqMax, pNPad,
                            pNSwa, pSwaType, nullptr, pFilterIdx, nullptr, nullptr, "idx_");
  }()) {}

dgs_memory_context_ptr dgs_memory_hybrid_idx::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  // note: repeats dgs_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
  do {
    pBalloc.mSplitReset();

    // follow the recurrent pattern for creating the ubatch splits
    std::vector<dgs_ubatch> lUbatches;

    while (true) {
      dgs_ubatch vUbatch;

      if (pEmbdAll) {
        // if all tokens are output, split by sequence
        vUbatch = pBalloc.mSplitSeq(pNUbatch);
      } else {
        // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
        const bool cUnified = (mGetMemAttn()->mGetNStream() == 1);

        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
        //   so that the rollback snapshots remain valid
        const uint32_t cNRsSeq = mGetMemRecr()->vNRsSeq;

        vUbatch = pBalloc.mSplitEqual(pNUbatch, !cUnified, cNRsSeq > 0 ? cNRsSeq + 1 : 0);
      }

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    // prepare the recurrent batches first
    if (!mGetMemRecr()->mPrepare(lUbatches)) {
      // TODO: will the recurrent cache be in an undefined context at this point?
      DGS_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
      return std::make_unique<dgs_memory_hybrid_idx_context>(cDgsMemoryStatusFailedPrepare);
    }

    // prepare the attention cache
    auto lHeadsAttn = mGetMemAttn()->mPrepare(lUbatches);
    if (lHeadsAttn.empty()) {
      DGS_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
      return std::make_unique<dgs_memory_hybrid_idx_context>(cDgsMemoryStatusFailedPrepare);
    }

    // the indexer uses the attention cache's slot layout; a separate one can drift from it
    dgs_kv_cache::slot_info_vec_t lHeadsIdx;
    if (cMemIdx) {
      lHeadsIdx = lHeadsAttn;
    }

    return std::make_unique<dgs_memory_hybrid_idx_context>(this, std::move(lHeadsAttn), std::move(lHeadsIdx),
                                                           std::move(lUbatches));
  } while (false);

  return std::make_unique<dgs_memory_hybrid_idx_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_memory_hybrid_idx::mInitFull() {
  return std::make_unique<dgs_memory_hybrid_idx_context>(this);
}

dgs_memory_context_ptr dgs_memory_hybrid_idx::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_memory_hybrid_idx_context>(this, pLctx, pOptimize);
}

void dgs_memory_hybrid_idx::clear(bool data) {
  dgs_memory_hybrid::clear(data);

  if (cMemIdx) {
    cMemIdx->clear(data);
    mMemIdxStaleSet(-1, 0);
  }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void dgs_memory_hybrid_idx::mMemIdxStaleSet(dgs_seq_id pSeqId, dgs_pos pP0) {
  pP0 = std::max<dgs_pos>(pP0, 0);

  if (pSeqId < 0) {
    for (auto & vP : aMemIdxStale) {
      vP = std::min(vP, pP0);
    }

    return;
  }

  GGML_ASSERT(pSeqId < (dgs_seq_id) DGS_MAX_SEQ);

  aMemIdxStale[pSeqId] = std::min(aMemIdxStale[pSeqId], pP0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
dgs_pos dgs_memory_hybrid_idx::mMemIdxStalePos(dgs_seq_id pSeqId, dgs_pos pP0) const {
  if (pSeqId < 0 || pP0 <= cMemIdx->mSeqPosMin(pSeqId)) {
    return 0;
  }

  return pP0;
}

bool dgs_memory_hybrid_idx::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  // same order as dgs_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
  if (!mGetMemRecr()->mSeqRm(pSeqId, pP0, pP1)) {
    return false;
  }

  if (cMemIdx) {
    const dgs_pos cStale = mMemIdxStalePos(pSeqId, pP0);
    cMemIdx->mSeqRm(pSeqId, pP0, pP1);
    mMemIdxStaleSet(pSeqId, cStale);
  }

  return mGetMemAttn()->mSeqRm(pSeqId, pP0, pP1);
}

void dgs_memory_hybrid_idx::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool grouping
  GGML_ASSERT(pP0 <= 0 && pP1 < 0 && "partial seq_cp is not supported");

  dgs_memory_hybrid::mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);

  if (cMemIdx) {
    cMemIdx->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
    // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
    // pooled keys, the destination rebuilds its layout and re-pools into the same rows
    mMemIdxStaleSet(pSeqIdDst, 0);
  }
}

void dgs_memory_hybrid_idx::mSeqKeep(dgs_seq_id pSeqId) {
  dgs_memory_hybrid::mSeqKeep(pSeqId);

  if (cMemIdx) {
    cMemIdx->mSeqKeep(pSeqId);
    // every other sequence loses its cells, so their layouts must rebuild
    mMemIdxStaleSet(-1, 0);
  }
}

void dgs_memory_hybrid_idx::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  dgs_memory_hybrid::mSeqAdd(pSeqId, pP0, pP1, pShift);

  if (cMemIdx) {
    // a negative shift moves the cells below p0, so they regroup as well
    const dgs_pos cStale = mMemIdxStalePos(pSeqId, pShift < 0 ? pP0 + pShift : pP0);
    cMemIdx->mSeqAdd(pSeqId, pP0, pP1, pShift);
    mMemIdxStaleSet(pSeqId, cStale);
  }
}

void dgs_memory_hybrid_idx::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  dgs_memory_hybrid::mSeqDiv(pSeqId, pP0, pP1, pD);

  if (cMemIdx) {
    cMemIdx->mSeqDiv(pSeqId, pP0, pP1, pD);
    mMemIdxStaleSet(pSeqId, 0);
  }
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_memory_hybrid_idx::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = dgs_memory_hybrid::mMemoryBreakdown();

  if (cMemIdx) {
    for (const auto & vBuftSize : cMemIdx->mMemoryBreakdown()) {
      dMb[vBuftSize.first] += vBuftSize.second;
    }
  }

  return dMb;
}

void dgs_memory_hybrid_idx::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  dgs_memory_hybrid::mStateWrite(pIo, pSeqId, pFlags);

  // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
  // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
  if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    if (cMemIdx) {
      cMemIdx->mStateWrite(pIo, pSeqId, pFlags);
    }
  }
}

void dgs_memory_hybrid_idx::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  // note: repeats dgs_memory_hybrid::state_read
  // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

  // [TAG_HYBRID_IDX_SINFO]
  // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
  // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
  dgs_kv_cache::slot_info_vec_t lSinfosAttn;

  try {
    if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
      mGetMemAttn()->mStateReadSinfo(pIo, pSeqId, pFlags, cMemIdx ? &lSinfosAttn : nullptr, nullptr);
    }

    mGetMemRecr()->mStateRead(pIo, pSeqId, pFlags);

    // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
    if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
      if (cMemIdx) {
        cMemIdx->mStateReadSinfo(pIo, pSeqId, pFlags, nullptr, &lSinfosAttn);
        // the restore rewrites the cells behind the pool layout's back
        mMemIdxStaleSet(pSeqId, 0);
      }
    }

  } catch (...) {
    // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
    // drop what was being restored from all of them, which is a state they do agree on.
    mStateDrop(pSeqId);

    throw;
  }
}

void dgs_memory_hybrid_idx::mStateDrop(dgs_seq_id pSeqId) {
  // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
  if (pSeqId < 0) {
    clear(true);

    return;
  }

  mGetMemAttn()->mStateClear(pSeqId);
  mGetMemRecr()->mSeqRm(pSeqId, -1, -1);

  if (cMemIdx) {
    cMemIdx->mStateClear(pSeqId);
    mMemIdxStaleSet(pSeqId, 0);
  }
}

dgs_kv_cache * dgs_memory_hybrid_idx::mGetMemIdx() const {
  return cMemIdx.get();
}

//
// dgs_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> fDgsMemoryHybridIdxNs(const dgs_kv_cache::slot_info_vec_t & pSinfos) {
  std::vector<uint32_t> lRes;
  lRes.reserve(pSinfos.size());

  for (const auto & vSinfo : pSinfos) {
    lRes.push_back(vSinfo.vS1 - vSinfo.vS0 + 1);
  }

  return lRes;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct dgs_memory_hybrid_idx::kpool_layout {
  struct seq {
    dgs_pos                                   vPosMin = 0;
    uint32_t                                  vStrm   = 0;  // Stream holding this sequence's cells
    std::vector<std::pair<dgs_pos, uint32_t>> lCells;       // Position and stream local cell pairs, sorted by position.
    std::vector<uint32_t>                     lPools;

    // Where the pool scan stopped, so an append resumes instead of starting over.
    size_t vJNext = 0;
  };

  std::array<seq, DGS_MAX_SEQ> aSeqs;

  uint32_t vNPoolReal = 0;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct dgs_memory_hybrid_idx_context::kpool_state {
  std::vector<uint32_t> lIsNew;
  std::vector<uint32_t> lRepGen;  // per global cell, the generation that last marked a pool with that rep
  uint32_t              vGeneration = 0;

  uint32_t vNPoolReal = 0;
  uint32_t vNNew      = 0;
  uint32_t vNNewG     = 1;  // graph size of the new pool list, stable across decode steps
};

namespace {

// The last padded pool is always unused.
uint32_t fKpoolPad(uint32_t pNPool) {
  return std::max<uint32_t>(64u, GGML_PAD(pNPool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t fKpoolRank(const std::vector<std::pair<dgs_pos, uint32_t>> & pCells, dgs_pos pPos, uint32_t pCell) {
  auto lIt = std::lower_bound(pCells.begin(), pCells.end(), std::make_pair(pPos, pCell));
  return lIt != pCells.end() && lIt->second == pCell && lIt->first == pPos ? lIt - pCells.begin() : -1;
}

}  // namespace

dgs_memory_hybrid_idx::~dgs_memory_hybrid_idx() = default;

const dgs_memory_hybrid_idx::kpool_layout & dgs_memory_hybrid_idx::mKpoolLayoutGet() const {
  GGML_ASSERT(vKpoolLay != nullptr);

  return *vKpoolLay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const dgs_memory_hybrid_idx::kpool_layout & dgs_memory_hybrid_idx::mKpoolLayoutUpdate() {
  GGML_ASSERT(cMemIdx != nullptr);

  if (!vKpoolLay) {
    vKpoolLay = std::make_unique<kpool_layout>();
  }

  auto & vLay = *vKpoolLay;

  const uint32_t cKpool     = mGetKpool();
  const uint32_t cNStreamKv = cMemIdx->mGetNStream();
  const bool     cUnified   = cNStreamKv == 1;

  vLay.vNPoolReal = 0;

  for (dgs_seq_id vS = 0; vS < DGS_MAX_SEQ; ++vS) {
    auto & vSq = vLay.aSeqs[vS];

    // a non unified cache gives each sequence its own stream, with stream local cell indices
    if (!cUnified && vS >= (dgs_seq_id) cNStreamKv) {
      vSq = kpool_layout::seq();
      continue;
    }

    const auto & vCells = cMemIdx->mGetCells(cUnified ? 0 : vS);
    const auto & sSp    = vCells.mSeqPosGet(vS);

    vSq.vStrm = cUnified ? 0 : cMemIdx->mGetStream(vS);

    if (aMemIdxStale[vS] == cPosClean && !vSq.lCells.empty() && !sSp.empty() && vSq.vPosMin == sSp.begin()->first) {
      for (auto vIt = sSp.upper_bound(vSq.lCells.back()); vIt != sSp.end(); ++vIt) {
        vSq.lCells.push_back(*vIt);
      }
    }

    // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
    // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
    if (vSq.lCells.size() != sSp.size() || aMemIdxStale[vS] != cPosClean) {
      vSq.lCells.assign(sSp.begin(), sSp.end());
      vSq.lPools.clear();
      vSq.vJNext  = 0;
      vSq.vPosMin = sSp.empty() ? 0 : sSp.begin()->first;
    }

    // Pools start at the first valid token
    size_t vJ = vSq.vJNext;
    if (vHparamsIdx.vIndexerKpoolByOrder) {
      // consecutive cells in sequence order, whatever their positions
      for (; vJ + cKpool <= vSq.lCells.size(); vJ += cKpool) {
        vSq.lPools.push_back((uint32_t) vJ);
      }
    } else {
      while (vJ + cKpool <= vSq.lCells.size()) {
        const dgs_pos cP0 = vSq.lCells[vJ].first;
        if ((cP0 - vSq.vPosMin) % (dgs_pos) cKpool != 0) {
          ++vJ;
          continue;
        }
        bool vOk = true;
        for (uint32_t vK = 1; vK < cKpool; ++vK) {
          if (vSq.lCells[vJ + vK].first != cP0 + (dgs_pos) vK) {
            vOk = false;
            break;
          }
        }
        if (vOk) {
          vSq.lPools.push_back((uint32_t) vJ);
          vJ += cKpool;
        } else {
          ++vJ;
        }
      }
    }
    vSq.vJNext = vJ;

    vLay.vNPoolReal += (uint32_t) vSq.lPools.size();
  }

  return vLay;
}

dgs_memory_hybrid_idx_context::dgs_memory_hybrid_idx_context(dgs_memory_status pStatus) :
  dgs_memory_hybrid_context(pStatus) {}

dgs_memory_hybrid_idx_context::dgs_memory_hybrid_idx_context(dgs_memory_hybrid_idx * pMem) :
  dgs_memory_hybrid_context(pMem),
  vMem(pMem),
  // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
  // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
  lNsUbatch(pMem->mGetMemIdx() == nullptr ? std::vector<uint32_t>() :
                                            std::vector<uint32_t>{ pMem->mGetMemIdx()->mGetNStream() }),
  cCtxIdx(pMem->mGetMemIdx() == nullptr ? nullptr : new dgs_kv_cache_context(pMem->mGetMemIdx())) {
  if (mKpoolTrack()) {
    pMem->mKpoolLayoutUpdate();
    auto           vSt       = mKpoolBuildSizes();
    const auto *   vIdx      = pMem->mGetMemIdx();
    const uint64_t cNPoolMax = uint64_t(vIdx->mGetSize() / pMem->mGetKpool()) * vIdx->mGetNSeqMax();
    GGML_ASSERT(cNPoolMax <= UINT32_MAX - 64);
    vSt.vNPoolReal = std::max(vSt.vNPoolReal, uint32_t(cNPoolMax));
    vSt.vNNew      = vSt.vNPoolReal;
    vSt.vNNewG     = std::max(vSt.vNNew, 1u);
    vKpoolSt       = std::make_unique<kpool_state>(std::move(vSt));
    vIKpool        = 0;
  }
}

dgs_memory_hybrid_idx_context::dgs_memory_hybrid_idx_context(dgs_memory_hybrid_idx * pMem,
                                                             dgs_context *           pLctx,
                                                             bool                    pOptimize) :
  dgs_memory_hybrid_context(pMem, pLctx, pOptimize),
  vMem(pMem),
  // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
  cCtxIdx(pMem->mGetMemIdx() == nullptr ? nullptr : pMem->mGetMemIdx()->mInitUpdate(pLctx, pOptimize)) {}

dgs_memory_hybrid_idx_context::dgs_memory_hybrid_idx_context(dgs_memory_hybrid_idx * pMem,
                                                             slot_info_vec_t         pSinfosAttn,
                                                             slot_info_vec_t         pSinfosIdx,
                                                             std::vector<dgs_ubatch> pUbatches) :
  // note: the base copies the ubatches; ctx_idx gets a copy of its own
  dgs_memory_hybrid_context(pMem, std::move(pSinfosAttn), pUbatches),
  vMem(pMem),
  lNsUbatch(fDgsMemoryHybridIdxNs(pSinfosIdx)),
  lSinfosKpool(pMem->mGetMemIdx() != nullptr && pMem->mGetKpool() > 0 && pMem->mGetKpoolByOrder() ? pSinfosIdx :
                                                                                                    slot_info_vec_t()),
  cCtxIdx(pMem->mGetMemIdx() == nullptr ?
            nullptr :
            new dgs_kv_cache_context(pMem->mGetMemIdx(), std::move(pSinfosIdx), pUbatches)) {
  // Sequence edits force the touched positions to re-pool.
  aMemIdxStaleBatch = pMem->mMemIdxStaleGet();
}

dgs_memory_hybrid_idx_context::~dgs_memory_hybrid_idx_context() = default;

bool dgs_memory_hybrid_idx_context::mNext() {
  // Clear only after a successful ubatch.
  if (vICur == 0 && vMem != nullptr) {
    vMem->mMemIdxStaleClear();
  }

  if (cCtxIdx) {
    cCtxIdx->mNext();
  }

  ++vICur;

  return dgs_memory_hybrid_context::mNext();
}

bool dgs_memory_hybrid_idx_context::apply() {
  bool vRes = dgs_memory_hybrid_context::apply();

  if (cCtxIdx) {
    vRes = vRes & cCtxIdx->apply();
  }

  // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
  if (vRes && mKpoolTrack()) {
    vMem->mKpoolLayoutUpdate();
    if (!vKpoolSt) {
      vKpoolSt = std::make_unique<kpool_state>();
    }
    mKpoolBuildState(mGetUbatch());
    vIKpool = vICur;
  }

  return vRes;
}

bool dgs_memory_hybrid_idx_context::mKpoolTrack() const {
  // Derived from mem instead of being cached.
  return vMem != nullptr && vMem->mGetMemIdx() != nullptr && vMem->mGetKpool() > 0 && !lNsUbatch.empty();
}

const dgs_kv_cache_context * dgs_memory_hybrid_idx_context::mGetIdx() const {
  return static_cast<const dgs_kv_cache_context *>(cCtxIdx.get());
}

uint32_t dgs_memory_hybrid_idx_context::mGetNStream() const {
  GGML_ASSERT(vICur < lNsUbatch.size());

  return lNsUbatch[vICur];
}

dgs_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * pCtx, ggml_tensor * pK, int64_t pNEmbd) :
  vCtx(pCtx) {
  // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
  const int64_t cNTok = pK->ne[0] - pNEmbd;
  GGML_ASSERT(cNTok > 0 && cNTok % pNEmbd == 0);

  const int64_t cNCells = pK->ne[1] * pK->ne[2];

  // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
  vKeyGate = ggml_view_2d(pCtx, pK, cNTok, cNCells, pK->nb[1], 0);
  vPooled  = ggml_view_2d(pCtx, pK, pNEmbd, cNCells, pK->nb[1], ggml_row_size(pK->type, cNTok));
}

ggml_tensor * dgs_memory_hybrid_idx_context::kpool_access::mGatherKeyGate(ggml_tensor * pIdxs) const {
  return ggml_get_rows(vCtx, vKeyGate, pIdxs);
}

ggml_tensor * dgs_memory_hybrid_idx_context::kpool_access::mScatterPooled(ggml_tensor * pValues,
                                                                          ggml_tensor * pIdxs) const {
  return ggml_set_rows(vCtx, vPooled, pValues, pIdxs);
}

ggml_tensor * dgs_memory_hybrid_idx_context::kpool_access::mGatherPooled(ggml_tensor * pIdxs) const {
  return ggml_get_rows(vCtx, vPooled, pIdxs);
}

dgs_memory_hybrid_idx_context::kpool_access dgs_memory_hybrid_idx_context::mGetKpoolAccess(ggml_context * pCtx,
                                                                                           int32_t        pIl,
                                                                                           int64_t pNEmbd) const {
  GGML_ASSERT(vMem != nullptr && vMem->mGetMemIdx() != nullptr);

  return kpool_access(pCtx, vMem->mGetMemIdx()->mGetKStorage(pIl), pNEmbd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
dgs_memory_hybrid_idx_context::kpool_state dgs_memory_hybrid_idx_context::mKpoolBuildSizes() const {
  const auto & vLay = vMem->mKpoolLayoutGet();

  kpool_state vSt;
  vSt.vNPoolReal = vLay.vNPoolReal;

  return vSt;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void dgs_memory_hybrid_idx_context::mKpoolBuildState(const dgs_ubatch & pUbatch) {
  const auto & vLay = vMem->mKpoolLayoutGet();
  auto &       vSt  = *vKpoolSt;

  const auto *   vIdx    = vMem->mGetMemIdx();
  const uint32_t cKvSize = vIdx->mGetSize();
  const uint32_t cKpool  = vMem->mGetKpool();

  vSt.vNPoolReal = vLay.vNPoolReal;
  vSt.vNNew      = 0;
  if (++vSt.vGeneration == 0) {
    std::fill(vSt.lIsNew.begin(), vSt.lIsNew.end(), 0);
    std::fill(vSt.lRepGen.begin(), vSt.lRepGen.end(), 0);
    vSt.vGeneration = 1;
  }
  vSt.lIsNew.resize(vLay.vNPoolReal, 0);
  vSt.lRepGen.resize((size_t) cKvSize * vIdx->mGetNStream(), 0);

  std::array<uint32_t, DGS_MAX_SEQ> aPoolStart;

  // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
  // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
  auto vMark = [&](dgs_seq_id pS, size_t pK) {
    const auto & vSq  = vLay.aSeqs[pS];
    const size_t cRep = (size_t) vSq.vStrm * cKvSize + vSq.lCells[vSq.lPools[pK] + cKpool - 1].second;
    if (vSt.lRepGen[cRep] != vSt.vGeneration) {
      vSt.lRepGen[cRep]               = vSt.vGeneration;
      vSt.lIsNew[aPoolStart[pS] + pK] = vSt.vGeneration;
      ++vSt.vNNew;
    }
  };

  uint32_t vIp = 0;
  for (dgs_seq_id vS = 0; vS < DGS_MAX_SEQ; ++vS) {
    const auto & vSq = vLay.aSeqs[vS];
    aPoolStart[vS]   = vIp;
    vIp += (uint32_t) vSq.lPools.size();

    // A sequence edit invalidates only pools ending after the edited position.
    const dgs_pos cStaleFrom = vICur == 0 ? aMemIdxStaleBatch[vS] : dgs_memory_hybrid_idx::cPosClean;
    if (cStaleFrom == dgs_memory_hybrid_idx::cPosClean) {
      continue;
    }

    auto lFirst = std::lower_bound(vSq.lPools.begin(), vSq.lPools.end(), cStaleFrom,
                                   [&](uint32_t pJ, dgs_pos pP) { return vSq.lCells[pJ + cKpool - 1].first < pP; });
    for (auto lIt = lFirst; lIt != vSq.lPools.end(); ++lIt) {
      vMark(vS, lIt - vSq.lPools.begin());
    }
  }
  GGML_ASSERT(vIp == vSt.lIsNew.size());

  // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
  const bool     cByOrder = vMem->mGetKpoolByOrder();
  const auto *   vSinfo   = cByOrder ? &lSinfosKpool[vICur] : nullptr;
  const uint32_t cNTps    = cByOrder ? (uint32_t) vSinfo->size() : 0;

  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    const dgs_pos cP = pUbatch.vPos[vI];
    for (int32_t vK = 0; vK < pUbatch.vNSeqId[vI]; ++vK) {
      const dgs_seq_id cS  = pUbatch.vSeqId[vI][vK];
      const auto &     vSq = vLay.aSeqs[cS];
      if (cByOrder) {
        const int64_t cR = fKpoolRank(vSq.lCells, cP, vSinfo->lIdxs[vI / cNTps][vI % cNTps]);
        GGML_ASSERT(cR >= 0);
        if ((size_t) cR / cKpool < vSq.lPools.size()) {
          vMark(cS, (size_t) cR / cKpool);
        }
        continue;
      }
      auto lIt = std::upper_bound(vSq.lPools.begin(), vSq.lPools.end(), cP,
                                  [&](dgs_pos pPos, uint32_t pJ) { return pPos < vSq.lCells[pJ].first; });
      if (lIt == vSq.lPools.begin()) {
        continue;
      }
      --lIt;
      if (cP <= vSq.lCells[*lIt + cKpool - 1].first) {
        vMark(cS, lIt - vSq.lPools.begin());
      }
    }
  }

  // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
  // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
  const uint32_t cNPoolMax = cKvSize / cKpool * vIdx->mGetNSeqMax();
  const uint32_t cBound    = pUbatch.vNTokens / cKpool + pUbatch.vNSeqsUnq;
  vSt.vNNewG = std::max({ vSt.vNNew, 1u, std::min({ cBound, fKpoolPad(vSt.vNPoolReal) - 1, cNPoolMax }) });
}

const dgs_memory_hybrid_idx_context::kpool_state & dgs_memory_hybrid_idx_context::mKpoolCur() const {
  GGML_ASSERT(vKpoolSt != nullptr && vIKpool == vICur && "k-pool state read before apply()");

  return *vKpoolSt;
}

uint32_t dgs_memory_hybrid_idx_context::mGetNKpool() const {
  return fKpoolPad(mKpoolCur().vNPoolReal);
}

uint32_t dgs_memory_hybrid_idx_context::mGetNKpoolNew() const {
  return mKpoolCur().vNNewG;
}

void dgs_memory_hybrid_idx_context::mSetInputKpool(ggml_tensor *      pPoolCells,
                                                   ggml_tensor *      pPoolIdxs,
                                                   ggml_tensor *      pPoolMask,
                                                   ggml_tensor *      pTailIdxs,
                                                   ggml_tensor *      pSelMask,
                                                   ggml_tensor *      pNewPoolIdxs,
                                                   ggml_tensor *      pNewPoolRep,
                                                   const dgs_ubatch * pUbatch,
                                                   ggml_tensor *      pNewPoolPos) const {
  GGML_ASSERT(vMem != nullptr && vMem->mGetMemIdx() != nullptr);
  GGML_ASSERT(ggml_backend_buffer_is_host(pPoolCells->buffer));
  GGML_ASSERT(ggml_backend_buffer_is_host(pPoolIdxs->buffer));
  GGML_ASSERT(ggml_backend_buffer_is_host(pPoolMask->buffer));
  GGML_ASSERT(ggml_backend_buffer_is_host(pTailIdxs->buffer));

  const uint32_t cKpool = vMem->mGetKpool();
  const uint32_t cNKv   = mGetIdx()->mGetNKv();

  const auto & vSt  = mKpoolCur();
  const auto & vLay = vMem->mKpoolLayoutGet();

  const uint32_t cNTokens = pUbatch->vNTokens;
  const uint32_t cNPool   = (uint32_t) pPoolCells->ne[0];
  const uint32_t cNNew    = vSt.vNNew;
  // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
  const uint32_t cNNewG   = vSt.vNNewG;

  const bool cByOrder = vMem->mGetKpoolByOrder();

  GGML_ASSERT(cNPool == fKpoolPad(vSt.vNPoolReal));
  GGML_ASSERT(vSt.lIsNew.size() == vSt.vNPoolReal);
  GGML_ASSERT(pPoolMask->ne[0] == (int64_t) cNPool && pPoolMask->ne[1] == (int64_t) cNTokens);
  GGML_ASSERT(pTailIdxs->ne[0] == (int64_t) cKpool - 1 && pTailIdxs->ne[1] == (int64_t) cNTokens);
  GGML_ASSERT(pPoolIdxs->ne[0] == (int64_t) cKpool && pPoolIdxs->ne[1] == (int64_t) cNPool);
  GGML_ASSERT(ggml_backend_buffer_is_host(pNewPoolIdxs->buffer));
  GGML_ASSERT(pNewPoolIdxs->ne[0] == (int64_t) cKpool && pNewPoolIdxs->ne[1] == (int64_t) cNNewG);
  // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
  GGML_ASSERT(pNewPoolRep != nullptr && ggml_backend_buffer_is_host(pNewPoolRep->buffer));
  GGML_ASSERT(pNewPoolRep->ne[0] == (int64_t) cNNewG);
  if (pNewPoolPos != nullptr) {
    GGML_ASSERT(ggml_backend_buffer_is_host(pNewPoolPos->buffer));
    GGML_ASSERT(pNewPoolPos->ne[0] == 4 * (int64_t) cNNewG);
  }

  const uint32_t cKvSize    = vMem->mGetMemIdx()->mGetSize();
  const uint32_t cNStreamKv = vMem->mGetMemIdx()->mGetNStream();

  auto vGcell = [&](const dgs_memory_hybrid_idx::kpool_layout::seq & pSq, uint32_t pCell) {
    return (int64_t) pSq.vStrm * cKvSize + pCell;
  };

  // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
  std::vector<uint8_t> lSeqInUb(DGS_MAX_SEQ, 0);
  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    for (int32_t vK = 0; vK < pUbatch->vNSeqId[vI]; ++vK) {
      lSeqInUb[pUbatch->vSeqId[vI][vK]] = 1;
    }
  }

  // a cell of this ubatch, written before any read, so the padded pools read a finite K row
  int64_t vDummyCell = 0;
  {
    const dgs_seq_id cS  = pUbatch->vSeqId[0][0];
    const auto &     vSq = vLay.aSeqs[cS];
    auto             lIt = std::lower_bound(vSq.lCells.begin(), vSq.lCells.end(), std::make_pair(pUbatch->vPos[0], 0u));
    GGML_ASSERT(lIt != vSq.lCells.end() && lIt->first == pUbatch->vPos[0]);
    vDummyCell = vGcell(vSq, lIt->second);
  }

  // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
  std::vector<int64_t> lRank;
  if (cByOrder) {
    const auto &   vSinfo = lSinfosKpool[vICur];
    const uint32_t cNTps  = (uint32_t) vSinfo.size();

    lRank.resize(cNTokens);
    for (uint32_t vI = 0; vI < cNTokens; ++vI) {
      lRank[vI] =
        fKpoolRank(vLay.aSeqs[pUbatch->vSeqId[vI][0]].lCells, pUbatch->vPos[vI], vSinfo.lIdxs[vI / cNTps][vI % cNTps]);
      GGML_ASSERT(lRank[vI] >= 0);
    }
  }

  // padding and absent cells point at the n_kv sentinel row, one past the live cells
  const int32_t cSentinel = (int32_t) cNKv;

  float *  vGm   = nullptr;
  uint32_t vNSel = 0;
  uint32_t vNTop = 0;  // Pools per token in the selection.
  if (pSelMask != nullptr) {
    GGML_ASSERT(ggml_backend_buffer_is_host(pSelMask->buffer));
    GGML_ASSERT(pSelMask->type == GGML_TYPE_F32);
    GGML_ASSERT(pSelMask->ne[3] == (int64_t) cNTokens && pSelMask->ne[1] == 1 && pSelMask->ne[2] == 1);
    vNSel = (uint32_t) pSelMask->ne[0];
    // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
    vNTop = vNSel / cKpool;
    GGML_ASSERT(vNSel % cKpool == 0 || vNSel % cKpool == cKpool - 1);
    vGm = (float *) pSelMask->data;
  }

  // pools are laid out per sequence
  std::vector<uint32_t> lSeqPoolStart(DGS_MAX_SEQ, 0);
  std::vector<dgs_pos>  lPoolEnd;
  lPoolEnd.reserve(cNPool);

  int32_t * vPcell = (int32_t *) pPoolCells->data;
  int32_t * vPidx  = (int32_t *) pPoolIdxs->data;
  int32_t * vNidx  = (int32_t *) pNewPoolIdxs->data;
  int64_t * vNrep  = (int64_t *) pNewPoolRep->data;
  int32_t * vNpos  = pNewPoolPos != nullptr ? (int32_t *) pNewPoolPos->data : nullptr;

  if (vNpos != nullptr) {
    std::fill(vNpos, vNpos + 4 * cNNewG, 0);
  }

  uint32_t vINew = 0;
  for (dgs_seq_id vS = 0; vS < DGS_MAX_SEQ; ++vS) {
    const auto & vSq  = vLay.aSeqs[vS];
    lSeqPoolStart[vS] = (uint32_t) lPoolEnd.size();

    const bool cInert = cNStreamKv > 1 && !lSeqInUb[vS];

    for (size_t vPi = 0; vPi < vSq.lPools.size(); ++vPi) {
      const uint32_t cJ  = vSq.lPools[vPi];
      const uint32_t cIp = (uint32_t) lPoolEnd.size();
      GGML_ASSERT(cIp + 1 < cNPool);

      // The pooled key lives in the last member's row.
      const uint32_t cRep = vSq.lCells[cJ + cKpool - 1].second;
      vPcell[cIp]         = (int32_t) vGcell(vSq, cRep);

      for (uint32_t vK = 0; vK < cKpool; ++vK) {
        vPidx[(size_t) cIp * cKpool + vK] = cInert ? cSentinel : (int32_t) vSq.lCells[cJ + vK].second;
      }

      if (vSt.lIsNew[cIp] == vSt.vGeneration) {
        GGML_ASSERT(vINew < cNNew);
        for (uint32_t vK = 0; vK < cKpool; ++vK) {
          vNidx[(size_t) vINew * cKpool + vK] = (int32_t) vGcell(vSq, vSq.lCells[cJ + vK].second);
        }
        vNrep[vINew] = vGcell(vSq, cRep);
        if (vNpos != nullptr) {
          // a pooled key is rotated to the M-RoPE position of its first member
          const uint32_t cC         = vSq.lCells[cJ].second;
          const auto &   vE         = vMem->mGetMemIdx()->mGetCells(vS).mExtGet(cC);
          vNpos[0 * cNNewG + vINew] = vSq.lCells[cJ].first;
          vNpos[1 * cNNewG + vINew] = vE.vY;
          vNpos[2 * cNNewG + vINew] = vE.vX;
          vNpos[3 * cNNewG + vINew] = vSq.lCells[cJ].first;
        }
        ++vINew;
      }

      lPoolEnd.push_back(vSq.lCells[cJ + cKpool - 1].first);
    }
  }
  GGML_ASSERT(vINew == cNNew);

  // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
  // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
  if (cNNewG > cNNew) {
    std::vector<int64_t> lReps(vPcell, vPcell + lPoolEnd.size());
    std::sort(lReps.begin(), lReps.end());

    int64_t vPadCell = 0;
    for (uint32_t vI = cNNew; vI < cNNewG; ++vI, ++vPadCell) {
      while (std::binary_search(lReps.begin(), lReps.end(), vPadCell)) {
        ++vPadCell;
      }
      GGML_ASSERT(vPadCell < (int64_t) cKvSize * cNStreamKv);
      for (uint32_t vK = 0; vK < cKpool; ++vK) {
        vNidx[(size_t) vI * cKpool + vK] = (int32_t) vPadCell;
      }
      vNrep[vI] = vPadCell;
    }
  }

  const uint32_t cNPoolReal = (uint32_t) lPoolEnd.size();
  for (uint32_t vIp = cNPoolReal; vIp < cNPool; ++vIp) {
    vPcell[vIp] = (int32_t) vDummyCell;  // pool_cells always addresses the K storage
    for (uint32_t vK = 0; vK < cKpool; ++vK) {
      vPidx[(size_t) vIp * cKpool + vK] = cSentinel;
    }
  }

  // a pool is visible when it belongs to the token's sequence and ends at or before it
  auto vFillMask = [&](auto * data) {
    using T       = std::remove_pointer_t<decltype(data)>;
    const T cKeep = fDgsCast<T>(0.0f);
    const T cDrop = fDgsCast<T>(-INFINITY);

    for (uint32_t vI = 0; vI < cNTokens; ++vI) {
      const dgs_seq_id cS = pUbatch->vSeqId[vI][0];
      const dgs_pos    cP = pUbatch->vPos[vI];

      T * vRow = data + (size_t) vI * cNPool;
      std::fill(vRow, vRow + cNPool, cDrop);

      const uint32_t cP0 = lSeqPoolStart[cS];
      const uint32_t cP1 = cP0 + (uint32_t) vLay.aSeqs[cS].lPools.size();
      const uint32_t cNv =
        cByOrder ?
          std::min(cP1 - cP0, (uint32_t) ((lRank[vI] + 1) / cKpool)) :
          (uint32_t) (std::upper_bound(lPoolEnd.begin() + cP0, lPoolEnd.begin() + cP1, cP) - (lPoolEnd.begin() + cP0));
      std::fill(vRow + cP0, vRow + cP0 + cNv, cKeep);

      // Finite visible pools occupy the first min(nv, n_top) ranked slots.
      if (vGm != nullptr) {
        const uint32_t cNvc  = std::min(cNv, vNTop);
        float *        vGrow = vGm + (size_t) vI * vNSel;
        std::fill(vGrow, vGrow + (size_t) cNvc * cKpool, 0.0f);
        std::fill(vGrow + (size_t) cNvc * cKpool, vGrow + (size_t) vNTop * cKpool, -INFINITY);
      }
    }
  };
  if (pPoolMask->type == GGML_TYPE_F16) {
    vFillMask((ggml_fp16_t *) pPoolMask->data);
  } else {
    vFillMask((float *) pPoolMask->data);
  }

  int32_t * vTidx = (int32_t *) pTailIdxs->data;
  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    const dgs_seq_id cS  = pUbatch->vSeqId[vI][0];
    const dgs_pos    cP  = pUbatch->vPos[vI];
    const auto &     vSq = vLay.aSeqs[cS];

    const uint32_t cNTail =
      cByOrder ? (uint32_t) ((lRank[vI] + 1) % cKpool) : (uint32_t) ((cP - vSq.vPosMin + 1) % (dgs_pos) cKpool);

    for (uint32_t vK = 0; vK < cKpool - 1; ++vK) {
      int32_t vCell = cSentinel;
      bool    vReal = false;
      if (vK < cNTail && cByOrder) {
        const uint32_t cC = vSq.lCells[lRank[vI] - vK].second;
        vCell             = (int32_t) cC;
        vReal             = true;
      } else if (vK < cNTail) {
        const dgs_pos cPt = cP - (dgs_pos) vK;
        auto          lIt = std::lower_bound(vSq.lCells.begin(), vSq.lCells.end(), std::make_pair(cPt, 0u));
        if (lIt != vSq.lCells.end() && lIt->first == cPt) {
          vCell = (int32_t) lIt->second;
          vReal = true;
        }
      }
      vTidx[(size_t) vI * (cKpool - 1) + vK] = vCell;

      if (vGm != nullptr && vNSel % cKpool != 0) {
        vGm[(size_t) vI * vNSel + (size_t) vNTop * cKpool + vK] = vReal ? 0.0f : -INFINITY;
      }
    }
  }
}
