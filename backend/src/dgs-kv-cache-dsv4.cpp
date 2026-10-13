#include "dgs-kv-cache-dsv4.h"

#include "dgs-batch.h"
#include "dgs-impl.h"
#include "dgs-io.h"
#include "dgs-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

static constexpr uint32_t cDsv4CsaRatio = 4;
static constexpr uint32_t cDsv4HcaRatio = 128;

static constexpr uint32_t cDsv4StateMagic       = 0x34565344;  // DSV4
static constexpr uint32_t cDsv4StateVersion     = 1;
static constexpr uint32_t cDsv4StateModeFull    = 0;
static constexpr uint32_t cDsv4StateModePartial = 1;
static constexpr uint32_t cDsv4KCacheStateVer   = 2;
static constexpr uint32_t cDsv4CompStateVer     = 1;

static uint32_t fDsv4CompSize(uint32_t pKvSize, uint32_t pRatio) {
  return std::max<uint32_t>(1, (pKvSize + pRatio - 1) / pRatio);
}

static void fDsv4ClearTensorStream(ggml_tensor * pTensor, uint32_t stream) {
  GGML_ASSERT(ggml_is_contiguous(pTensor));
  GGML_ASSERT(pTensor->ne[3] == 1);
  GGML_ASSERT(stream < (uint32_t) pTensor->ne[2]);

  const size_t cStreamSize = pTensor->nb[2];
  ggml_backend_tensor_memset(pTensor, 0, stream * cStreamSize, cStreamSize);
}

static uint32_t fDsv4StateNUsedKRows(dgs_pos pPosMax, uint32_t pRatio, uint32_t pKvSize) {
  if (pPosMax < 0) {
    return 0;
  }

  const uint64_t cNRows = ((uint64_t) pPosMax + 1) / pRatio;

  return (uint32_t) std::min<uint64_t>(pKvSize, cNRows);
}

static int64_t fDsv4StreamOffset(uint32_t pNStream, dgs_seq_id pSeqId, uint32_t size) {
  if (pNStream <= 1) {
    return 0;
  }
  if (pSeqId < 0 || (uint32_t) pSeqId >= pNStream) {
    throw std::runtime_error("DSV4 sequence id out of stream range");
  }

  return (int64_t) pSeqId * size;
}

static bool fDsv4UbatchHasCoupled(const dgs_ubatch & pUbatch) {
  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    if (pUbatch.vNSeqId[vI] > 1) {
      return true;
    }
  }

  return false;
}

static bool fDsv4TokenHasSeq(const dgs_ubatch & pUbatch, uint32_t pI, dgs_seq_id pSeqId) {
  for (int32_t vS = 0; vS < pUbatch.vNSeqId[pI]; ++vS) {
    if (pUbatch.vSeqId[pI][vS] == pSeqId) {
      return true;
    }
  }

  return false;
}

static dgs_ubatch fDsv4BuildRawWriteUbatch(const dgs_ubatch & pUbatch) {
  if (!fDsv4UbatchHasCoupled(pUbatch)) {
    return pUbatch;
  }
  ASSERT_EMBD_OR_TOKEN(pUbatch);
  if (pUbatch.vEmbd) {
    throw std::runtime_error("DSV4 coupled embedding ubatches are not supported");
  }

  std::vector<uint32_t> lCounts(pUbatch.vNSeqsUnq, 0);
  uint32_t              vNTokens = 0;
  for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
    const dgs_seq_id cSeqId = pUbatch.vSeqIdUnq[vS];
    for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
      if (fDsv4TokenHasSeq(pUbatch, vI, cSeqId)) {
        ++lCounts[vS];
        ++vNTokens;
      }
    }
  }

  if (vNTokens == 0) {
    return pUbatch;
  }

  const uint32_t cNSeqTokens = lCounts[0];
  for (uint32_t vS = 1; vS < lCounts.size(); ++vS) {
    if (lCounts[vS] != cNSeqTokens) {
      throw std::runtime_error("DSV4 coupled raw writes require equal sequence lengths");
    }
  }

  auto data = std::make_shared<dgs_ubatch::data_t>();
  data->lPos.resize((size_t) vNTokens * pUbatch.vNPos);
  data->lNSeqId.reserve(vNTokens);
  data->lSeqId.reserve(vNTokens);
  data->lSeqIdData.reserve(vNTokens);
  data->lSeqIdUnq.assign(pUbatch.vSeqIdUnq, pUbatch.vSeqIdUnq + pUbatch.vNSeqsUnq);
  data->lSeqIdx.assign(DGS_MAX_SEQ, -1);
  data->lOutput.assign(vNTokens, 0);
  if (pUbatch.token) {
    data->token.reserve(vNTokens);
  }

  for (uint32_t vS = 0; vS < data->lSeqIdUnq.size(); ++vS) {
    data->lSeqIdx[data->lSeqIdUnq[vS]] = vS;
  }

  for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
    const dgs_seq_id cSeqId = pUbatch.vSeqIdUnq[vS];
    for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
      if (!fDsv4TokenHasSeq(pUbatch, vI, cSeqId)) {
        continue;
      }

      const uint32_t cDst = data->lNSeqId.size();
      if (pUbatch.token) {
        data->token.push_back(pUbatch.token[vI]);
      }
      for (uint32_t vP = 0; vP < pUbatch.vNPos; ++vP) {
        data->lPos[(size_t) vP * vNTokens + cDst] = pUbatch.vPos[(size_t) vP * pUbatch.vNTokens + vI];
      }
      data->lNSeqId.push_back(1);
      data->lSeqIdData.push_back(cSeqId);
    }
  }

  for (uint32_t vI = 0; vI < vNTokens; ++vI) {
    data->lSeqId.push_back(&data->lSeqIdData[vI]);
  }

  dgs_ubatch vRes{
    /*.b_equal_seqs =*/true,
    /*.n_tokens     =*/vNTokens,
    /*.n_seq_tokens =*/cNSeqTokens,
    /*.n_seqs       =*/pUbatch.vNSeqsUnq,
    /*.n_seqs_unq   =*/pUbatch.vNSeqsUnq,
    /*.n_pos        =*/pUbatch.vNPos,
    /*.token        =*/data->token.empty() ? nullptr : data->token.data(),
    /*.embd         =*/nullptr,
    /*.embd_state   =*/nullptr,
    /*.pos          =*/data->lPos.data(),
    /*.n_seq_id     =*/data->lNSeqId.data(),
    /*.seq_id       =*/data->lSeqId.data(),
    /*.seq_id_unq   =*/data->lSeqIdUnq.data(),
    /*.seq_idx      =*/data->lSeqIdx.data(),
    /*.output       =*/data->lOutput.data(),
    /*.type         =*/nullptr,
    /*.decision_order =*/nullptr,
    /*.data         =*/data,
  };

  return vRes;
}

static std::vector<dgs_ubatch> fDsv4BuildRawWriteUbatches(const std::vector<dgs_ubatch> & pUbatches) {
  std::vector<dgs_ubatch> lRes;
  lRes.reserve(pUbatches.size());
  for (const dgs_ubatch & vUbatch : pUbatches) {
    lRes.push_back(fDsv4BuildRawWriteUbatch(vUbatch));
  }
  return lRes;
}

static bool fDsv4BatchHasCoupled(const dgs_batch & pBatch) {
  if (!pBatch.vNSeqId) {
    return false;
  }

  for (int32_t vI = 0; vI < pBatch.vNTokens; ++vI) {
    if (pBatch.vNSeqId[vI] > 1) {
      return true;
    }
  }

  return false;
}

static int64_t fDsv4CompGraphNStream(const dgs_ubatch & pUbatch, uint32_t pNStream) {
  // Coupled sequence sets must stay in one graph stream because their
  // compressed state is shared. Independent per-seq state can fan out.
  if (pNStream <= 1 || pUbatch.vNSeqsUnq <= 1 || fDsv4UbatchHasCoupled(pUbatch)) {
    return 1;
  }

  return pUbatch.vNSeqsUnq;
}

static void fDsv4StateSrcStreamRange(uint32_t pNStream, dgs_seq_id pSeqId, uint32_t & pS0, uint32_t & pNs) {
  if (pSeqId >= 0 && pNStream > 1) {
    if ((uint32_t) pSeqId >= pNStream) {
      throw std::runtime_error("DSV4 state sequence id out of stream range");
    }

    pS0 = (uint32_t) pSeqId;
    pNs = 1;
    return;
  }

  pS0 = 0;
  pNs = pSeqId >= 0 ? 1 : pNStream;
}

static void fDsv4StateDstStreamRange(uint32_t pNStream, dgs_seq_id pSeqId, uint32_t pNs, uint32_t & pS0) {
  if (pSeqId >= 0) {
    if (pNs != 1) {
      throw std::runtime_error("DSV4 sequence state stream count mismatch");
    }
    if (pNStream > 1 && (uint32_t) pSeqId >= pNStream) {
      throw std::runtime_error("DSV4 state sequence id out of stream range");
    }

    pS0 = pNStream > 1 ? (uint32_t) pSeqId : 0;
    return;
  }

  if (pNs != pNStream) {
    throw std::runtime_error("DSV4 full state stream count mismatch");
  }

  pS0 = 0;
}

static void fDsv4StateWriteTensorStreams(dgs_io_write_i &              pIo,
                                         ggml_tensor *                 pTensor,
                                         uint32_t                      pTensorRows,
                                         uint32_t                      pNRows,
                                         uint32_t                      pS0,
                                         uint32_t                      pNs,
                                         const std::vector<uint32_t> * pStreamIds = nullptr) {
  const int32_t  cTypeI   = (int32_t) pTensor->type;
  const uint64_t cNe0     = pTensor->ne[0];
  const uint64_t cRows    = pNRows;
  const uint64_t cRowSize = ggml_row_size(pTensor->type, pTensor->ne[0]);

  if (pNRows > pTensorRows) {
    throw std::runtime_error("DSV4 state tensor row count exceeds storage");
  }

  pIo.write(&cTypeI, sizeof(cTypeI));
  pIo.write(&cNe0, sizeof(cNe0));
  pIo.write(&cRows, sizeof(cRows));
  pIo.write(&cRowSize, sizeof(cRowSize));

  const size_t cStreamStride = (size_t) pTensorRows * cRowSize;
  const size_t size          = (size_t) pNRows * cRowSize;
  if (size == 0) {
    return;
  }

  if (pStreamIds && pStreamIds->size() != pNs) {
    throw std::runtime_error("DSV4 state tensor stream map size mismatch");
  }

  for (uint32_t vS = 0; vS < pNs; ++vS) {
    const uint32_t stream = pStreamIds ? (*pStreamIds)[vS] : pS0 + vS;
    if ((int64_t) stream >= pTensor->ne[2]) {
      throw std::runtime_error("DSV4 state tensor stream out of range");
    }
    const size_t cOffset = (size_t) stream * cStreamStride;
    pIo.mWriteTensor(pTensor, cOffset, size);
  }
}

static void fDsv4StateReadTensorStreams(dgs_io_read_i & pIo,
                                        ggml_tensor *   pTensor,
                                        uint32_t        pTensorRows,
                                        uint32_t        pNRows,
                                        uint32_t        pS0,
                                        uint32_t        pNs) {
  int32_t  vTypeIRef;
  uint64_t vNe0Ref;
  uint64_t vRowsRef;
  uint64_t vRowSizeRef;

  pIo.read(&vTypeIRef, sizeof(vTypeIRef));
  pIo.read(&vNe0Ref, sizeof(vNe0Ref));
  pIo.read(&vRowsRef, sizeof(vRowsRef));
  pIo.read(&vRowSizeRef, sizeof(vRowSizeRef));

  const int32_t  cTypeI   = (int32_t) pTensor->type;
  const uint64_t cNe0     = pTensor->ne[0];
  const uint64_t cRows    = pNRows;
  const uint64_t cRowSize = ggml_row_size(pTensor->type, pTensor->ne[0]);

  if (cTypeI != vTypeIRef || cNe0 != vNe0Ref || cRows != vRowsRef || cRowSize != vRowSizeRef) {
    throw std::runtime_error("DSV4 state tensor metadata mismatch");
  }
  if (pNRows > pTensorRows) {
    throw std::runtime_error("DSV4 state tensor row count exceeds storage");
  }

  const size_t cStreamStride = (size_t) pTensorRows * cRowSize;
  const size_t size          = (size_t) pNRows * cRowSize;
  if (size == 0) {
    return;
  }

  for (uint32_t vS = 0; vS < pNs; ++vS) {
    const size_t cOffset = (size_t) (pS0 + vS) * cStreamStride;
    pIo.mReadTensor(pTensor, cOffset, size);
  }
}

static void fDsv4StateWriteKCache(dgs_io_write_i &     pIo,
                                  const dgs_kv_cache * pKv,
                                  dgs_seq_id           pSeqId,
                                  dgs_state_seq_flags  pFlags,
                                  uint32_t             pNRows) {
  GGML_UNUSED(pFlags);

  uint32_t vS0;
  uint32_t vNs;
  fDsv4StateSrcStreamRange(pKv->mGetNStream(), pSeqId, vS0, vNs);

  const uint32_t cVersion  = cDsv4KCacheStateVer;
  const uint32_t cKvSize   = pKv->mGetSize();
  const auto     lLayerIds = pKv->mGetLayerIds();
  const uint32_t cNLayer   = lLayerIds.size();

  if (pNRows > cKvSize) {
    throw std::runtime_error("DSV4 K-cache state row count exceeds cache size");
  }

  pIo.write(&cVersion, sizeof(cVersion));
  pIo.write(&pNRows, sizeof(pNRows));
  pIo.write(&vNs, sizeof(vNs));
  pIo.write(&cNLayer, sizeof(cNLayer));

  for (uint32_t vIl : lLayerIds) {
    pIo.write(&vIl, sizeof(vIl));
    fDsv4StateWriteTensorStreams(pIo, pKv->mGetKStorage(vIl), cKvSize, pNRows, vS0, vNs);
  }
}

static void fDsv4StateReadKCache(dgs_io_read_i &     pIo,
                                 dgs_kv_cache *      pKv,
                                 dgs_seq_id          pSeqId,
                                 dgs_state_seq_flags pFlags) {
  GGML_UNUSED(pFlags);

  uint32_t vVersion;
  uint32_t vNRowsRef;
  uint32_t vNs;
  uint32_t vNLayerRef;

  pIo.read(&vVersion, sizeof(vVersion));
  pIo.read(&vNRowsRef, sizeof(vNRowsRef));
  pIo.read(&vNs, sizeof(vNs));
  pIo.read(&vNLayerRef, sizeof(vNLayerRef));

  if (vVersion != 1 && vVersion != cDsv4KCacheStateVer) {
    throw std::runtime_error("DSV4 K-cache state version mismatch");
  }

  const uint32_t cKvSize = pKv->mGetSize();
  if (vVersion == 1 && vNRowsRef != cKvSize) {
    DGS_LOG_INFO("kv size ref %d kv %d\n", vNRowsRef, cKvSize);
    throw std::runtime_error("DSV4 K-cache state size mismatch");
  }
  if (vNRowsRef > cKvSize) {
    DGS_LOG_INFO("kv rows ref %d kv %d\n", vNRowsRef, cKvSize);
    throw std::runtime_error("DSV4 K-cache state size mismatch");
  }

  uint32_t vS0;
  fDsv4StateDstStreamRange(pKv->mGetNStream(), pSeqId, vNs, vS0);

  const auto lLayerIds = pKv->mGetLayerIds();
  if (vNLayerRef != lLayerIds.size()) {
    throw std::runtime_error("DSV4 K-cache layer count mismatch");
  }

  for (uint32_t vIl : lLayerIds) {
    uint32_t vIlRef;
    pIo.read(&vIlRef, sizeof(vIlRef));
    if (vIlRef != vIl) {
      throw std::runtime_error("DSV4 K-cache layer id mismatch");
    }

    fDsv4StateReadTensorStreams(pIo, pKv->mGetKStorage(vIl), cKvSize, vNRowsRef, vS0, vNs);
  }
}

static std::string fDsv4PlanPositions(const std::vector<int32_t> & pValues) {
  std::ostringstream vSs;
  vSs << "[";
  for (size_t vI = 0; vI < pValues.size(); ++vI) {
    if (vI > 0) {
      vSs << ", ";
    }
    vSs << pValues[vI];
  }
  vSs << "]";
  return vSs.str();
}

static dgs_kv_cache_dsv4_context::comp_plan fDsv4BuildCompPlan(const dgs_ubatch &            pUbatch,
                                                               uint32_t                      pRatio,
                                                               bool                          pOverlap,
                                                               uint32_t                      pStateSize,
                                                               uint32_t                      pKvSize,
                                                               uint32_t                      pNStream,
                                                               uint32_t                      pNRsSeq,
                                                               const std::vector<uint32_t> & pRsIdx) {
  dgs_kv_cache_dsv4_context::comp_plan vPlan;
  vPlan.lNVisible.resize(pUbatch.vNTokens);
  vPlan.vNStream = fDsv4CompGraphNStream(pUbatch, pNStream);

  // n_stream is the persistent cache/state layout; plan.n_stream is the
  // graph view for this ubatch and can be a subset of those streams.
  if (pNStream <= 1 && pUbatch.vNSeqsUnq > 1) {
    throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
  }

  const int64_t cStateRows = (int64_t) pStateSize * pNStream;

  struct persist_row {
    int32_t vDst;
    int32_t vSrc;
    dgs_pos vPos;
  };

  std::vector<persist_row> lPersistRows;

  // For the overlap compressor, build_overlap_compressed_kv_from_state() consumes
  // state_read_idxs as two contiguous halves: the first ratio*n_blocks entries are
  // the "previous-window" gather indices for every block, followed by the
  // "current-window" indices for every block. Collect them separately here and
  // append cur after prev once the loop has visited all completed blocks
  std::vector<int32_t> lOverlapPrevReads;
  std::vector<int32_t> lOverlapCurReads;

  std::map<std::pair<dgs_seq_id, dgs_pos>, int64_t> dCurrTokenIdxMap;
  std::map<dgs_seq_id, uint32_t>                    dStateWriteCounts;

  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    for (int32_t vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
      dCurrTokenIdxMap[std::make_pair(pUbatch.vSeqId[vI][vS], pUbatch.vPos[vI])] = vI;
    }
  }

  const auto cStateSourceIdx = [&](dgs_seq_id pSeqId, dgs_pos pPos) -> int32_t {
    if (pPos < 0) {
      // The overlap compressor needs a zero/-inf source for the first
      // block's previous half. The graph appends that row after the
      // current-ubatch scratch rows.
      return (int32_t) (cStateRows + pUbatch.vNTokens);
    }

    const auto cKey = std::make_pair(pSeqId, pPos);
    if (dCurrTokenIdxMap.find(cKey) != dCurrTokenIdxMap.end()) {
      return (int32_t) (cStateRows + dCurrTokenIdxMap.at(cKey));
    }

    const int64_t cStreamOff = fDsv4StreamOffset(pNStream, pSeqId, pStateSize);
    return (int32_t) (cStreamOff + pPos % pStateSize);
  };

  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    const dgs_pos cPos = pUbatch.vPos[vI];

    if (cPos < 0) {
      continue;
    }

    vPlan.lStatePos.push_back((int32_t) (cPos % pRatio));

    const int64_t cNVisible = (int64_t) (cPos + 1) / pRatio;
    vPlan.lNVisible[vI]     = (int32_t) cNVisible;
    vPlan.vNKv              = std::max(vPlan.vNKv, cNVisible);

    for (int32_t vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
      const dgs_seq_id cSeqId     = pUbatch.vSeqId[vI][vS];
      const int64_t    cStreamOff = fDsv4StreamOffset(pNStream, cSeqId, pStateSize);
      const int32_t    cStateIdx  = (int32_t) (cStreamOff + cPos % pStateSize);

      const auto lIt = std::find_if(lPersistRows.begin(), lPersistRows.end(),
                                    [cStateIdx](const persist_row & pRow) { return pRow.vDst == cStateIdx; });
      if (lIt == lPersistRows.end()) {
        lPersistRows.push_back({ cStateIdx, (int32_t) vI, cPos });
      } else if (cPos > lIt->vPos) {
        lIt->vSrc = (int32_t) vI;
        lIt->vPos = cPos;
      }

      if ((cPos + 1) % pRatio != 0) {
        continue;
      }

      const dgs_pos cSourceStart = cPos + 1 - pRatio;
      const int64_t cCacheOff    = fDsv4StreamOffset(pNStream, cSeqId, pKvSize);

      vPlan.lStateWriteIdxs.push_back(cCacheOff + cPos / pRatio);
      vPlan.lStateWritePos.push_back((int32_t) cSourceStart);
      ++dStateWriteCounts[cSeqId];

      if (pOverlap) {
        const dgs_pos cPrevStart = cSourceStart - pRatio;

        for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
          lOverlapPrevReads.push_back(cStateSourceIdx(cSeqId, cPrevStart + vJ));
        }
        for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
          lOverlapCurReads.push_back(cStateSourceIdx(cSeqId, cSourceStart + vJ));
        }
      } else {
        for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
          vPlan.lStateReadIdxs.push_back(cStateSourceIdx(cSeqId, cSourceStart + vJ));
        }
      }
    }
  }

  if (pRatio == cDsv4CsaRatio && !vPlan.lStatePos.empty()) {
    assert(kv_size > 0);

    // Pad each stream to the reserve plan's block count.
    const auto cAppendDummyBlock = [&](dgs_seq_id pSeqId, uint32_t pI) {
      const int64_t cCacheOff  = fDsv4StreamOffset(pNStream, pSeqId, pKvSize);
      const int32_t cSourceIdx = cStateSourceIdx(pSeqId, pUbatch.vPos[pI]);

      vPlan.lStateWriteIdxs.push_back(cCacheOff + pKvSize - 1);
      vPlan.lStateWritePos.push_back(0);

      if (pOverlap) {
        for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
          lOverlapPrevReads.push_back(cSourceIdx);
          lOverlapCurReads.push_back(cSourceIdx);
        }
      } else {
        for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
          vPlan.lStateReadIdxs.push_back(cSourceIdx);
        }
      }
    };

    if (fDsv4UbatchHasCoupled(pUbatch)) {
      if (vPlan.lStateWriteIdxs.empty()) {
        uint32_t vI = 0;
        while (vI < pUbatch.vNTokens && pUbatch.vPos[vI] < 0) {
          ++vI;
        }
        assert(i < ubatch.n_tokens);
        cAppendDummyBlock(pUbatch.vSeqId[vI][0], vI);
      }
    } else {
      const uint32_t cNBlocks = (std::max<uint32_t>(1, pUbatch.vNSeqTokens) + pRatio - 1) / pRatio;

      for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
        const dgs_seq_id cSeqId   = pUbatch.vSeqIdUnq[vS];
        const uint32_t   cNWrites = dStateWriteCounts[cSeqId];
        if (cNWrites >= cNBlocks) {
          continue;
        }
        if (cNWrites + 1 != cNBlocks) {
          throw std::runtime_error("DSV4 CSA sequence positions are not contiguous");
        }

        uint32_t vI = 0;
        while (vI < pUbatch.vNTokens && (pUbatch.vPos[vI] < 0 || !fDsv4TokenHasSeq(pUbatch, vI, cSeqId))) {
          ++vI;
        }
        assert(i < ubatch.n_tokens);
        cAppendDummyBlock(cSeqId, vI);
      }
    }
  }

  if (pRatio == cDsv4HcaRatio && !vPlan.lStatePos.empty() && vPlan.lStateWriteIdxs.empty()) {
    assert(kv_size > 0);
    // the last slot must not be live, or the dummy write would corrupt it;
    // a full stream implies a completed block, which implies real writes
    assert(plan.n_kv < (int64_t) kv_size);

    // Keep the compress/write ops in the graph when no HCA block completes
    // in this ubatch. The dummy block writes to the last cache slot and is
    // masked out.
    uint32_t vI = 0;
    while (vI < pUbatch.vNTokens && pUbatch.vPos[vI] < 0) {
      ++vI;
    }
    assert(i < ubatch.n_tokens);

    const dgs_seq_id cSeqId     = pUbatch.vSeqId[vI][0];
    const int64_t    cCacheOff  = fDsv4StreamOffset(pNStream, cSeqId, pKvSize);
    const int32_t    cSourceIdx = cStateSourceIdx(cSeqId, pUbatch.vPos[vI]);

    vPlan.lStateWriteIdxs.push_back(cCacheOff + pKvSize - 1);
    vPlan.lStateWritePos.push_back(0);

    for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
      vPlan.lStateReadIdxs.push_back(cSourceIdx);
    }
  }

  if (pOverlap) {
    // [ all blocks' prev-window indices | all blocks' cur-window indices ]
    vPlan.lStateReadIdxs.reserve(lOverlapPrevReads.size() + lOverlapCurReads.size());
    vPlan.lStateReadIdxs.insert(vPlan.lStateReadIdxs.end(), lOverlapPrevReads.begin(), lOverlapPrevReads.end());
    vPlan.lStateReadIdxs.insert(vPlan.lStateReadIdxs.end(), lOverlapCurReads.begin(), lOverlapCurReads.end());
  }

  // Keep the mask (and with it the compressed-attention branch) present even
  // before the first block is visible, so the graph topology never changes.
  // Padded slots are masked out; comp cache buffers are zero-initialized.
  vPlan.vNKv = std::max<int64_t>(GGML_PAD(vPlan.vNKv, 256u), 256);

  std::sort(lPersistRows.begin(), lPersistRows.end(),
            [](const persist_row & pA, const persist_row & pB) { return pA.vDst < pB.vDst; });

  for (const persist_row & vRow : lPersistRows) {
    vPlan.lStatePersistSrcIdxs.push_back(vRow.vSrc);
    vPlan.lStatePersistDstIdxs.push_back(vRow.vDst);
  }

  if (pNRsSeq > 0) {
    // Emit restore/snapshot entries for all layout streams so that the
    // graph tensor sizes do not depend on the ubatch's sequence count.
    // Streams not present in the ubatch get no-op entries.
    for (uint32_t stream = 0; stream < pNStream; ++stream) {
      dgs_seq_id vSeqId = -1;
      if (pNStream == 1) {
        // a unified stream serves any single sequence
        vSeqId = pUbatch.vNSeqsUnq > 0 ? pUbatch.vSeqIdUnq[0] : -1;
      } else {
        for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
          if (pUbatch.vSeqIdUnq[vS] == (dgs_seq_id) stream) {
            vSeqId = pUbatch.vSeqIdUnq[vS];
            break;
          }
        }
      }

      const int64_t  cStreamOff = (int64_t) stream * pStateSize;
      const uint32_t cRollback  = vSeqId >= 0 && (uint32_t) vSeqId < pRsIdx.size() ? pRsIdx[vSeqId] : 0;
      // Keep the restore graph fixed-width when no rollback is pending.
      const int64_t  cSrcPlane  = cRollback > 0 && cRollback <= pNRsSeq ? (int64_t) cRollback * cStateRows : 0;
      for (uint32_t vR = 0; vR < pStateSize; ++vR) {
        vPlan.lStateRestoreSrcIdxs.push_back((int32_t) (cSrcPlane + cStreamOff + vR));
        vPlan.lStateRestoreDstIdxs.push_back((int32_t) (cStreamOff + vR));
      }

      std::vector<uint32_t> lTokenIdxs;
      lTokenIdxs.reserve(pUbatch.vNTokens);
      if (vSeqId >= 0) {
        for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
          if (fDsv4TokenHasSeq(pUbatch, vI, vSeqId)) {
            lTokenIdxs.push_back(vI);
          }
        }
      }

      const uint32_t cNSeqTokens = (uint32_t) lTokenIdxs.size();
      const int64_t  cScratchOff = (int64_t) cStateRows * (1 + pNRsSeq);
      for (uint32_t vD = 1; vD <= pNRsSeq; ++vD) {
        const int64_t  cDstPlane = (int64_t) vD * cStateRows;
        const uint32_t cPrefix   = vD <= cNSeqTokens ? cNSeqTokens - vD : 0;

        for (uint32_t vR = 0; vR < pStateSize; ++vR) {
          int32_t vSrc = (int32_t) (cStreamOff + vR);

          for (uint32_t vJ = 0; vJ < cPrefix; ++vJ) {
            const uint32_t cITok = lTokenIdxs[vJ];
            if (pUbatch.vPos[cITok] >= 0 && (uint32_t) (pUbatch.vPos[cITok] % pStateSize) == vR) {
              vSrc = (int32_t) (cScratchOff + cITok);
            }
          }

          if (cNSeqTokens == 0) {
            // no-op: copy the snapshot plane onto itself
            vSrc = (int32_t) (cDstPlane + cStreamOff + vR);
          }

          vPlan.lStateSnapshotSrcIdxs.push_back(vSrc);
          vPlan.lStateSnapshotDstIdxs.push_back((int32_t) (cDstPlane + cStreamOff + vR));
        }
      }
    }
  }

  static const bool cDebug = []() {
    const char * vEnv = getenv("DGS_DSV4_COMPRESS_DEBUG");
    return vEnv && atoi(vEnv) > 0;
  }();

  if (cDebug) {
    DGS_LOG_DEBUG("%s: ratio=%u, n_tokens=%u, n_seqs_unq=%u, state_persist_dst=%s, state_write_pos=%s\n", __func__,
                  pRatio, pUbatch.vNTokens, pUbatch.vNSeqsUnq, fDsv4PlanPositions(vPlan.lStatePersistDstIdxs).c_str(),
                  fDsv4PlanPositions(vPlan.lStateWritePos).c_str());
    for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
      const dgs_seq_id cSeqId    = pUbatch.vSeqIdUnq[vS];
      const uint32_t   cRollback = cSeqId >= 0 && (uint32_t) cSeqId < pRsIdx.size() ? pRsIdx[cSeqId] : 0;
      DGS_LOG_DEBUG("%s:   seq %d pos [%d, %d] rollback=%u\n", __func__, cSeqId, pUbatch.vPos[0],
                    pUbatch.vPos[pUbatch.vNTokens - 1], cRollback);
    }
  }

  return vPlan;
}

static std::vector<dgs_kv_cache_dsv4_context::comp_plan> fDsv4BuildCompPlans(const std::vector<dgs_ubatch> & pUbatches,
                                                                             uint32_t                        pRatio,
                                                                             bool                            pOverlap,
                                                                             uint32_t                        pStateSize,
                                                                             uint32_t                        pKvSize,
                                                                             uint32_t                        pNStream,
                                                                             uint32_t                        pNRsSeq,
                                                                             const std::vector<uint32_t> &   pRsIdx) {
  std::vector<dgs_kv_cache_dsv4_context::comp_plan> lPlans;
  lPlans.reserve(pUbatches.size());

  // the first ubatch touching a seq consumes its rollback restore
  std::vector<uint32_t> lRs(pRsIdx);
  for (const dgs_ubatch & vUbatch : pUbatches) {
    lPlans.push_back(fDsv4BuildCompPlan(vUbatch, pRatio, pOverlap, pStateSize, pKvSize, pNStream, pNRsSeq, lRs));

    for (uint32_t vS = 0; vS < vUbatch.vNSeqsUnq; ++vS) {
      const dgs_seq_id cSeqId = vUbatch.vSeqIdUnq[vS];
      if (cSeqId >= 0 && (size_t) cSeqId < lRs.size()) {
        lRs[cSeqId] = 0;
      }
    }
  }

  return lPlans;
}

static dgs_kv_cache::slot_info_vec_t fDsv4BuildCompSinfos(const std::vector<dgs_ubatch> & pUbatches,
                                                          uint32_t                        pNStream) {
  dgs_kv_cache::slot_info_vec_t lSinfos;
  lSinfos.reserve(pUbatches.size());

  for (const dgs_ubatch & vUbatch : pUbatches) {
    if (pNStream <= 1 && vUbatch.vNSeqsUnq > 1) {
      throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
    }

    const uint32_t          cNs = (uint32_t) fDsv4CompGraphNStream(vUbatch, pNStream);
    dgs_kv_cache::slot_info vSinfo;
    vSinfo.vS0 = pNStream > 1 ? DGS_MAX_SEQ : 0;
    vSinfo.vS1 = 0;
    vSinfo.resize(cNs);

    for (uint32_t vS = 0; vS < cNs; ++vS) {
      const dgs_seq_id cSeqId = pNStream > 1 ? vUbatch.vSeqIdUnq[vS] : 0;
      const uint32_t   cStrm  = (uint32_t) fDsv4StreamOffset(pNStream, cSeqId, 1);

      vSinfo.vS0       = std::min(vSinfo.vS0, cStrm);
      vSinfo.vS1       = std::max(vSinfo.vS1, cStrm);
      vSinfo.lStrm[vS] = cStrm;
      vSinfo.lIdxs[vS].resize(1, 0);
    }

    if (pNStream > 1 && vSinfo.vS1 - vSinfo.vS0 + 1 != cNs) {
      throw std::runtime_error("DSV4 compressed streams are not contiguous in ubatch");
    }

    lSinfos.push_back(std::move(vSinfo));
  }

  return lSinfos;
}

static dgs_kv_cache::slot_info_vec_t fDsv4BuildRawReadSinfos(const dgs_kv_cache::slot_info_vec_t & pSinfosWrite,
                                                             const std::vector<dgs_ubatch> &       pUbatches) {
  dgs_kv_cache::slot_info_vec_t lSinfos;
  lSinfos.reserve(pUbatches.size());

  for (size_t vI = 0; vI < pUbatches.size(); ++vI) {
    const dgs_ubatch & vUbatch     = pUbatches[vI];
    const auto &       vSinfoWrite = pSinfosWrite[vI];

    if (!fDsv4UbatchHasCoupled(vUbatch)) {
      lSinfos.push_back(vSinfoWrite);
      continue;
    }

    const dgs_seq_id cSeqId   = vUbatch.vSeqId[0][0];
    uint32_t         vIStream = 0;
    for (; vIStream < vSinfoWrite.mNStream(); ++vIStream) {
      if (vSinfoWrite.lStrm[vIStream] == cSeqId) {
        break;
      }
    }
    if (vIStream == vSinfoWrite.mNStream()) {
      throw std::runtime_error("DSV4 raw write stream not found for coupled read");
    }

    dgs_kv_cache::slot_info vSinfo;
    vSinfo.vS0 = vSinfoWrite.lStrm[vIStream];
    vSinfo.vS1 = vSinfoWrite.lStrm[vIStream];
    vSinfo.resize(1);
    vSinfo.lStrm[0] = vSinfoWrite.lStrm[vIStream];
    vSinfo.lIdxs[0] = vSinfoWrite.lIdxs[vIStream];
    lSinfos.push_back(std::move(vSinfo));
  }

  return lSinfos;
}

static dgs_kv_cache_dsv4_context::comp_plan fDsv4BuildReserveCompPlan(const dgs_ubatch & pUbatch,
                                                                      uint32_t           pRatio,
                                                                      bool               pOverlap,
                                                                      uint32_t           pStateSize,
                                                                      uint32_t           pKvSize,
                                                                      uint32_t           pNStream,
                                                                      uint32_t           pNRsSeq) {
  dgs_kv_cache_dsv4_context::comp_plan vPlan;
  vPlan.lNVisible.resize(pUbatch.vNTokens);
  vPlan.vNStream = fDsv4CompGraphNStream(pUbatch, pNStream);
  vPlan.vNKv     = pKvSize;

  if (pUbatch.vNTokens == 0) {
    return vPlan;
  }

  // worst case over every seq split: sum of per-seq ceil(tokens/ratio) is at
  // most floor(n_tokens/ratio) + n_seqs
  const uint32_t cNSeqs   = std::max<uint32_t>(1, pUbatch.vNSeqs);
  const size_t   cNBlocks = (size_t) pUbatch.vNTokens / pRatio + cNSeqs;

  const uint64_t cStateRows = (uint64_t) pStateSize * pNStream;
  const size_t   cNPersist  = (size_t) std::min<uint64_t>(pUbatch.vNTokens, cStateRows);
  const size_t   cNRestore  = pNRsSeq > 0 ? (size_t) pStateSize * pNStream : 0;
  const size_t   cNSnapshot = (size_t) pNRsSeq * pStateSize * pNStream;

  vPlan.lStatePos.resize(pUbatch.vNTokens);
  vPlan.lStatePersistSrcIdxs.resize(cNPersist);
  vPlan.lStatePersistDstIdxs.resize(cNPersist);
  vPlan.lStateRestoreSrcIdxs.resize(cNRestore);
  vPlan.lStateRestoreDstIdxs.resize(cNRestore);
  vPlan.lStateSnapshotSrcIdxs.resize(cNSnapshot);
  vPlan.lStateSnapshotDstIdxs.resize(cNSnapshot);
  vPlan.lStateReadIdxs.resize((pOverlap ? 2u : 1u) * pRatio * cNBlocks);
  vPlan.lStateWriteIdxs.resize(cNBlocks);
  vPlan.lStateWritePos.resize(cNBlocks);

  return vPlan;
}

static void fDsv4MakeKOnly(dgs_hparams & pHparams) {
  // dgs_kv_cache uses hparams.is_mla() to allocate K-only storage.
  pHparams.vNEmbdHeadKMlaImpl = pHparams.n_embd_head_k();
  pHparams.vNEmbdHeadVMlaImpl = pHparams.n_embd_head_k();
}

//
// dgs_dsv4_comp_state
//

dgs_dsv4_comp_state::dgs_dsv4_comp_state(const dgs_model &                     pModel,
                                         bool                                  pOffload,
                                         bool                                  pUnified,
                                         uint32_t                              pNSeqMax,
                                         uint32_t                              pRatio,
                                         uint32_t                              pStateSize,
                                         uint32_t                              pNEmbdState,
                                         uint32_t                              pNRsSeq,
                                         const char *                          name,
                                         const dgs_memory_i::layer_filter_cb & pFilter) :
  cRatio(pRatio),
  cStateSize(pStateSize),
  cNEmbdState(pNEmbdState),
  cNStream(pUnified ? 1 : pNSeqMax),
  cNRsSeq(pNRsSeq) {
  const dgs_hparams & vHparams = pModel.hparams;

  struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t & pLhs, const ggml_backend_buffer_type_t & pRhs) const {
      return strcmp(ggml_backend_buft_name(pLhs), ggml_backend_buft_name(pRhs)) < 0;
    }
  };

  std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> dCtxMap;

  auto vCtxForBuft = [&](ggml_backend_buffer_type_t pBuft) -> ggml_context * {
    auto vIt = dCtxMap.find(pBuft);
    if (vIt == dCtxMap.end()) {
      ggml_init_params vParams = {
        /*.mem_size   =*/size_t(2u * (1 + cNStream) * vHparams.n_layer() * ggml_tensor_overhead()),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
      };

      ggml_context * vCtx = ggml_init(vParams);
      if (!vCtx) {
        return nullptr;
      }

      dCtxMap.emplace(pBuft, vCtx);

      return vCtx;
    }

    return vIt->second.get();
  };

  for (uint32_t vIl = 0; vIl < vHparams.n_layer(); ++vIl) {
    if (pFilter && !pFilter(vIl)) {
      continue;
    }

    const char * vDevName = "CPU";

    ggml_backend_buffer_type_t vBuft = ggml_backend_cpu_buffer_type();

    if (pOffload) {
      auto * vDev = pModel.mDevLayer(vIl);
      vBuft       = ggml_backend_dev_buffer_type(vDev);

      vDevName = ggml_backend_dev_name(vDev);
    }

    DGS_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, vIl, vDevName);

    ggml_context * vCtx = vCtxForBuft(vBuft);
    if (!vCtx) {
      throw std::runtime_error("failed to create ggml context for DSV4 compressor state");
    }

    const uint32_t cNPlanes = cNStream * (1 + pNRsSeq);
    ggml_tensor *  vKv      = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pNEmbdState, pStateSize, cNPlanes);
    ggml_tensor *  vScore   = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pNEmbdState, pStateSize, cNPlanes);

    ggml_format_name(vKv, "dsv4_%s_state_kv_l%d", name, vIl);
    ggml_format_name(vScore, "dsv4_%s_state_score_l%d", name, vIl);

    std::vector<ggml_tensor *> lKvStream;
    std::vector<ggml_tensor *> lScoreStream;

    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      lKvStream.push_back(ggml_view_2d(vCtx, vKv, pNEmbdState, pStateSize, vKv->nb[1], vS * vKv->nb[2]));
      lScoreStream.push_back(ggml_view_2d(vCtx, vScore, pNEmbdState, pStateSize, vScore->nb[1], vS * vScore->nb[2]));
    }

    dMapLayerIds[vIl] = lLayers.size();

    lLayers.push_back({ vIl, vKv, vScore, std::move(lKvStream), std::move(lScoreStream) });
  }

  for (auto & [buft, ctx] : dCtxMap) {
    ggml_backend_buffer_t vBuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
    if (!vBuf) {
      throw std::runtime_error("failed to allocate buffer for DSV4 compressor state");
    }

    ggml_backend_buffer_clear(vBuf, 0);

    DGS_LOG_INFO("%s: %10s DSV4 %s state buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf), name,
                 ggml_backend_buffer_get_size(vBuf) / 1024.0 / 1024.0);

    lCtxsBufs.emplace_back(std::move(ctx), vBuf);
  }

  DGS_LOG_INFO("%s: %s ratio = %u, state = %u x %u, streams = %u, rs_seq = %u, layers = %zu, size = %7.2f MiB\n",
               __func__, name, pRatio, pStateSize, pNEmbdState, cNStream, pNRsSeq, lLayers.size(),
               mTotalSize() / 1024.0 / 1024.0);
}

void dgs_dsv4_comp_state::clear(dgs_seq_id pSeqId, bool data) {
  if (!data) {
    return;
  }

  if (pSeqId >= 0) {
    GGML_ASSERT((uint32_t) pSeqId < cNStream);

    for (const auto & layer : lLayers) {
      for (uint32_t vD = 0; vD <= cNRsSeq; ++vD) {
        const uint32_t stream = vD * cNStream + (uint32_t) pSeqId;
        fDsv4ClearTensorStream(layer.vKv, stream);
        fDsv4ClearTensorStream(layer.vScore, stream);
      }
    }
    return;
  }

  for (auto & [_, buf] : lCtxsBufs) {
    ggml_backend_buffer_clear(buf.get(), 0);
  }
}

void dgs_dsv4_comp_state::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst) {
  GGML_ASSERT(pSeqIdSrc >= 0 && (uint32_t) pSeqIdSrc < cNStream);
  GGML_ASSERT(pSeqIdDst >= 0 && (uint32_t) pSeqIdDst < cNStream);

  if (pSeqIdSrc == pSeqIdDst) {
    return;
  }

  clear(pSeqIdDst, true);

  vScInfo.lSsrc.push_back((uint32_t) pSeqIdSrc);
  vScInfo.lSdst.push_back((uint32_t) pSeqIdDst);
}

void dgs_dsv4_comp_state::mApplyCopies(const stream_copy_info & pScInfo) const {
  for (size_t vI = 0; vI < pScInfo.lSsrc.size(); ++vI) {
    const uint32_t cSsrc = pScInfo.lSsrc[vI];
    const uint32_t cSdst = pScInfo.lSdst[vI];

    for (const auto & layer : lLayers) {
      ggml_backend_tensor_copy(layer.lKvStream[cSsrc], layer.lKvStream[cSdst]);
      ggml_backend_tensor_copy(layer.lScoreStream[cSsrc], layer.lScoreStream[cSdst]);
    }
  }
}

uint32_t dgs_dsv4_comp_state::mGetRatio() const {
  return cRatio;
}

uint32_t dgs_dsv4_comp_state::mGetStateSize() const {
  return cStateSize;
}

uint32_t dgs_dsv4_comp_state::mGetNStream() const {
  return cNStream;
}

uint32_t dgs_dsv4_comp_state::mGetNRsSeq() const {
  return cNRsSeq;
}

uint32_t dgs_dsv4_comp_state::mGetNRows() const {
  return cStateSize * cNStream;
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_dsv4_comp_state::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dRet;
  for (const auto & [_, buf] : lCtxsBufs) {
    ggml_backend_buffer_type_t vBuft = ggml_backend_buffer_get_type(buf.get());
    dRet[vBuft] += ggml_backend_buffer_get_size(buf.get());
  }
  return dRet;
}

void dgs_dsv4_comp_state::mStateWrite(dgs_io_write_i &              pIo,
                                      dgs_seq_id                    pSeqId,
                                      dgs_state_seq_flags           pFlags,
                                      const std::vector<uint32_t> & pRsIdx) const {
  GGML_UNUSED(pFlags);

  uint32_t vS0;
  uint32_t vNs;
  fDsv4StateSrcStreamRange(cNStream, pSeqId, vS0, vNs);

  std::vector<uint32_t> lStreamIds(vNs);
  for (uint32_t vS = 0; vS < vNs; ++vS) {
    const uint32_t cSeq = pSeqId >= 0 ? (uint32_t) pSeqId : vS0 + vS;
    if (cSeq >= pRsIdx.size() || pRsIdx[cSeq] > cNRsSeq) {
      throw std::runtime_error("DSV4 recurrent state rollback index out of range");
    }
    lStreamIds[vS] = pRsIdx[cSeq] * cNStream + vS0 + vS;
  }

  const uint32_t cVersion = cDsv4CompStateVer;
  const uint32_t cNLayer  = lLayers.size();

  pIo.write(&cVersion, sizeof(cVersion));
  pIo.write(&cRatio, sizeof(cRatio));
  pIo.write(&cStateSize, sizeof(cStateSize));
  pIo.write(&cNEmbdState, sizeof(cNEmbdState));
  pIo.write(&vNs, sizeof(vNs));
  pIo.write(&cNLayer, sizeof(cNLayer));

  for (const auto & layer : lLayers) {
    pIo.write(&layer.vIl, sizeof(layer.vIl));

    fDsv4StateWriteTensorStreams(pIo, layer.vKv, cStateSize, cStateSize, vS0, vNs, &lStreamIds);
    fDsv4StateWriteTensorStreams(pIo, layer.vScore, cStateSize, cStateSize, vS0, vNs, &lStreamIds);
  }
}

void dgs_dsv4_comp_state::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  GGML_UNUSED(pFlags);

  uint32_t vVersion;
  uint32_t vRatioRef;
  uint32_t vStateSizeRef;
  uint32_t vNEmbdStateRef;
  uint32_t vNs;
  uint32_t vNLayerRef;

  pIo.read(&vVersion, sizeof(vVersion));
  pIo.read(&vRatioRef, sizeof(vRatioRef));
  pIo.read(&vStateSizeRef, sizeof(vStateSizeRef));
  pIo.read(&vNEmbdStateRef, sizeof(vNEmbdStateRef));
  pIo.read(&vNs, sizeof(vNs));
  pIo.read(&vNLayerRef, sizeof(vNLayerRef));

  if (vVersion != cDsv4CompStateVer) {
    throw std::runtime_error("DSV4 compressor state version mismatch");
  }
  if (vRatioRef != cRatio || vStateSizeRef != cStateSize || vNEmbdStateRef != cNEmbdState) {
    throw std::runtime_error("DSV4 compressor state metadata mismatch");
  }
  if (vNLayerRef != lLayers.size()) {
    throw std::runtime_error("DSV4 compressor state layer count mismatch");
  }

  uint32_t vS0;
  fDsv4StateDstStreamRange(cNStream, pSeqId, vNs, vS0);

  for (const auto & layer : lLayers) {
    uint32_t vIlRef;
    pIo.read(&vIlRef, sizeof(vIlRef));
    if (vIlRef != layer.vIl) {
      throw std::runtime_error("DSV4 compressor state layer id mismatch");
    }

    fDsv4StateReadTensorStreams(pIo, layer.vKv, cStateSize, cStateSize, vS0, vNs);
    fDsv4StateReadTensorStreams(pIo, layer.vScore, cStateSize, cStateSize, vS0, vNs);
  }
}

ggml_tensor * dgs_dsv4_comp_state::mGetKvAll(ggml_context * pCtx, int32_t pIl) const {
  const int32_t cIds   = dMapLayerIds.at(pIl);
  ggml_tensor * vState = lLayers[cIds].vKv;

  return ggml_view_2d(pCtx, vState, vState->ne[0], mGetNRows() * (1 + cNRsSeq), vState->nb[1], 0);
}

ggml_tensor * dgs_dsv4_comp_state::mGetScoreAll(ggml_context * pCtx, int32_t pIl) const {
  const int32_t cIds   = dMapLayerIds.at(pIl);
  ggml_tensor * vState = lLayers[cIds].vScore;

  return ggml_view_2d(pCtx, vState, vState->ne[0], mGetNRows() * (1 + cNRsSeq), vState->nb[1], 0);
}

ggml_tensor * dgs_dsv4_comp_state::mGetKv(ggml_context * pCtx, int32_t pIl) const {
  ggml_tensor * vState   = mGetKvAll(pCtx, pIl);
  const size_t  cRowSize = ggml_row_size(vState->type, vState->ne[0]);

  return ggml_view_2d(pCtx, vState, vState->ne[0], mGetNRows(), vState->nb[1], 0 * cRowSize);
}

ggml_tensor * dgs_dsv4_comp_state::mGetScore(ggml_context * pCtx, int32_t pIl) const {
  ggml_tensor * vState   = mGetScoreAll(pCtx, pIl);
  const size_t  cRowSize = ggml_row_size(vState->type, vState->ne[0]);

  return ggml_view_2d(pCtx, vState, vState->ne[0], mGetNRows(), vState->nb[1], 0 * cRowSize);
}

ggml_tensor * dgs_dsv4_comp_state::mCpyKv(ggml_context * pCtx,
                                          ggml_tensor *  pCur,
                                          ggml_tensor *  pIdxs,
                                          int32_t        pIl) const {
  return ggml_set_rows(pCtx, mGetKvAll(pCtx, pIl), pCur, pIdxs);
}

ggml_tensor * dgs_dsv4_comp_state::mCpyScore(ggml_context * pCtx,
                                             ggml_tensor *  pCur,
                                             ggml_tensor *  pIdxs,
                                             int32_t        pIl) const {
  return ggml_set_rows(pCtx, mGetScoreAll(pCtx, pIl), pCur, pIdxs);
}

size_t dgs_dsv4_comp_state::mTotalSize() const {
  size_t size = 0;

  for (const auto & [_, buf] : lCtxsBufs) {
    size += ggml_backend_buffer_get_size(buf.get());
  }

  return size;
}

//
// dgs_kv_cache_dsv4
//

dgs_kv_cache_dsv4::dgs_kv_cache_dsv4(const dgs_model &       pModel,
                                     ggml_type               pTypeK,
                                     ggml_type               pTypeV,
                                     bool                    pVTrans,
                                     bool                    pOffload,
                                     bool                    pSwaFull,
                                     bool                    pUnified,
                                     uint32_t                pKvSize,
                                     uint32_t                pNSeqMax,
                                     uint32_t                pNUbatch,
                                     uint32_t                pNPad,
                                     uint32_t                pNRsSeq,
                                     const layer_filter_cb & pFilter,
                                     const layer_reuse_cb &  pReuse) :
  vHparamsRaw(pModel.hparams),
  vHparamsCsa(pModel.hparams),
  vHparamsHca(pModel.hparams),
  vHparamsLid(pModel.hparams),
  cNSeqMax(pNSeqMax),
  cNRsSeq(pNRsSeq),
  lRsIdx(pNSeqMax, 0) {
  const layer_filter_cb cFilterRaw = [&](int32_t pIl) {
    if (pFilter && !pFilter(pIl)) {
      return false;
    }

    return true;
  };

  GGML_UNUSED(pUnified);

  // Keep DSV4 KV/state streams per sequence even when public KV mode is unified.
  const bool cUnifiedRaw = false;

  vHparamsRaw.n_layer_nextn = 0;
  vHparamsCsa.n_layer_nextn = 0;
  vHparamsHca.n_layer_nextn = 0;
  vHparamsLid.n_layer_nextn = 0;

  DGS_LOG_INFO("%s: creating DSV4 raw KV cache\n", __func__);

  fDsv4MakeKOnly(vHparamsRaw);

  vKvRaw =
    std::make_unique<dgs_kv_cache_iswa>(pModel, vHparamsRaw, pTypeK, pTypeV, pVTrans, pOffload, pSwaFull, cUnifiedRaw,
                                        pKvSize, pNSeqMax, pNUbatch, pNPad, nullptr, cFilterRaw, pReuse, nullptr);

  fDsv4MakeKOnly(vHparamsCsa);
  fDsv4MakeKOnly(vHparamsHca);

  std::fill(vHparamsLid.aNHeadKvArr.begin(), vHparamsLid.aNHeadKvArr.end(), 1);
  vHparamsLid.vNEmbdHeadKFull = pModel.hparams.vIndexerHeadSize;
  vHparamsLid.vNEmbdHeadVFull = pModel.hparams.vIndexerHeadSize;
  vHparamsLid.vNEmbdHeadKSwa  = pModel.hparams.vIndexerHeadSize;
  vHparamsLid.vNEmbdHeadVSwa  = pModel.hparams.vIndexerHeadSize;
  vHparamsLid.vRopeType       = cDgsRopeTypeNeox;
  fDsv4MakeKOnly(vHparamsLid);

  const layer_filter_cb cFilterCsa = [&](int32_t pIl) {
    if (pFilter && !pFilter(pIl)) {
      return false;
    }

    return pModel.hparams.aDsv4CompressRatios[pIl] == cDsv4CsaRatio;
  };

  const layer_filter_cb cFilterHca = [&](int32_t pIl) {
    if (pFilter && !pFilter(pIl)) {
      return false;
    }

    return pModel.hparams.aDsv4CompressRatios[pIl] == cDsv4HcaRatio;
  };

  const bool cUnifiedCompressed = false;

  DGS_LOG_INFO("%s: creating DSV4 CSA compressed KV cache, size = %u cells\n", __func__,
               fDsv4CompSize(pKvSize, cDsv4CsaRatio));

  vKvCsa = std::make_unique<dgs_kv_cache>(pModel, vHparamsCsa, pTypeK, pTypeV, pVTrans, pOffload, cUnifiedCompressed,
                                          GGML_PAD(fDsv4CompSize(pKvSize, cDsv4CsaRatio), 256u), pNSeqMax, pNPad, 0,
                                          cDgsSwaTypeNone, nullptr, cFilterCsa, nullptr, nullptr);

  DGS_LOG_INFO("%s: creating DSV4 HCA compressed KV cache, size = %u cells\n", __func__,
               fDsv4CompSize(pKvSize, cDsv4HcaRatio));

  vKvHca = std::make_unique<dgs_kv_cache>(pModel, vHparamsHca, pTypeK, pTypeV, pVTrans, pOffload, cUnifiedCompressed,
                                          GGML_PAD(fDsv4CompSize(pKvSize, cDsv4HcaRatio), 256u), pNSeqMax, pNPad, 0,
                                          cDgsSwaTypeNone, nullptr, cFilterHca, nullptr, nullptr);

  DGS_LOG_INFO("%s: creating DSV4 lightning-indexer KV cache, size = %u cells\n", __func__,
               fDsv4CompSize(pKvSize, cDsv4CsaRatio));

  vKvLid = std::make_unique<dgs_kv_cache>(pModel, vHparamsLid, pTypeK, pTypeV, pVTrans, pOffload, cUnifiedCompressed,
                                          GGML_PAD(fDsv4CompSize(pKvSize, cDsv4CsaRatio), 256u), pNSeqMax, pNPad, 0,
                                          cDgsSwaTypeNone, nullptr, cFilterCsa, nullptr, nullptr);

  DGS_LOG_INFO("%s: creating DSV4 CSA compressor state\n", __func__);

  vCsaState = std::make_unique<dgs_dsv4_comp_state>(pModel, pOffload, cUnifiedCompressed, pNSeqMax, cDsv4CsaRatio,
                                                    2 * cDsv4CsaRatio, 2 * pModel.hparams.n_embd_head_k(), pNRsSeq,
                                                    "csa", cFilterCsa);

  DGS_LOG_INFO("%s: creating DSV4 HCA compressor state\n", __func__);

  vHcaState =
    std::make_unique<dgs_dsv4_comp_state>(pModel, pOffload, cUnifiedCompressed, pNSeqMax, cDsv4HcaRatio, cDsv4HcaRatio,
                                          pModel.hparams.n_embd_head_k(), pNRsSeq, "hca", cFilterHca);

  DGS_LOG_INFO("%s: creating DSV4 lightning-indexer compressor state\n", __func__);

  vLidState = std::make_unique<dgs_dsv4_comp_state>(pModel, pOffload, cUnifiedCompressed, pNSeqMax, cDsv4CsaRatio,
                                                    2 * cDsv4CsaRatio, 2 * pModel.hparams.vIndexerHeadSize, pNRsSeq,
                                                    "lid", cFilterCsa);

  // DSV4 attention reads compressed-K / compressor-state rows that the current
  // graph does not necessarily overwrite; uninitialized buffer contents would
  // otherwise leak in (instance-specific garbage) and corrupt recall. Zero all
  // compressed buffers up front so reads of un-written rows are deterministic.
  mClearCompressed(-1, true);
}

dgs_memory_context_ptr dgs_kv_cache_dsv4::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

  const bool cRawPerSeq  = vKvRaw->mGetBase()->mGetNStream() != 1;
  const bool cCompPerSeq = vCsaState->mGetNStream() > 1;
  const bool cHasCoupled = fDsv4BatchHasCoupled(pBalloc.mGetBatch());

  const auto cMakeContext = [&](std::vector<dgs_ubatch> pUbatches) -> dgs_memory_context_ptr {
    auto lUbatchesRaw = fDsv4BuildRawWriteUbatches(pUbatches);

    auto lSinfosRawBaseWrite = vKvRaw->mGetBase()->mPrepare(lUbatchesRaw);
    if (lSinfosRawBaseWrite.empty()) {
      return nullptr;
    }

    auto lSinfosRawSwaWrite = vKvRaw->mGetSwa()->mPrepare(lUbatchesRaw);
    if (lSinfosRawSwaWrite.empty()) {
      return nullptr;
    }

    auto lSinfosRawSwaRead = fDsv4BuildRawReadSinfos(lSinfosRawSwaWrite, pUbatches);

    return std::make_unique<dgs_kv_cache_dsv4_context>(this, std::move(lSinfosRawBaseWrite),
                                                       std::move(lSinfosRawSwaWrite), std::move(lSinfosRawSwaRead),
                                                       std::move(pUbatches), std::move(lUbatchesRaw));
  };

  // Match dgs_kv_cache_iswa splitting when DSV4 compressed state does not
  // require per-sequence graph layout.
  do {
    if (cRawPerSeq || cCompPerSeq) {
      break;
    }

    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = pBalloc.mSplitSimple(pNUbatch);
      if (vUbatch.vNTokens == 0) {
        break;
      }
      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      break;
    }

    if (auto vCtx = cMakeContext(std::move(lUbatches))) {
      return vCtx;
    }
  } while (false);

  // When raw or compressed state is per-sequence, independent sequences can
  // share an equal-length ubatch. Coupled sequence sets still serialize until
  // DSV4 has explicit shared-state handling for compressed streams.
  do {
    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      dgs_ubatch vUbatch;
      if (cHasCoupled) {
        vUbatch = pBalloc.mSplitSeq(pNUbatch);
      } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
        vUbatch = pBalloc.mSplitEqual(pNUbatch, cRawPerSeq || cCompPerSeq, cNRsSeq > 0 ? cNRsSeq + 1 : 0);
      }

      if (vUbatch.vNTokens == 0) {
        break;
      }
      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      break;
    }

    if (auto vCtx = cMakeContext(std::move(lUbatches))) {
      return vCtx;
    }
  } while (false);

  return std::make_unique<dgs_kv_cache_dsv4_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_kv_cache_dsv4::mInitFull() {
  return std::make_unique<dgs_kv_cache_dsv4_context>(this);
}

dgs_memory_context_ptr dgs_kv_cache_dsv4::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_kv_cache_dsv4_context>(this, pLctx, pOptimize, std::move(vCsaState->vScInfo),
                                                     std::move(vHcaState->vScInfo), std::move(vLidState->vScInfo));
}

bool dgs_kv_cache_dsv4::mGetCanShift() const {
  // Compressed row metadata uses block-derived positions. Keep shifting
  // disabled until DSV4 compressed-cache shift semantics are wired.
  return false;
}

void dgs_kv_cache_dsv4::clear(bool data) {
  vKvRaw->clear(data);
  mClearCompressed(-1, true);  // DSV4 compressed buffers must never expose stale/uninit rows
}

bool dgs_kv_cache_dsv4::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  if (pP1 >= 0) {
    return false;
  }

  if (pP0 > 0) {
    if (pSeqId < 0 || (uint32_t) pSeqId >= cNSeqMax) {
      return false;
    }

    const dgs_pos cPosMax = vKvRaw->mSeqPosMax(pSeqId);
    if (pP0 > cPosMax) {
      bool vRes = true;

      vRes = vRes & vKvRaw->mSeqRm(pSeqId, pP0, -1);
      vRes = vRes & vKvCsa->mSeqRm(pSeqId, pP0 / cDsv4CsaRatio, -1);
      vRes = vRes & vKvHca->mSeqRm(pSeqId, pP0 / cDsv4HcaRatio, -1);
      vRes = vRes & vKvLid->mSeqRm(pSeqId, pP0 / cDsv4CsaRatio, -1);

      return vRes;
    }

    if (cNRsSeq == 0) {
      return false;
    }

    const dgs_pos cRollback = cPosMax - (pP0 - 1);
    if (cRollback < 1 || cRollback > (dgs_pos) cNRsSeq) {
      return false;
    }

    // pending rollback is single-use: stacked partial removals don't compose
    if (lRsIdx[pSeqId] != 0) {
      return false;
    }

    const bool cRes = vKvRaw->mSeqRm(pSeqId, pP0, pP1);
    if (cRes) {
      lRsIdx[pSeqId] = (uint32_t) cRollback;
    }

    return cRes;
  }

  const bool cRes = vKvRaw->mSeqRm(pSeqId, pP0, pP1);

  if (cRes) {
    mClearCompressed(pSeqId, true);
  }

  return cRes;
}

void dgs_kv_cache_dsv4::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  GGML_ASSERT(pP0 <= 0 && pP1 < 0 && "DSV4 only supports full sequence copies");

  vKvRaw->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
  vKvCsa->mSeqCp(pSeqIdSrc, pSeqIdDst, -1, -1);
  vKvHca->mSeqCp(pSeqIdSrc, pSeqIdDst, -1, -1);
  vKvLid->mSeqCp(pSeqIdSrc, pSeqIdDst, -1, -1);

  vCsaState->mSeqCp(pSeqIdSrc, pSeqIdDst);
  vHcaState->mSeqCp(pSeqIdSrc, pSeqIdDst);
  vLidState->mSeqCp(pSeqIdSrc, pSeqIdDst);

  if (pSeqIdSrc != pSeqIdDst) {
    lRsIdx[pSeqIdDst] = 0;
  }
}

void dgs_kv_cache_dsv4::mSeqKeep(dgs_seq_id pSeqId) {
  GGML_ASSERT(pSeqId >= 0 && (uint32_t) pSeqId < cNSeqMax);

  vKvRaw->mSeqKeep(pSeqId);

  for (dgs_seq_id vId = 0; vId < (dgs_seq_id) cNSeqMax; ++vId) {
    if (vId == pSeqId) {
      continue;
    }

    vKvRaw->mSeqRm(vId, -1, -1);
    mClearCompressed(vId, true);
  }
}

void dgs_kv_cache_dsv4::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  vKvRaw->mSeqAdd(pSeqId, pP0, pP1, pShift);
}

void dgs_kv_cache_dsv4::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  vKvRaw->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos dgs_kv_cache_dsv4::mSeqPosMin(dgs_seq_id pSeqId) const {
  if (pSeqId < 0 || (uint32_t) pSeqId >= cNSeqMax) {
    return -1;
  }

  // The raw SWA cache may contain a wider window, but the compressed DSV4
  // state cannot be rolled back within that window. Report only the current
  // boundary so server-context uses checkpoints for rollback.
  return vKvRaw->mSeqPosMax(pSeqId);
}

dgs_pos dgs_kv_cache_dsv4::mSeqPosMax(dgs_seq_id pSeqId) const {
  if (pSeqId < 0 || (uint32_t) pSeqId >= cNSeqMax) {
    return -1;
  }

  return vKvRaw->mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_kv_cache_dsv4::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvRaw->mMemoryBreakdown();
  for (const auto & vBuftSize : vKvCsa->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  for (const auto & vBuftSize : vKvHca->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  for (const auto & vBuftSize : vKvLid->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  for (const auto & vBuftSize : vCsaState->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  for (const auto & vBuftSize : vHcaState->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  for (const auto & vBuftSize : vLidState->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

void dgs_kv_cache_dsv4::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  const bool cPartialOnly = pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY;

  const uint32_t cMagic   = cDsv4StateMagic;
  const uint32_t cVersion = cDsv4StateVersion;
  const uint32_t cMode    = cPartialOnly ? cDsv4StateModePartial : cDsv4StateModeFull;

  pIo.write(&cMagic, sizeof(cMagic));
  pIo.write(&cVersion, sizeof(cVersion));
  pIo.write(&cMode, sizeof(cMode));

  vKvRaw->mStateWrite(pIo, pSeqId, pFlags);

  if (!cPartialOnly) {
    const dgs_pos cPosMax = pSeqId >= 0 ? vKvRaw->mSeqPosMax(pSeqId) : -1;

    //FIXME : note that we conflate token positions with rows, which is not true for multi-modal case.
    const uint32_t cNRowsCsa =
      pSeqId >= 0 ? fDsv4StateNUsedKRows(cPosMax, cDsv4CsaRatio, vKvCsa->mGetSize()) : vKvCsa->mGetSize();
    const uint32_t cNRowsHca =
      pSeqId >= 0 ? fDsv4StateNUsedKRows(cPosMax, cDsv4HcaRatio, vKvHca->mGetSize()) : vKvHca->mGetSize();
    const uint32_t cNRowsLid =
      pSeqId >= 0 ? fDsv4StateNUsedKRows(cPosMax, cDsv4CsaRatio, vKvLid->mGetSize()) : vKvLid->mGetSize();

    fDsv4StateWriteKCache(pIo, vKvCsa.get(), pSeqId, pFlags, cNRowsCsa);
    fDsv4StateWriteKCache(pIo, vKvHca.get(), pSeqId, pFlags, cNRowsHca);
    fDsv4StateWriteKCache(pIo, vKvLid.get(), pSeqId, pFlags, cNRowsLid);
  }

  vCsaState->mStateWrite(pIo, pSeqId, pFlags, lRsIdx);
  vHcaState->mStateWrite(pIo, pSeqId, pFlags, lRsIdx);
  vLidState->mStateWrite(pIo, pSeqId, pFlags, lRsIdx);
}

void dgs_kv_cache_dsv4::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  uint32_t vMagic;
  uint32_t vVersion;
  uint32_t vMode = cDsv4StateModeFull;

  pIo.read(&vMagic, sizeof(vMagic));
  pIo.read(&vVersion, sizeof(vVersion));

  if (vMagic != cDsv4StateMagic) {
    throw std::runtime_error("DSV4 state magic mismatch");
  }
  if (vVersion != cDsv4StateVersion) {
    throw std::runtime_error("DSV4 state version mismatch");
  }

  pIo.read(&vMode, sizeof(vMode));
  if (vMode != cDsv4StateModeFull && vMode != cDsv4StateModePartial) {
    throw std::runtime_error("DSV4 state mode mismatch");
  }

  const bool cPartialOnly = vMode == cDsv4StateModePartial;
  if (cPartialOnly != !!(pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
    throw std::runtime_error("DSV4 state flags mismatch");
  }

  vKvRaw->mStateRead(pIo, pSeqId, pFlags);

  if (!cPartialOnly) {
    mClearCompressed(pSeqId, true);

    fDsv4StateReadKCache(pIo, vKvCsa.get(), pSeqId, pFlags);
    fDsv4StateReadKCache(pIo, vKvHca.get(), pSeqId, pFlags);
    fDsv4StateReadKCache(pIo, vKvLid.get(), pSeqId, pFlags);
  }

  vCsaState->mStateRead(pIo, pSeqId, pFlags);
  vHcaState->mStateRead(pIo, pSeqId, pFlags);
  vLidState->mStateRead(pIo, pSeqId, pFlags);

  if (pSeqId >= 0) {
    GGML_ASSERT((uint32_t) pSeqId < cNSeqMax);
    lRsIdx[pSeqId] = 0;
  } else {
    std::fill(lRsIdx.begin(), lRsIdx.end(), 0);
  }
}

dgs_kv_cache_iswa * dgs_kv_cache_dsv4::mGetRaw() const {
  return vKvRaw.get();
}

dgs_kv_cache * dgs_kv_cache_dsv4::mGetCsa() const {
  return vKvCsa.get();
}

dgs_kv_cache * dgs_kv_cache_dsv4::mGetHca() const {
  return vKvHca.get();
}

dgs_kv_cache * dgs_kv_cache_dsv4::mGetLid() const {
  return vKvLid.get();
}

dgs_dsv4_comp_state * dgs_kv_cache_dsv4::mGetCsaState() const {
  return vCsaState.get();
}

dgs_dsv4_comp_state * dgs_kv_cache_dsv4::mGetHcaState() const {
  return vHcaState.get();
}

dgs_dsv4_comp_state * dgs_kv_cache_dsv4::mGetLidState() const {
  return vLidState.get();
}

uint32_t dgs_kv_cache_dsv4::mGetNRsSeq() const {
  return cNRsSeq;
}

const std::vector<uint32_t> & dgs_kv_cache_dsv4::mGetRsIdx() const {
  return lRsIdx;
}

void dgs_kv_cache_dsv4::mResetRsIdxForUbatches(const std::vector<dgs_ubatch> & pUbatches) {
  if (cNRsSeq == 0) {
    return;
  }

  for (const dgs_ubatch & vUbatch : pUbatches) {
    for (uint32_t vI = 0; vI < vUbatch.vNTokens; ++vI) {
      for (int32_t vS = 0; vS < vUbatch.vNSeqId[vI]; ++vS) {
        const dgs_seq_id cSeqId = vUbatch.vSeqId[vI][vS];
        if (cSeqId >= 0 && (uint32_t) cSeqId < cNSeqMax) {
          lRsIdx[cSeqId] = 0;
        }
      }
    }
  }
}

void dgs_kv_cache_dsv4::mClearCompressed(dgs_seq_id pSeqId, bool data) {
  if (pSeqId < 0) {
    vKvCsa->clear(data);
    vKvHca->clear(data);
    vKvLid->clear(data);
  } else {
    GGML_ASSERT((uint32_t) pSeqId < cNSeqMax);

    const auto cClearSeq = [pSeqId, data](dgs_kv_cache * pKv) {
      pKv->mSeqRm(pSeqId, -1, -1);

      if (data) {
        //TODO: do not clear the kv-cache during `seq_rm`, ref: 
        for (uint32_t vIl : pKv->mGetLayerIds()) {
          fDsv4ClearTensorStream(pKv->mGetKStorage(vIl), (uint32_t) pSeqId);
        }
      }
    };

    cClearSeq(vKvCsa.get());
    cClearSeq(vKvHca.get());
    cClearSeq(vKvLid.get());
  }

  vCsaState->clear(pSeqId, data);
  vHcaState->clear(pSeqId, data);
  vLidState->clear(pSeqId, data);

  if (pSeqId >= 0) {
    lRsIdx[pSeqId] = 0;
  } else {
    std::fill(lRsIdx.begin(), lRsIdx.end(), 0);
  }
}

//
// dgs_kv_cache_dsv4_raw_context
//

static dgs_kv_cache::slot_info fDsv4BuildFullSinfo(const dgs_kv_cache * pKv) {
  const uint32_t cNStream = pKv->mGetNStream();

  dgs_kv_cache::slot_info vSinfo;
  vSinfo.vS0 = 0;
  vSinfo.vS1 = cNStream - 1;
  vSinfo.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vSinfo.lStrm[vS] = vS;
    vSinfo.lIdxs[vS].resize(1, 0);
  }

  return vSinfo;
}

dgs_kv_cache_dsv4_raw_context::dgs_kv_cache_dsv4_raw_context(dgs_kv_cache_iswa * pKv) :
  vKvSwa(pKv->mGetSwa()),
  cCtxBaseMem(nullptr),
  cCtxSwaMem(nullptr),
  vNKv(vKvSwa->mGetSize()),
  cStatus(cDgsMemoryStatusSuccess) {
  lSinfosRead.push_back(fDsv4BuildFullSinfo(vKvSwa));
  lSinfosWrite = lSinfosRead;
}

dgs_kv_cache_dsv4_raw_context::dgs_kv_cache_dsv4_raw_context(dgs_kv_cache_iswa * pKv,
                                                             dgs_context *       pLctx,
                                                             bool                pOptimize) :
  vKvSwa(pKv->mGetSwa()),
  cCtxBaseMem(pKv->mGetBase()->mInitUpdate(pLctx, pOptimize)),
  cCtxSwaMem(pKv->mGetSwa()->mInitUpdate(pLctx, pOptimize)),
  vNKv(vKvSwa->mGetSize()),
  cStatus(fDgsMemoryStatusCombine(cCtxBaseMem->mGetStatus(), cCtxSwaMem->mGetStatus())) {}

dgs_kv_cache_dsv4_raw_context::dgs_kv_cache_dsv4_raw_context(dgs_kv_cache_iswa *     pKv,
                                                             slot_info_vec_t         pSinfosBaseWrite,
                                                             slot_info_vec_t         pSinfosSwaWrite,
                                                             slot_info_vec_t         pSinfosSwaRead,
                                                             std::vector<dgs_ubatch> pUbatches,
                                                             std::vector<dgs_ubatch> pUbatchesWrite) :
  vKvSwa(pKv->mGetSwa()),
  lSinfosWrite(std::move(pSinfosSwaWrite)),
  lSinfosRead(std::move(pSinfosSwaRead)),
  lUbatches(std::move(pUbatches)),
  lUbatchesWrite(std::move(pUbatchesWrite)),
  cCtxBaseMem(
    std::make_unique<dgs_kv_cache_context>(pKv->mGetBase(), std::move(pSinfosBaseWrite), this->lUbatchesWrite)),
  cCtxSwaMem(nullptr),
  vNKv(vKvSwa->mGetSize()),
  cStatus(cDgsMemoryStatusSuccess) {}

bool dgs_kv_cache_dsv4_raw_context::mNext() {
  if (lUbatches.empty()) {
    return true;
  }

  if (cCtxBaseMem) {
    cCtxBaseMem->mNext();
  }

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_kv_cache_dsv4_raw_context::apply() {
  bool vRes = true;

  if (cCtxBaseMem) {
    vRes = vRes & cCtxBaseMem->apply();
  }
  if (cCtxSwaMem) {
    vRes = vRes & cCtxSwaMem->apply();
  }
  if (!lUbatchesWrite.empty()) {
    vKvSwa->mApplyUbatch(lSinfosWrite[vINext], lUbatchesWrite[vINext]);
    vNKv = vKvSwa->mGetNKv(lSinfosRead[vINext]);
  }

  return vRes;
}

dgs_memory_status dgs_kv_cache_dsv4_raw_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_kv_cache_dsv4_raw_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

uint32_t dgs_kv_cache_dsv4_raw_context::mGetNKv() const {
  return vNKv;
}

uint32_t dgs_kv_cache_dsv4_raw_context::mGetNWrite() const {
  if (lUbatchesWrite.empty()) {
    return 0;
  }

  return lUbatchesWrite[vINext].vNTokens;
}

ggml_tensor * dgs_kv_cache_dsv4_raw_context::mGetK(ggml_context * pCtx, int32_t pIl) const {
  return vKvSwa->mGetK(pCtx, pIl, vNKv, lSinfosRead[vINext]);
}

ggml_tensor * dgs_kv_cache_dsv4_raw_context::mCpyK(ggml_context * pCtx,
                                                   ggml_tensor *  pKCur,
                                                   ggml_tensor *  pKIdxs,
                                                   int32_t        pIl) const {
  const auto & vSinfo = lSinfosWrite[vINext];

  if (pKCur->ne[2] == pKIdxs->ne[0]) {
    return vKvSwa->mCpyK(pCtx, pKCur, pKIdxs, pIl, vSinfo);
  }

  // k_idxs may be expanded to one block per stream while k_cur is only
  // the token block. Keep zero deps on all copies so each write executes.
  const int64_t cNFanout = (int64_t) vSinfo.size() * vSinfo.mNStream();

  GGML_ASSERT(vSinfo.mNStream() > 1);
  GGML_ASSERT(pKCur->ne[2] == (int64_t) vSinfo.size());
  GGML_ASSERT(pKIdxs->ne[0] == cNFanout);

  ggml_tensor * vRes = nullptr;
  for (uint32_t vS = 0; vS < vSinfo.mNStream(); ++vS) {
    ggml_tensor * vKIdxsS = ggml_view_1d(pCtx, pKIdxs, vSinfo.size(), vS * vSinfo.size() * ggml_element_size(pKIdxs));
    ggml_tensor * vCur    = vKvSwa->mCpyK(pCtx, pKCur, vKIdxsS, pIl, vSinfo);
    if (vRes == nullptr) {
      vRes = vCur;
    } else {
      vRes = ggml_add(pCtx, vRes, ggml_sub(pCtx, vCur, vCur));
    }
  }

  return vRes;
}

ggml_tensor * dgs_kv_cache_dsv4_raw_context::mBuildInputKIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const {
  const uint32_t cNTokens = lUbatchesWrite.empty() ? pUbatch.vNTokens : lUbatchesWrite[vINext].vNTokens;

  ggml_tensor * vKIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens);
  ggml_set_input(vKIdxs);

  return vKIdxs;
}

ggml_tensor * dgs_kv_cache_dsv4_raw_context::mBuildInputKRot(ggml_context * pCtx) const {
  return vKvSwa->mBuildInputKRot(pCtx);
}

void dgs_kv_cache_dsv4_raw_context::mSetInputKIdxs(ggml_tensor * pDst) const {
  vKvSwa->mSetInputKIdxs(pDst, &lUbatchesWrite[vINext], lSinfosWrite[vINext]);
}

void dgs_kv_cache_dsv4_raw_context::mSetInputKqMask(ggml_tensor *      pDst,
                                                    const dgs_ubatch * pUbatch,
                                                    bool               pCausalAttn) const {
  vKvSwa->mSetInputKqMask(pDst, pUbatch, pCausalAttn);
}

void dgs_kv_cache_dsv4_raw_context::mSetInputKRot(ggml_tensor * pDst) const {
  vKvSwa->mSetInputKRot(pDst);
}

//
// dgs_kv_cache_dsv4_comp_context
//

dgs_kv_cache_dsv4_comp_context::dgs_kv_cache_dsv4_comp_context(dgs_kv_cache * pKv) : vKv(pKv), vNKv(pKv->mGetSize()) {
  const uint32_t cNStream = pKv->mGetNStream();

  lSinfos.resize(1);
  lSinfos[0].vS0 = 0;
  lSinfos[0].vS1 = cNStream - 1;
  lSinfos[0].lIdxs.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    lSinfos[0].lStrm.push_back(vS);
    lSinfos[0].lIdxs[vS].resize(1, 0);
  }
}

dgs_kv_cache_dsv4_comp_context::dgs_kv_cache_dsv4_comp_context(dgs_kv_cache *          pKv,
                                                               slot_info_vec_t         pSinfos,
                                                               std::vector<dgs_ubatch> pUbatches) :
  vKv(pKv),
  lSinfos(std::move(pSinfos)),
  lUbatches(std::move(pUbatches)),
  vNKv(pKv->mGetSize()) {}

bool dgs_kv_cache_dsv4_comp_context::mNext() {
  if (lUbatches.empty()) {
    return true;
  }

  if (++vICur >= lUbatches.size()) {
    return false;
  }

  return true;
}

uint32_t dgs_kv_cache_dsv4_comp_context::mGetNKv() const {
  return vNKv;
}

ggml_tensor * dgs_kv_cache_dsv4_comp_context::mGetK(ggml_context * pCtx, int32_t pIl) const {
  return vKv->mGetK(pCtx, pIl, vNKv, lSinfos[vICur]);
}

ggml_tensor * dgs_kv_cache_dsv4_comp_context::mCpyK(ggml_context * pCtx,
                                                    ggml_tensor *  pKCur,
                                                    ggml_tensor *  pKIdxs,
                                                    int32_t        pIl) const {
  return vKv->mCpyK(pCtx, pKCur, pKIdxs, pIl, lSinfos[vICur]);
}

ggml_tensor * dgs_kv_cache_dsv4_comp_context::mBuildInputKRot(ggml_context * pCtx) const {
  return vKv->mBuildInputKRot(pCtx);
}

void dgs_kv_cache_dsv4_comp_context::mSetInputKRot(ggml_tensor * pDst) const {
  vKv->mSetInputKRot(pDst);
}

//
// dgs_kv_cache_dsv4_context
//

dgs_kv_cache_dsv4_context::dgs_kv_cache_dsv4_context(dgs_memory_status pStatus) : cStatus(pStatus) {}

dgs_kv_cache_dsv4_context::dgs_kv_cache_dsv4_context(dgs_kv_cache_dsv4 * pKv) :
  cCtxRaw(std::make_unique<dgs_kv_cache_dsv4_raw_context>(pKv->mGetRaw())),
  cCtxCsaMem(pKv->mGetCsa()->mInitFull()),
  cCtxHcaMem(pKv->mGetHca()->mInitFull()),
  cCtxLidMem(pKv->mGetLid()->mInitFull()),
  cCtxCsa(std::make_unique<dgs_kv_cache_dsv4_comp_context>(pKv->mGetCsa())),
  cCtxHca(std::make_unique<dgs_kv_cache_dsv4_comp_context>(pKv->mGetHca())),
  cCtxLid(std::make_unique<dgs_kv_cache_dsv4_comp_context>(pKv->mGetLid())),
  vCsaState(pKv->mGetCsaState()),
  vHcaState(pKv->mGetHcaState()),
  vLidState(pKv->mGetLidState()),
  vReservePlans(true),
  cStatus(fDgsMemoryStatusCombine(fDgsMemoryStatusCombine(cCtxRaw->mGetStatus(), cCtxCsaMem->mGetStatus()),
                                  fDgsMemoryStatusCombine(cCtxHcaMem->mGetStatus(), cCtxLidMem->mGetStatus()))) {}

dgs_kv_cache_dsv4_context::dgs_kv_cache_dsv4_context(dgs_kv_cache_dsv4 * pKv,
                                                     dgs_context *       pLctx,
                                                     bool                pOptimize,
                                                     stream_copy_info    pScInfoCsa,
                                                     stream_copy_info    pScInfoHca,
                                                     stream_copy_info    pScInfoLid) :
  cCtxRaw(std::make_unique<dgs_kv_cache_dsv4_raw_context>(pKv->mGetRaw(), pLctx, pOptimize)),
  cCtxCsaMem(pKv->mGetCsa()->mInitUpdate(pLctx, pOptimize)),
  cCtxHcaMem(pKv->mGetHca()->mInitUpdate(pLctx, pOptimize)),
  cCtxLidMem(pKv->mGetLid()->mInitUpdate(pLctx, pOptimize)),
  vCsaState(pKv->mGetCsaState()),
  vHcaState(pKv->mGetHcaState()),
  vLidState(pKv->mGetLidState()),
  vScInfoCsa(std::move(pScInfoCsa)),
  vScInfoHca(std::move(pScInfoHca)),
  vScInfoLid(std::move(pScInfoLid)),
  cStatus(fDgsMemoryStatusCombine(
    fDgsMemoryStatusCombine(fDgsMemoryStatusCombine(cCtxRaw->mGetStatus(), cCtxCsaMem->mGetStatus()),
                            fDgsMemoryStatusCombine(cCtxHcaMem->mGetStatus(), cCtxLidMem->mGetStatus())),
    this->vScInfoCsa.empty() && this->vScInfoHca.empty() && this->vScInfoLid.empty() ? cDgsMemoryStatusNoUpdate :
                                                                                       cDgsMemoryStatusSuccess)) {}

dgs_kv_cache_dsv4_context::dgs_kv_cache_dsv4_context(dgs_kv_cache_dsv4 *     pKv,
                                                     slot_info_vec_t         pSinfosRawBaseWrite,
                                                     slot_info_vec_t         pSinfosRawSwaWrite,
                                                     slot_info_vec_t         pSinfosRawSwaRead,
                                                     std::vector<dgs_ubatch> pUbatches,
                                                     std::vector<dgs_ubatch> pUbatchesRaw) :
  lUbatches(std::move(pUbatches)),
  lPlansCsa(fDsv4BuildCompPlans(this->lUbatches,
                                cDsv4CsaRatio,
                                true,
                                pKv->mGetCsaState()->mGetStateSize(),
                                pKv->mGetCsa()->mGetSize(),
                                pKv->mGetCsaState()->mGetNStream(),
                                pKv->mGetNRsSeq(),
                                pKv->mGetRsIdx())),
  lPlansHca(fDsv4BuildCompPlans(this->lUbatches,
                                cDsv4HcaRatio,
                                false,
                                pKv->mGetHcaState()->mGetStateSize(),
                                pKv->mGetHca()->mGetSize(),
                                pKv->mGetHcaState()->mGetNStream(),
                                pKv->mGetNRsSeq(),
                                pKv->mGetRsIdx())),
  lPlansLid(fDsv4BuildCompPlans(this->lUbatches,
                                cDsv4CsaRatio,
                                true,
                                pKv->mGetLidState()->mGetStateSize(),
                                pKv->mGetLid()->mGetSize(),
                                pKv->mGetLidState()->mGetNStream(),
                                pKv->mGetNRsSeq(),
                                pKv->mGetRsIdx())),
  cCtxRaw(std::make_unique<dgs_kv_cache_dsv4_raw_context>(pKv->mGetRaw(),
                                                          std::move(pSinfosRawBaseWrite),
                                                          std::move(pSinfosRawSwaWrite),
                                                          std::move(pSinfosRawSwaRead),
                                                          this->lUbatches,
                                                          std::move(pUbatchesRaw))),
  cCtxCsaMem(nullptr),
  cCtxHcaMem(nullptr),
  cCtxLidMem(nullptr),
  cCtxCsa(std::make_unique<dgs_kv_cache_dsv4_comp_context>(
    pKv->mGetCsa(),
    fDsv4BuildCompSinfos(this->lUbatches, pKv->mGetCsa()->mGetNStream()),
    this->lUbatches)),
  cCtxHca(std::make_unique<dgs_kv_cache_dsv4_comp_context>(
    pKv->mGetHca(),
    fDsv4BuildCompSinfos(this->lUbatches, pKv->mGetHca()->mGetNStream()),
    this->lUbatches)),
  cCtxLid(std::make_unique<dgs_kv_cache_dsv4_comp_context>(
    pKv->mGetLid(),
    fDsv4BuildCompSinfos(this->lUbatches, pKv->mGetLid()->mGetNStream()),
    this->lUbatches)),
  vCsaState(pKv->mGetCsaState()),
  vHcaState(pKv->mGetHcaState()),
  vLidState(pKv->mGetLidState()),
  cStatus(cCtxRaw->mGetStatus()) {
  pKv->mResetRsIdxForUbatches(this->lUbatches);
}

dgs_kv_cache_dsv4_context::~dgs_kv_cache_dsv4_context() = default;

bool dgs_kv_cache_dsv4_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  cCtxRaw->mNext();
  cCtxCsa->mNext();
  cCtxHca->mNext();
  cCtxLid->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_kv_cache_dsv4_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  bool vRes = true;

  vRes = vRes & cCtxRaw->apply();

  if (cCtxCsaMem) {
    vRes = vRes & cCtxCsaMem->apply();
    vRes = vRes & cCtxHcaMem->apply();
    vRes = vRes & cCtxLidMem->apply();
  }

  if (lUbatches.empty()) {
    vCsaState->mApplyCopies(vScInfoCsa);
    vHcaState->mApplyCopies(vScInfoHca);
    vLidState->mApplyCopies(vScInfoLid);
  }

  return vRes;
}

dgs_memory_status dgs_kv_cache_dsv4_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_kv_cache_dsv4_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const dgs_kv_cache_dsv4_raw_context * dgs_kv_cache_dsv4_context::mGetRaw() const {
  assert(status == cDgsMemoryStatusSuccess);

  return cCtxRaw.get();
}

const dgs_kv_cache_dsv4_comp_context * dgs_kv_cache_dsv4_context::mGetCsa() const {
  assert(status == cDgsMemoryStatusSuccess);

  return cCtxCsa.get();
}

const dgs_kv_cache_dsv4_comp_context * dgs_kv_cache_dsv4_context::mGetHca() const {
  assert(status == cDgsMemoryStatusSuccess);

  return cCtxHca.get();
}

const dgs_kv_cache_dsv4_comp_context * dgs_kv_cache_dsv4_context::mGetLid() const {
  assert(status == cDgsMemoryStatusSuccess);

  return cCtxLid.get();
}

const dgs_dsv4_comp_state * dgs_kv_cache_dsv4_context::mGetCsaState() const {
  assert(status == cDgsMemoryStatusSuccess);

  return vCsaState;
}

const dgs_dsv4_comp_state * dgs_kv_cache_dsv4_context::mGetHcaState() const {
  assert(status == cDgsMemoryStatusSuccess);

  return vHcaState;
}

const dgs_dsv4_comp_state * dgs_kv_cache_dsv4_context::mGetLidState() const {
  assert(status == cDgsMemoryStatusSuccess);

  return vLidState;
}

const dgs_kv_cache_dsv4_context::comp_plan & dgs_kv_cache_dsv4_context::mGetCsaPlan() const {
  assert(status == cDgsMemoryStatusSuccess);

  static const comp_plan empty;
  if (lPlansCsa.empty()) {
    return empty;
  }

  return lPlansCsa[vINext];
}

const dgs_kv_cache_dsv4_context::comp_plan & dgs_kv_cache_dsv4_context::mGetHcaPlan() const {
  assert(status == cDgsMemoryStatusSuccess);

  static const comp_plan empty;
  if (lPlansHca.empty()) {
    return empty;
  }

  return lPlansHca[vINext];
}

const dgs_kv_cache_dsv4_context::comp_plan & dgs_kv_cache_dsv4_context::mGetLidPlan() const {
  assert(status == cDgsMemoryStatusSuccess);

  static const comp_plan empty;
  if (lPlansLid.empty()) {
    return empty;
  }

  return lPlansLid[vINext];
}

const dgs_kv_cache_dsv4_context::comp_plan & dgs_kv_cache_dsv4_context::mGetCsaPlan(const dgs_ubatch & pUbatch) const {
  assert(status == cDgsMemoryStatusSuccess);

  if (!vReservePlans) {
    return mGetCsaPlan();
  }

  vReservePlanCsa = fDsv4BuildReserveCompPlan(pUbatch, cDsv4CsaRatio, true, vCsaState->mGetStateSize(),
                                              mGetCsa()->mGetNKv(), vCsaState->mGetNStream(), vCsaState->mGetNRsSeq());

  return vReservePlanCsa;
}

const dgs_kv_cache_dsv4_context::comp_plan & dgs_kv_cache_dsv4_context::mGetHcaPlan(const dgs_ubatch & pUbatch) const {
  assert(status == cDgsMemoryStatusSuccess);

  if (!vReservePlans) {
    return mGetHcaPlan();
  }

  vReservePlanHca = fDsv4BuildReserveCompPlan(pUbatch, cDsv4HcaRatio, false, vHcaState->mGetStateSize(),
                                              mGetHca()->mGetNKv(), vHcaState->mGetNStream(), vHcaState->mGetNRsSeq());

  return vReservePlanHca;
}

const dgs_kv_cache_dsv4_context::comp_plan & dgs_kv_cache_dsv4_context::mGetLidPlan(const dgs_ubatch & pUbatch) const {
  assert(status == cDgsMemoryStatusSuccess);

  if (!vReservePlans) {
    return mGetLidPlan();
  }

  vReservePlanLid = fDsv4BuildReserveCompPlan(pUbatch, cDsv4CsaRatio, true, vLidState->mGetStateSize(),
                                              mGetLid()->mGetNKv(), vLidState->mGetNStream(), vLidState->mGetNRsSeq());

  return vReservePlanLid;
}
