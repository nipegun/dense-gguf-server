#include "dgs-batch.h"

#include "dgs-context.h"
#include "dgs-hparams.h"
#include "dgs-impl.h"
#include "dgs-memory.h"
#include "dgs-model.h"
#include "dgs-vocab.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>

dgs_batch_allocr::dgs_batch_allocr(uint32_t pNPosPerEmbd, bool pAllowMixed) :
  cNPosPerEmbd(pNPosPerEmbd),
  cAllowMixed(pAllowMixed) {
  const char * vDgsBatchDebug = getenv("DGS_BATCH_DEBUG");
  vDebug                      = vDgsBatchDebug ? atoi(vDgsBatchDebug) : 0;

  sSeqPos.resize(DGS_MAX_SEQ);
  lSeqCpl.resize(DGS_MAX_SEQ);
  for (auto & lCur : lSeqCpl) {
    lCur.resize(DGS_MAX_SEQ);
  }

  lSeqIdx.resize(DGS_MAX_SEQ, -1);
}

bool dgs_batch_allocr::init(const dgs_batch_ext & pBatchInp, const dgs_vocab & pVocab, bool pOutputAll) {
  clear();

  this->vVocab      = &pVocab;
  this->vNEmbd      = pBatchInp.vNEmbd > 0 ? pBatchInp.vNEmbd : pBatchInp.cNEmbdInp;
  this->vNEmbdState = pBatchInp.cNEmbdState;
  this->vNSeqMax    = pBatchInp.cNSeqMax;

  const int32_t cNTok = (int32_t) pBatchInp.lTokens.size();

  GGML_ASSERT(cNTok > 0);

  if ((uint32_t) vNSeqMax > DGS_MAX_SEQ) {
    DGS_LOG_ERROR("%s: n_seq_max = %d > %d\n", __func__, vNSeqMax, DGS_MAX_SEQ);
    return false;
  }

  const dgs_memory_i * vMem = pBatchInp.vMem;

  //
  // determine the content types of the batch
  // an entry can carry a token id, a token embedding, or both
  // all entries must carry the same combination, or be a mix of token and embd entries
  // a state embedding (e.g. MTP hook batches) is set on all entries or on none
  //

  int32_t vNTokOnly  = 0;
  int32_t vNEmbdOnly = 0;
  int32_t vNBoth     = 0;

  const bool cHasState = pBatchInp.lTokens[0].vHasState;

  for (int32_t vI = 0; vI < cNTok; ++vI) {
    const bool cIsTok = pBatchInp.lTokens[vI].vId != DGS_TOKEN_NULL;
    const bool cIsEmb = pBatchInp.lTokens[vI].vHasEmbd;

    if (!cIsTok && !cIsEmb) {
      DGS_LOG_ERROR("%s: entry %d has neither a token id nor an embedding\n", __func__, vI);
      return false;
    }

    if (pBatchInp.lTokens[vI].vHasState != cHasState) {
      DGS_LOG_ERROR("%s: all entries in the batch must have the same state embedding presence\n", __func__);
      return false;
    }

    vNTokOnly += cIsTok && !cIsEmb;
    vNEmbdOnly += cIsEmb && !cIsTok;
    vNBoth += cIsTok && cIsEmb;
  }

  if (vNBoth > 0 && vNBoth != cNTok) {
    DGS_LOG_ERROR("%s: entries with both a token id and an embedding cannot be mixed with other entries\n", __func__);
    return false;
  }

  const bool cMixed = vNTokOnly > 0 && vNEmbdOnly > 0;

  if (cMixed && !cAllowMixed) {
    DGS_LOG_ERROR("%s: this model or context does not support batches mixing token and embedding entries\n", __func__);
    return false;
  }

  const bool cHasToken = vNTokOnly > 0 || vNBoth > 0;
  const bool cHasEmbd  = vNEmbdOnly > 0 || vNBoth > 0;

  if (cMixed) {
    lIsEmbdVec.resize(cNTok);
    for (int32_t vI = 0; vI < cNTok; ++vI) {
      lIsEmbdVec[vI] = pBatchInp.lTokens[vI].vHasEmbd ? 1 : 0;
    }
  }

  //
  // build flat token/embd array
  //

  if (cHasToken) {
    lTokenVec.resize(cNTok);
    for (int32_t vI = 0; vI < cNTok; ++vI) {
      if (cMixed && lIsEmbdVec[vI]) {
        lTokenVec[vI] = 0;  // placeholder
        continue;
      }
      const dgs_token cId = pBatchInp.lTokens[vI].vId;
      if (cId < 0 || cId >= pBatchInp.cNVocab) {
        DGS_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, vI, cId);
        return false;
      }
      lTokenVec[vI] = cId;
    }
  }

  if (cMixed) {
    lEmbdVec.assign((size_t) cNTok * vNEmbd, 0.0f);
    for (int32_t vI = 0; vI < cNTok; ++vI) {
      if (lIsEmbdVec[vI]) {
        const float * vSrc = pBatchInp.lEmbd.data() + pBatchInp.lTokens[vI].vEmbdOff;
        std::copy(vSrc, vSrc + vNEmbd, lEmbdVec.data() + (size_t) vI * vNEmbd);
      }
    }
  } else if (cHasEmbd) {
    lEmbdVec = pBatchInp.lEmbd;
  }

  if (cHasState) {
    lStateVec = pBatchInp.lState;
  }

  //
  // build flat pos array, section-major: pos[j*n_tok + i] = section j of entry i
  // token entry: [p, p, p, 0] (M-RoPE text position)
  // embd entry:  tokens[i].pos as-is
  //

  lPos.resize((size_t) cNTok * cNPosPerEmbd);
  for (int32_t vI = 0; vI < cNTok; ++vI) {
    const auto & vTok    = pBatchInp.lTokens[vI];
    const bool   cExpand = vTok.vId != DGS_TOKEN_NULL;
    for (uint32_t vJ = 0; vJ < cNPosPerEmbd; ++vJ) {
      dgs_pos vP = vTok.aPos[vJ];
      if (cExpand) {
        // expand [p] to [p, p, p, 0] for M-RoPE
        vP = vJ < 3 ? vTok.aPos[0] : 0;
      }
      lPos[(size_t) vJ * cNTok + vI] = vP;
    }
  }

  //
  // build n_seq_id / seq_id arrays
  //

  lNSeqId.resize(cNTok);
  lSeqId.resize(cNTok + 1);
  lSeqId[cNTok] = nullptr;

  {
    size_t vTotal = 0;
    for (int32_t vI = 0; vI < cNTok; ++vI) {
      vTotal += pBatchInp.lTokens[vI].sSeqIds.size();
    }
    lSeqIdData.reserve(vTotal);

    for (int32_t vI = 0; vI < cNTok; ++vI) {
      for (auto vSid : pBatchInp.lTokens[vI].sSeqIds) {
        lSeqIdData.push_back(vSid);
      }
    }

    size_t vOff = 0;
    for (int32_t vI = 0; vI < cNTok; ++vI) {
      lNSeqId[vI] = (int32_t) pBatchInp.lTokens[vI].sSeqIds.size();
      lSeqId[vI]  = lSeqIdData.data() + vOff;
      vOff += lNSeqId[vI];

      for (int32_t vS = 0; vS < lNSeqId[vI]; ++vS) {
        if (lSeqId[vI][vS] < 0 || lSeqId[vI][vS] >= (dgs_seq_id) vNSeqMax) {
          DGS_LOG_ERROR("%s: invalid seq_id[%d][%d] = %d >= %d\n", __func__, vI, vS, lSeqId[vI][vS],
                        (dgs_seq_id) vNSeqMax);
          return false;
        }
      }
    }
  }

  //
  // build output/logits array
  //

  {
    lOutput.resize(cNTok, 0);
    for (int32_t vI = 0; vI < cNTok; ++vI) {
      lOutput[vI] = pBatchInp.lTokens[vI].vOutput ? 1 : 0;
    }

    if (pOutputAll) {
      bool vWarn = false;
      for (int32_t vI = 0; vI < cNTok; ++vI) {
        if (!lOutput[vI]) {
          vWarn = true;
          break;
        }
      }
      if (vWarn) {
        DGS_LOG_WARN("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n",
                     __func__);
        std::fill(lOutput.begin(), lOutput.end(), 1);
      }
    }
  }

  // kept empty if no entry has one
  for (int32_t vI = 0; vI < cNTok; ++vI) {
    if (pBatchInp.lTokens[vI].vDecisionOrder != 0) {
      lDecisionOrder.resize(cNTok, 0);
      lDecisionOrder[vI] = pBatchInp.lTokens[vI].vDecisionOrder;
    }
  }

  //
  // set up the internal dgs_batch to point to our owned arrays
  //

  vBatch.vNTokens = cNTok;
  vBatch.token    = cHasToken ? lTokenVec.data() : nullptr;
  vBatch.vEmbd    = cHasEmbd ? lEmbdVec.data() : nullptr;
  vBatch.vPos     = lPos.data();
  vBatch.vNSeqId  = lNSeqId.data();
  vBatch.vSeqId   = lSeqId.data();
  vBatch.vLogits  = lOutput.data();

  //
  // compute stats
  //

  // count the outputs in this batch
  for (int32_t vI = 0; vI < vBatch.vNTokens; ++vI) {
    vNOutputs += vBatch.vLogits[vI] != 0;
  }

  vHasCpl = false;

  // determine coupled sequences
  // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
  for (int32_t vI = 0; vI < vBatch.vNTokens; ++vI) {
    const dgs_seq_id cS0 = vBatch.vSeqId[vI][0];

    for (int32_t vS = 0; vS < vBatch.vNSeqId[vI]; ++vS) {
      const dgs_seq_id cS1 = vBatch.vSeqId[vI][vS];

      sSeqPos[cS1].insert(vBatch.vPos[vI]);

      if (vS > 0) {
        // mark that sequence s1 is coupled to s0
        lSeqCpl[cS1][cS0] = true;

        // note: tracking the other way around is not necessary for now
        //seq_cpl[s0][s1] = true;

        vHasCpl = true;
      }
    }
  }

  // precompute the sequence sets for each token and determine the unique sequence ids that participate in the batch
  {
    seq_set_t vSeqSetUnq;

    for (int32_t vI = 0; vI < vBatch.vNTokens; ++vI) {
      seq_set_t vCur;
      for (int32_t vS = 0; vS < vBatch.vNSeqId[vI]; ++vS) {
        const dgs_seq_id cSeqId = vBatch.vSeqId[vI][vS];

        vCur.set(cSeqId);
        vSeqSetUnq.set(cSeqId);
      }

      lSeqSet.push_back(vCur);
      dSeqSetMap[vCur].push_back(vI);
    }

    for (uint32_t vS = 0; vS < vNSeqMax; ++vS) {
      if (vSeqSetUnq.test(vS)) {
        lSeqIdx[vS] = lSeqIdUnq.size();
        lSeqIdUnq.push_back(vS);
      }
    }
  }

  if (vDebug > 0) {
    DGS_LOG_DEBUG("%s: input batch info:\n", __func__);

    dgs_ubatch vUbatch{
      /*.b_equal_seqs =*/false,
      /*.n_tokens     =*/(uint32_t) vBatch.vNTokens,
      /*.n_seq_tokens =*/(uint32_t) 1,
      /*.n_seqs       =*/(uint32_t) vBatch.vNTokens,
      /*.n_seqs_unq   =*/(uint32_t) this->lSeqIdUnq.size(),
      /*.n_pos        =*/cNPosPerEmbd,
      /*.token        =*/vBatch.token,
      /*.embd         =*/vBatch.vEmbd,
      /*.embd_state   =*/lStateVec.empty() ? nullptr : lStateVec.data(),
      /*.pos          =*/vBatch.vPos,
      /*.n_seq_id     =*/vBatch.vNSeqId,
      /*.seq_id       =*/vBatch.vSeqId,
      /*.seq_id_unq   =*/this->lSeqIdUnq.data(),
      /*.seq_idx      =*/this->lSeqIdx.data(),
      /*.output       =*/vBatch.vLogits,
      /*.type         =*/lIsEmbdVec.empty() ? nullptr : lIsEmbdVec.data(),
      /*.decision_order =*/lDecisionOrder.empty() ? nullptr : lDecisionOrder.data(),
      /*.data         =*/{},
    };

    mUbatchPrint(vUbatch, vDebug);

    DGS_LOG_DEBUG("%s:   seq       = [\n", __func__);
    for (int vS0 = 0; vS0 < (int) sSeqPos.size(); ++vS0) {
      if (sSeqPos[vS0].empty()) {
        continue;
      }

      std::stringstream vSs;
      for (int vS1 = 0; vS1 < (int) lSeqCpl[vS0].size(); ++vS1) {
        if (lSeqCpl[vS0][vS1]) {
          vSs << vS1 << " ";
        }
      }

      DGS_LOG_DEBUG("%s:  %4d: pos = [%4d, %4d], cpl = %s\n", __func__, vS0, mSeqPosMin(vS0), mSeqPosMax(vS0),
                    vSs.str().empty() ? "-" : vSs.str().c_str());
    }
    DGS_LOG_DEBUG("%s:   ]\n", __func__);
  }

  //
  // consistency checks
  //

  if (cNPosPerEmbd > 1) {
    // in a mixed batch, the first entry of each seq picks the rule
    std::vector<int8_t> lSeqFirstEmbd(vNSeqMax, vBatch.token ? 0 : 1);
    if (cMixed) {
      std::vector<bool> lSeen(vNSeqMax, false);
      for (int32_t vI = 0; vI < vBatch.vNTokens; ++vI) {
        for (int32_t vS = 0; vS < vBatch.vNSeqId[vI]; ++vS) {
          const dgs_seq_id cSid = vBatch.vSeqId[vI][vS];
          if (!lSeen[cSid]) {
            lSeen[cSid]         = true;
            lSeqFirstEmbd[cSid] = lIsEmbdVec[vI];
          }
        }
      }
    }

    // M-RoPE case: allow position to "jump" forward only (non-continuous positions are allowed)
    for (uint32_t vS = 0; vS < vNSeqMax; ++vS) {
      if (sSeqPos[vS].empty()) {
        continue;
      }

      const dgs_pos cP0 = vMem ? vMem->mSeqPosMax(vS) : -1;

      if (!lSeqFirstEmbd[vS]) {
        if (cP0 >= 0 && cP0 >= mSeqPosMin(vS)) {
          DGS_LOG_ERROR(
            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
            " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X "
            "= %d\n"
            " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
            " for M-RoPE, it is required that the position satisfies: X < Y\n",
            __func__, vS, vS, cP0, vS, mSeqPosMin(vS));

          return false;
        }
      } else {
        // embedding inputs can have overlapping positions
        if (cP0 >= 0 && cP0 > mSeqPosMin(vS)) {
          DGS_LOG_ERROR(
            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
            " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X "
            "= %d\n"
            " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
            " for M-RoPE, it is required that the position satisfies: X <= Y\n",
            __func__, vS, vS, cP0, vS, mSeqPosMin(vS));

          return false;
        }
      }
    }
  } else {
    for (uint32_t vS = 0; vS < vNSeqMax; ++vS) {
      if (sSeqPos[vS].empty()) {
        continue;
      }

      const dgs_pos cP0 = vMem ? vMem->mSeqPosMax(vS) : -1;

      if (cP0 >= 0) {
        bool vOk = true;

        if (mSeqPosMin(vS) != cP0 + 1) {
          vOk = false;
        }

        if (!vOk) {
          DGS_LOG_ERROR(
            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
            " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X "
            "= %d\n"
            " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
            " it is required that the sequence positions remain consecutive: Y = X + 1\n",
            __func__, vS, vS, cP0, vS, mSeqPosMin(vS));

          return false;
        }
      }

      if (mSeqPosMax(vS) - mSeqPosMin(vS) + 1 > (int) sSeqPos[vS].size()) {
        DGS_LOG_ERROR("%s: sequence %d positions are not continuous\n", __func__, vS);
        return false;
      }
    }
  }

  if (vMem) {
    for (uint32_t vS0 = 0; vS0 < vNSeqMax; ++vS0) {
      for (uint32_t vS1 = 0; vS1 < vNSeqMax; ++vS1) {
        if (lSeqCpl[vS0][vS1]) {
          if (vMem->mSeqPosMin(vS0) != vMem->mSeqPosMin(vS1) || vMem->mSeqPosMax(vS0) != vMem->mSeqPosMax(vS1)) {
            DGS_LOG_ERROR("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", __func__, vS0,
                          vS1);
            return false;
          }
        }
      }
    }
  }

  // disallow partial sequence sub-sets:
  //
  // invalid:          x
  //            i: 0 1 2 ...
  // ---------------------------------------
  // seq_id[i][0]: 0 0 1
  // seq_id[i][1]: 1 1 2
  // seq_id[i][2]: 2
  //
  // disallow decreasing sequence positions:
  //
  // invalid:                  x
  //            i: 0 1 2 3 4 5 6 ...
  // ---------------------------------------
  //       pos[i]: 4 5 0 1 6 2 3
  // seq_id[i][0]: 0 0 1 1 0 1 0
  //
  {
    seq_set_t aCurSeqSet[DGS_MAX_SEQ];
    for (uint32_t vS = 0; vS < vNSeqMax; ++vS) {
      aCurSeqSet[vS].set();
    }

    dgs_pos aCurSeqPos[DGS_MAX_SEQ];
    for (uint32_t vS = 0; vS < vNSeqMax; ++vS) {
      aCurSeqPos[vS] = -1;
    }

    for (int32_t vI = 0; vI < vBatch.vNTokens; ++vI) {
      const dgs_pos cPos = vBatch.vPos[vI];

      for (int32_t vS = 0; vS < vBatch.vNSeqId[vI]; ++vS) {
        const dgs_seq_id cSeqId = vBatch.vSeqId[vI][vS];

        aCurSeqSet[cSeqId] &= lSeqSet[vI];

        if (aCurSeqSet[cSeqId].none()) {
          DGS_LOG_ERROR("%s: sequence %d belongs to incompatible sequence sets (not allowed)\n", __func__, cSeqId);
          return false;
        }

        if (cPos < aCurSeqPos[cSeqId]) {
          DGS_LOG_ERROR("%s: sequence %d positions are decreasing (not allowed)\n", __func__, cSeqId);
          return false;
        }

        aCurSeqPos[cSeqId] = cPos;
      }
    }
  }

  mSplitReset();

  return true;
}

dgs_ubatch dgs_batch_allocr::mUbatchReserve(uint32_t pNSeqTokens, uint32_t pNSeqs) {
  const uint32_t cNTokens = pNSeqTokens * pNSeqs;

  clear();
  mSplitReset();

  const int64_t cNPosAll = (int64_t) cNTokens * cNPosPerEmbd;

  auto vUdata = std::make_shared<dgs_ubatch::data_t>();

  vUdata->token.resize(cNTokens);
  vUdata->lEmbd.clear();
  vUdata->lEmbdState.clear();
  vUdata->lPos.resize(cNPosAll);
  vUdata->lNSeqId.resize(cNTokens);
  vUdata->lSeqId.resize(cNTokens);
  vUdata->lSeqIdUnq.resize(0);
  vUdata->lSeqIdx.resize(DGS_MAX_SEQ, -1);
  vUdata->lOutput.resize(cNTokens);

  for (uint32_t vS = 0; vS < pNSeqs; ++vS) {
    vUdata->lSeqIdx[vS] = vS;
    vUdata->lSeqIdUnq.push_back(vS);
  }

  dgs_ubatch vRes{
    /*.b_equal_seqs =*/true,
    /*.n_tokens     =*/cNTokens,
    /*.n_seq_tokens =*/pNSeqTokens,
    /*.n_seqs       =*/pNSeqs,
    /*.n_seqs_unq   =*/pNSeqs,
    /*.n_pos        =*/cNPosPerEmbd,

    /*.token        =*/vUdata->token.data(),
    /*.embd         =*/nullptr,
    /*.embd_state   =*/nullptr,
    /*.pos          =*/vUdata->lPos.data(),
    /*.n_seq_id     =*/vUdata->lNSeqId.data(),
    /*.seq_id       =*/vUdata->lSeqId.data(),
    /*.seq_id_unq   =*/vUdata->lSeqIdUnq.data(),
    /*.seq_idx      =*/vUdata->lSeqIdx.data(),
    /*.output       =*/vUdata->lOutput.data(),
    /*.type         =*/nullptr,
    /*.decision_order =*/nullptr,
    /*.data         =*/std::move(vUdata),
  };

  return vRes;
}

const dgs_batch & dgs_batch_allocr::mGetBatch() const {
  return vBatch;
}

uint32_t dgs_batch_allocr::mGetNTokens() const {
  return vBatch.vNTokens;
}

uint32_t dgs_batch_allocr::mGetNOutputs() const {
  return vNOutputs;
}

uint32_t dgs_batch_allocr::mGetNUsed() const {
  return vNUsed;
}

std::vector<int32_t> & dgs_batch_allocr::mGetOutIds() {
  return lOutIds;
}

dgs_pos dgs_batch_allocr::mSeqPosMin(dgs_seq_id pSeqId) const {
  return sSeqPos[pSeqId].empty() ? -1 : *sSeqPos[pSeqId].begin();
}

dgs_pos dgs_batch_allocr::mSeqPosMax(dgs_seq_id pSeqId) const {
  return sSeqPos[pSeqId].empty() ? -1 : *sSeqPos[pSeqId].rbegin();
}

void dgs_batch_allocr::mSplitReset() {
  lOutIds.clear();

  vNUsed = 0;

  lUsed.clear();
  lUsed.resize(mGetNTokens(), false);
}

dgs_ubatch dgs_batch_allocr::mSplitSimple(uint32_t pNUbatch) {
  // find the first unused token
  uint32_t vCurIdx = 0;
  while (vCurIdx < lUsed.size() && lUsed[vCurIdx]) {
    ++vCurIdx;
  }

  // we are done
  if (vCurIdx >= lUsed.size()) {
    return {};
  }

  std::vector<int32_t> lIdxs;

  while (true) {
    lIdxs.push_back(vCurIdx);

    lUsed[vCurIdx] = true;
    ++vNUsed;

    ++vCurIdx;

    if (vCurIdx >= lUsed.size()) {
      break;
    }

    if (lIdxs.size() >= pNUbatch) {
      break;
    }
  }

  return mUbatchAdd(lIdxs, lIdxs.size(), false);
}

dgs_ubatch dgs_batch_allocr::mSplitEqual(uint32_t pNUbatch, bool pSequential, uint32_t pNKeepTail) {
  if (pSequential && vHasCpl) {
    DGS_LOG_ERROR(
      "%s: sequential split is not supported when there are coupled sequences in the input batch (you may need to use "
      "the -kvu flag)\n",
      __func__);

    return {};
  }

  std::vector<seq_set_t> lCurSeqSet;

  dgs_seq_id vLastSeqId = -1;

  // determine the non-overlapping sequence sets participating in this ubatch
  for (int32_t vI = 0; vI < vBatch.vNTokens; ++vI) {
    if (lUsed[vI]) {
      continue;
    }

    bool vAdd = true;

    for (uint32_t vS = 0; vS < lCurSeqSet.size(); ++vS) {
      // no overlap with existing sequence sets:
      if (!(lCurSeqSet[vS] & lSeqSet[vI]).none()) {
        vAdd = false;
        break;
      }
    }

    // accept only increasing sequence ids
    if (pSequential) {
      vAdd = vAdd && (lCurSeqSet.empty() || vBatch.vSeqId[vI][0] == vLastSeqId + 1);
    }

    if (vAdd) {
      lCurSeqSet.push_back(lSeqSet[vI]);

      vLastSeqId = vBatch.vSeqId[vI][0];

      if (lCurSeqSet.size() > pNUbatch) {
        break;
      }
    }
  }

  uint32_t vNSeqs = lCurSeqSet.size();

  // we are done
  if (vNSeqs == 0) {
    return {};
  }

  // the current batch index of each sequence set
  std::vector<int32_t> lCurIdx(vNSeqs, 0);

  for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
    while (lUsed[dSeqSetMap[lCurSeqSet[vS]][lCurIdx[vS]]]) {
      ++lCurIdx[vS];
    }
  }

  // the list of batch indices for each sequence set
  // at the end we will concat these to get the final ubatch
  std::vector<idx_vec_t> lIdxsPerSeq(vNSeqs);

  while (true) {
    // we can only add new n_seq_tokens tokens if all the sequence sets have at least 1 more unused tokens and
    //   if we haven't reached n_ubatch
    bool vCanExpand = true;

    for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
      if (lCurIdx[vS] >= (int32_t) dSeqSetMap[lCurSeqSet[vS]].size()) {
        vCanExpand = false;
        break;
      }
    }

    if (!vCanExpand) {
      break;
    }

    for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
      const int32_t cIdx = dSeqSetMap[lCurSeqSet[vS]][lCurIdx[vS]];

      lIdxsPerSeq[vS].push_back(cIdx);

      lUsed[cIdx] = true;
      ++vNUsed;

      ++lCurIdx[vS];
    }

    if ((lIdxsPerSeq[0].size() + 1) * vNSeqs > pNUbatch) {
      break;
    }
  }

  // if n_keep_tail > 0, keep only the seqs that either finish in this ubatch or have at least
  //   n_keep_tail tokens remaining for a future ubatch, so that the trailing n_keep_tail tokens
  //   of each seq are never split across ubatches
  if (pNKeepTail > 0) {
    GGML_ASSERT(pNUbatch > pNKeepTail);

    auto vNRemaining = [&](uint32_t pS) {
      return (uint32_t) (dSeqSetMap[lCurSeqSet[pS]].size() - lCurIdx[pS]);
    };

    // keep the longest prefix of seqs that satisfy the constraint, to preserve sequential seq ids
    uint32_t vNKeep = 0;
    while (vNKeep < vNSeqs) {
      const uint32_t cRemaining = vNRemaining(vNKeep);

      if (cRemaining != 0 && cRemaining < pNKeepTail) {
        break;
      }

      vNKeep++;
    }

    // all seqs violate the constraint - resolve the first one directly and emit it alone
    if (vNKeep == 0) {
      auto & lIdxs = lIdxsPerSeq[0];

      const auto & lSeqIdxs = dSeqSetMap[lCurSeqSet[0]];

      if (lIdxs.size() + vNRemaining(0) <= pNUbatch) {
        // extend the seq to completion
        while (vNRemaining(0) > 0) {
          const int32_t cIdx = lSeqIdxs[lCurIdx[0]];

          lIdxs.push_back(cIdx);

          lUsed[cIdx] = true;
          ++vNUsed;

          ++lCurIdx[0];
        }
      } else {
        // truncate the seq so that at least n_keep_tail tokens remain
        while (vNRemaining(0) < pNKeepTail) {
          lUsed[lIdxs.back()] = false;
          --vNUsed;

          lIdxs.pop_back();

          --lCurIdx[0];
        }
      }

      vNKeep = 1;
    }

    // return the tokens of the deferred seqs back to the pool
    for (uint32_t vS = vNKeep; vS < vNSeqs; ++vS) {
      for (const int32_t cIdx : lIdxsPerSeq[vS]) {
        lUsed[cIdx] = false;
        --vNUsed;
      }
    }

    vNSeqs = vNKeep;
  }

  // concat the per-sequence-set lists
  std::vector<int32_t> lIdxs;

  for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
    lIdxs.insert(lIdxs.end(), lIdxsPerSeq[vS].begin(), lIdxsPerSeq[vS].end());
  }

  return mUbatchAdd(lIdxs, vNSeqs, true);
}

dgs_ubatch dgs_batch_allocr::mSplitSeq(uint32_t pNUbatch) {
  // find the first unused token
  uint32_t vCurIdx = 0;
  while (vCurIdx < lUsed.size() && lUsed[vCurIdx]) {
    ++vCurIdx;
  }

  // we are done
  if (vCurIdx >= lUsed.size()) {
    return {};
  }

  // this is the starting sequence set
  // we allow adding tokens only if their sequence set is a subset of the current sequence set
  auto vCurSeqSet = lSeqSet[vCurIdx];

  std::vector<int32_t> lIdxs;

  while (true) {
    lIdxs.push_back(vCurIdx);

    lUsed[vCurIdx] = true;
    ++vNUsed;

    if (lIdxs.size() >= pNUbatch) {
      break;
    }

    do {
      ++vCurIdx;
    } while (vCurIdx < mGetNTokens() && (lUsed[vCurIdx] || ((vCurSeqSet & lSeqSet[vCurIdx]) != lSeqSet[vCurIdx])));

    if (vCurIdx == mGetNTokens()) {
      break;
    }

    vCurSeqSet = lSeqSet[vCurIdx];
  }

  return mUbatchAdd(lIdxs, 1, true);
}

void dgs_batch_allocr::clear() {
  vNOutputs = 0;

  vBatch = {};

  lTokenVec.clear();
  lEmbdVec.clear();
  lIsEmbdVec.clear();
  lStateVec.clear();
  lSeqIdData.clear();
  lPos.clear();
  lNSeqId.clear();
  lSeqId.clear();
  lSeqIdUnq.clear();
  lOutput.clear();
  lDecisionOrder.clear();

  for (auto & sCur : sSeqPos) {
    sCur.clear();
  }

  for (auto & lCur : lSeqCpl) {
    std::fill(lCur.begin(), lCur.end(), false);
  }

  lSeqSet.clear();

  dSeqSetMap.clear();

  std::fill(lSeqIdx.begin(), lSeqIdx.end(), -1);
}

dgs_ubatch dgs_batch_allocr::mUbatchAdd(const std::vector<int32_t> & pIdxs, uint32_t pNSeqs, bool pEqualSeqs) {
  const uint32_t cNTokens = pIdxs.size();

  assert(n_tokens % n_seqs == 0);

  auto vUdata = std::make_shared<dgs_ubatch::data_t>();

  const bool cMixedBatch = !lIsEmbdVec.empty();

  // a ubatch with a single kind of rows is emitted as a plain token or embd ubatch
  uint32_t vNEmbdRows = 0;
  if (cMixedBatch) {
    for (int32_t vIdx : pIdxs) {
      vNEmbdRows += lIsEmbdVec[vIdx];
    }
  }
  const bool cMixed    = cMixedBatch && vNEmbdRows > 0 && vNEmbdRows < cNTokens;
  const bool cUseToken = vBatch.token && !(cMixedBatch && vNEmbdRows == cNTokens);
  const bool cUseEmbd  = vBatch.vEmbd && !(cMixedBatch && vNEmbdRows == 0);
  const bool cHasState = !lStateVec.empty();

  const int64_t cNEmbdAll  = cUseEmbd ? (int64_t) cNTokens * vNEmbd : 0;
  const int64_t cNStateAll = cHasState ? (int64_t) cNTokens * vNEmbdState : 0;
  const int64_t cNPosAll   = (int64_t) cNTokens * cNPosPerEmbd;

  vUdata->token.resize(cNTokens);
  vUdata->lEmbd.resize(cNEmbdAll);
  vUdata->lEmbdState.resize(cNStateAll);
  vUdata->lPos.resize(cNPosAll);
  vUdata->lNSeqId.resize(cNTokens);
  vUdata->lSeqId.resize(cNTokens);
  vUdata->lSeqIdUnq.resize(0);
  vUdata->lSeqIdx.resize(DGS_MAX_SEQ, -1);
  vUdata->lOutput.resize(cNTokens);
  vUdata->type.resize(cMixed ? cNTokens : 0);
  vUdata->lDecisionOrder.resize(lDecisionOrder.empty() ? 0 : cNTokens);

  vUdata->lBatchIdxs = pIdxs;
  vUdata->lSeqIdData.reserve(cNTokens);

  seq_set_t vSeqSetUnq;

  for (size_t vI = 0; vI < pIdxs.size(); ++vI) {
    if (cUseToken) {
      vUdata->token[vI] = vBatch.token[pIdxs[vI]];
    }

    if (cUseEmbd) {
      memcpy(vUdata->lEmbd.data() + vI * vNEmbd, vBatch.vEmbd + (int64_t) pIdxs[vI] * vNEmbd, vNEmbd * sizeof(float));
    }

    if (cMixed) {
      vUdata->type[vI] = lIsEmbdVec[pIdxs[vI]];
    }

    if (cHasState) {
      memcpy(vUdata->lEmbdState.data() + vI * vNEmbdState, lStateVec.data() + (int64_t) pIdxs[vI] * vNEmbdState,
             vNEmbdState * sizeof(float));
    }

    for (size_t vJ = 0; vJ < (size_t) cNPosPerEmbd; ++vJ) {
      vUdata->lPos[vJ * cNTokens + vI] = vBatch.vPos[vJ * vBatch.vNTokens + pIdxs[vI]];
    }

    vUdata->lNSeqId[vI] = vBatch.vNSeqId[pIdxs[vI]];
    vUdata->lOutput[vI] = vBatch.vLogits[pIdxs[vI]];

    if (!lDecisionOrder.empty()) {
      vUdata->lDecisionOrder[vI] = lDecisionOrder[pIdxs[vI]];
    }

    for (int vS = 0; vS < vUdata->lNSeqId[vI]; ++vS) {
      const dgs_seq_id cSeqId = vBatch.vSeqId[pIdxs[vI]][vS];

      vUdata->lSeqIdData.push_back(cSeqId);
      vSeqSetUnq.set(cSeqId);
    }

    if (vUdata->lOutput[vI]) {
      lOutIds.push_back(pIdxs[vI]);
    }
  }

  dgs_seq_id * vSeqIdPtr = vUdata->lSeqIdData.data();
  for (size_t vI = 0; vI < pIdxs.size(); ++vI) {
    vUdata->lSeqId[vI] = vSeqIdPtr;
    vSeqIdPtr += vUdata->lNSeqId[vI];
  }

  for (uint32_t vS = 0; vS < vNSeqMax; ++vS) {
    if (vSeqSetUnq.test(vS)) {
      vUdata->lSeqIdx[vS] = vUdata->lSeqIdUnq.size();
      vUdata->lSeqIdUnq.push_back(vS);
    }
  }

  dgs_ubatch vRes{
    /*.b_equal_seqs =*/pEqualSeqs,
    /*.n_tokens     =*/cNTokens,
    /*.n_seq_tokens =*/cNTokens / pNSeqs,
    /*.n_seqs       =*/pNSeqs,
    /*.n_seqs_unq   =*/(uint32_t) vUdata->lSeqIdUnq.size(),
    /*.n_pos        =*/cNPosPerEmbd,

    /*.token        =*/cUseToken ? vUdata->token.data() : nullptr,
    /*.embd         =*/cUseEmbd ? vUdata->lEmbd.data() : nullptr,
    /*.embd_state   =*/cHasState ? vUdata->lEmbdState.data() : nullptr,
    /*.pos          =*/vUdata->lPos.data(),
    /*.n_seq_id     =*/vUdata->lNSeqId.data(),
    /*.seq_id       =*/vUdata->lSeqId.data(),
    /*.seq_id_unq   =*/vUdata->lSeqIdUnq.data(),
    /*.seq_idx      =*/vUdata->lSeqIdx.data(),
    /*.output       =*/vUdata->lOutput.data(),
    /*.type         =*/cMixed ? vUdata->type.data() : nullptr,
    /*.decision_order =*/vUdata->lDecisionOrder.empty() ? nullptr : vUdata->lDecisionOrder.data(),
    /*.data         =*/std::move(vUdata),
  };

  if (vDebug > 0) {
    DGS_LOG_DEBUG("%s: added ubatch to split:\n", __func__);

    mUbatchPrint(vRes, vDebug);
  }

  return vRes;
}

void dgs_batch_allocr::mUbatchPrint(const dgs_ubatch & pUbatch, int pDebug) {
  if (pDebug > 0) {
    DGS_LOG_DEBUG("%s:   equal_seqs   = %d\n", __func__, pUbatch.mEqualSeqs());
    DGS_LOG_DEBUG("%s:   n_tokens     = %d\n", __func__, pUbatch.vNTokens);
    DGS_LOG_DEBUG("%s:   n_seq_tokens = %d\n", __func__, pUbatch.vNSeqTokens);
    DGS_LOG_DEBUG("%s:   n_seqs       = %d\n", __func__, pUbatch.vNSeqs);
    DGS_LOG_DEBUG("%s:   n_seqs_unq   = %d\n", __func__, pUbatch.vNSeqsUnq);

    std::stringstream vSsSeqIdUnq;
    std::stringstream vSsSeqIdx;

    vSsSeqIdUnq << "[ ";
    vSsSeqIdx << "[";

    for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
      vSsSeqIdUnq << pUbatch.vSeqIdUnq[vS] << " ";
    }

    for (uint32_t vS = 0; vS < DGS_MAX_SEQ; ++vS) {
      if (pUbatch.vSeqIdx[vS] >= 0) {
        vSsSeqIdx << pUbatch.vSeqIdx[vS] % 10;
      } else {
        vSsSeqIdx << ".";
      }
    }

    vSsSeqIdUnq << "]";
    vSsSeqIdx << "]";

    DGS_LOG_DEBUG("%s:   token      = %p\n", __func__, (void *) pUbatch.token);
    DGS_LOG_DEBUG("%s:   embd       = %p\n", __func__, (void *) pUbatch.vEmbd);
    DGS_LOG_DEBUG("%s:   embd_state = %p\n", __func__, (void *) pUbatch.vEmbdState);
    DGS_LOG_DEBUG("%s:   pos        = %p\n", __func__, (void *) pUbatch.vPos);
    DGS_LOG_DEBUG("%s:   n_seq_id   = %p\n", __func__, (void *) pUbatch.vNSeqId);
    DGS_LOG_DEBUG("%s:   seq_id     = %p\n", __func__, (void *) pUbatch.vSeqId);
    DGS_LOG_DEBUG("%s:   seq_id_unq = %s\n", __func__, vSsSeqIdUnq.str().c_str());
    DGS_LOG_DEBUG("%s:   seq_idx    = %s\n", __func__, vSsSeqIdx.str().c_str());
    DGS_LOG_DEBUG("%s:   output     = %p\n", __func__, (void *) pUbatch.vOutput);
    DGS_LOG_DEBUG("%s:   type       = %p\n", __func__, (void *) pUbatch.type);
    DGS_LOG_DEBUG("%s:   n_outputs  = %d\n", __func__, vNOutputs);

    if (pDebug > 0) {
      int vSeqIdMax = 0;
      for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
        for (int vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
          for (int vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
            vSeqIdMax = std::max(vSeqIdMax, pUbatch.vSeqId[vI][vS]);
          }
        }
      }
      ++vSeqIdMax;

      DGS_LOG_DEBUG("%s:   token     = [\n", __func__);
      for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
        std::vector<int8_t> lSeqId(vSeqIdMax);

        for (int vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
          lSeqId[pUbatch.vSeqId[vI][vS]] = 1;
        }

        std::stringstream vSs;
        for (int vS = 0; vS < vSeqIdMax; ++vS) {
          if (lSeqId[vS]) {
            vSs << vS % 10;
          } else {
            vSs << ".";
          }
        }

        if (pUbatch.token && !(pUbatch.mIsMixed() && pUbatch.type[vI])) {
          DGS_LOG_DEBUG("%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n", __func__,
                        vI, pUbatch.token[vI], vVocab->mTokenToPiece(pUbatch.token[vI]).c_str(), pUbatch.vPos[vI],
                        pUbatch.vNSeqId[vI], vSs.str().c_str(), pUbatch.vOutput[vI]);
        } else {
          DGS_LOG_DEBUG("%s:  %4d: [embd], pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n", __func__, vI,
                        pUbatch.vPos[vI], pUbatch.vNSeqId[vI], vSs.str().c_str(), pUbatch.vOutput[vI]);
        }
      }
      DGS_LOG_DEBUG("%s:   ]\n", __func__);
    }
  }
}

//
// interface implementation
//

struct dgs_batch fDgsBatchGetOne(dgs_token * pTokens, int32_t pNTokens) {
  return {
    /*n_tokens =*/pNTokens,
    /*tokens   =*/pTokens,
    /*embd     =*/nullptr,
    /*pos      =*/nullptr,
    /*n_seq_id =*/nullptr,
    /*seq_id   =*/nullptr,
    /*logits   =*/nullptr,
  };
}

struct dgs_batch fDgsBatchInit(int32_t pNTokensAlloc, int32_t pEmbd, int32_t pNSeqMax) {
  dgs_batch vBatch = {
    /*n_tokens =*/0,
    /*tokens   =*/nullptr,
    /*embd     =*/nullptr,
    /*pos      =*/nullptr,
    /*n_seq_id =*/nullptr,
    /*seq_id   =*/nullptr,
    /*logits   =*/nullptr,
  };

  if (pEmbd) {
    vBatch.vEmbd = (float *) malloc(sizeof(float) * pNTokensAlloc * pEmbd);
  } else {
    vBatch.token = (dgs_token *) malloc(sizeof(dgs_token) * pNTokensAlloc);
  }

  vBatch.vPos    = (dgs_pos *) malloc(sizeof(dgs_pos) * pNTokensAlloc);
  vBatch.vNSeqId = (int32_t *) malloc(sizeof(int32_t) * pNTokensAlloc);
  vBatch.vSeqId  = (dgs_seq_id **) malloc(sizeof(dgs_seq_id *) * (pNTokensAlloc + 1));
  for (int vI = 0; vI < pNTokensAlloc; ++vI) {
    vBatch.vSeqId[vI] = (dgs_seq_id *) malloc(sizeof(dgs_seq_id) * pNSeqMax);
  }
  vBatch.vSeqId[pNTokensAlloc] = nullptr;

  vBatch.vLogits = (int8_t *) malloc(sizeof(int8_t) * pNTokensAlloc);

  return vBatch;
}

void fDgsBatchFree(struct dgs_batch pBatch) {
  if (pBatch.token) {
    free(pBatch.token);
  }
  if (pBatch.vEmbd) {
    free(pBatch.vEmbd);
  }
  if (pBatch.vPos) {
    free(pBatch.vPos);
  }
  if (pBatch.vNSeqId) {
    free(pBatch.vNSeqId);
  }
  if (pBatch.vSeqId) {
    for (int vI = 0; pBatch.vSeqId[vI] != nullptr; ++vI) {
      free(pBatch.vSeqId[vI]);
    }
    free(pBatch.vSeqId);
  }
  if (pBatch.vLogits) {
    free(pBatch.vLogits);
  }
}

// dgs_batch_ext

size_t fDgsBatchExtSelectNEmbdInp(dgs_context_type pCtxType, llm_arch pArch, const dgs_hparams & pHparams) {
  GGML_UNUSED(pCtxType);
  if (pArch == cLlmArchDflash) {
    return pHparams.mNEmbdInpEnc();
  }
  return pHparams.mNEmbdInp();
}

size_t fDgsBatchExtSelectNEmbdState(dgs_context_type pCtxType, const dgs_hparams & pHparams) {
  if (pCtxType == cDgsContextTypeMtp) {
    return pHparams.mNEmbdOut();
  }
  return 0;
}

dgs_batch_ext::dgs_batch_ext(dgs_context * pCtx) :
  cNTokensMax(fDgsNBatch(pCtx)),
  cNEmbdInp(
    fDgsBatchExtSelectNEmbdInp(pCtx->mGetCparams().vCtxType, fDgsGetModel(pCtx)->vArch, fDgsGetModel(pCtx)->hparams)),
  cNEmbdInpEnc(fDgsGetModel(pCtx)->hparams.mNEmbdInpEnc()),
  cNEmbdState(fDgsBatchExtSelectNEmbdState(pCtx->mGetCparams().vCtxType, fDgsGetModel(pCtx)->hparams)),
  cNSeqMax(fDgsNSeqMax(pCtx)),
  vMem(fDgsGetMemory(pCtx)),
  cNVocab(fDgsVocabNTokens(fDgsModelGetVocab(fDgsGetModel(pCtx)))),
  cNPosPerEmbd(fDgsGetModel(pCtx)->hparams.mNPosPerEmbd()) {
  clear();
}

dgs_batch_ext::dgs_batch_ext(size_t         pNTokensMax,
                             size_t         pNEmbdInp,
                             size_t         pNEmbdInpEnc,
                             dgs_seq_id     pNSeqMax,
                             dgs_memory_i * pMem,
                             dgs_token      pNVocab,
                             size_t         pNPosPerEmbd) :
  cNTokensMax(pNTokensMax),
  cNEmbdInp(pNEmbdInp),
  cNEmbdInpEnc(pNEmbdInpEnc),
  cNEmbdState(0),
  cNSeqMax(pNSeqMax),
  vMem(pMem),
  cNVocab(pNVocab),
  cNPosPerEmbd(pNPosPerEmbd) {
  clear();
}

void dgs_batch_ext::clear() {
  lTokens.clear();
  lEmbd.clear();
  lState.clear();
  vNEmbd = 0;
}

int32_t dgs_batch_ext::mAddToken(dgs_seq_id pSeqId) {
  if (lTokens.size() >= cNTokensMax) {
    return -1;  // size limit reached
  }
  if (pSeqId < 0 || pSeqId >= cNSeqMax) {
    return -3;  // invalid sequence id
  }

  // position is left undefined; call set_token_pos() before decoding
  token vT;
  vT.sSeqIds.insert(pSeqId);

  lTokens.push_back(vT);

  return (int32_t) (lTokens.size() - 1);
}

bool dgs_batch_ext::mAddSeq(int32_t pIdx, dgs_seq_id pSeqId) {
  if (pIdx < 0 || pIdx >= (int32_t) lTokens.size()) {
    return false;
  }
  if (pSeqId < 0 || pSeqId >= cNSeqMax) {
    return false;
  }

  token & vT = lTokens[pIdx];

  vT.sSeqIds.insert(pSeqId);

  return true;
}

bool dgs_batch_ext::mSetTokenId(int32_t pIdx, dgs_token pId) {
  if (pIdx < 0 || pIdx >= (int32_t) lTokens.size()) {
    return false;
  }
  if (pId < 0 || pId >= cNVocab) {
    return false;
  }
  lTokens[pIdx].vId = pId;
  return true;
}

bool dgs_batch_ext::mSetTokenEmbd(int32_t pIdx, dgs_embd pEmbdIn) {
  if (pIdx < 0 || pIdx >= (int32_t) lTokens.size()) {
    return false;
  }
  if (!pEmbdIn.data) {
    return false;
  }

  const size_t cNTotal = pEmbdIn.vNRows * pEmbdIn.vNEmbd;
  if (vNEmbd == 0) {
    if (cNTotal != cNEmbdInp && cNTotal != cNEmbdInpEnc) {
      DGS_LOG_ERROR("%s: embedding size mismatch, got %zu rows x %zu = %zu, expected %zu or %zu\n", __func__,
                    pEmbdIn.vNRows, pEmbdIn.vNEmbd, cNTotal, cNEmbdInp, cNEmbdInpEnc);
      return false;
    }
    vNEmbd = cNTotal;
  } else if (cNTotal != vNEmbd) {
    DGS_LOG_ERROR("%s: embedding size mismatch, got %zu rows x %zu = %zu, expected %zu\n", __func__, pEmbdIn.vNRows,
                  pEmbdIn.vNEmbd, cNTotal, vNEmbd);
    return false;
  }

  token & vT = lTokens[pIdx];

  if (vT.vHasEmbd) {
    DGS_LOG_ERROR("%s: embedding for token %d is already set\n", __func__, pIdx);
    return false;
  }

  vT.vHasEmbd = true;
  vT.vEmbdOff = lEmbd.size();
  lEmbd.insert(lEmbd.end(), pEmbdIn.data, pEmbdIn.data + cNTotal);

  return true;
}

bool dgs_batch_ext::mSetTokenState(int32_t pIdx, dgs_embd pStateIn) {
  if (pIdx < 0 || pIdx >= (int32_t) lTokens.size()) {
    return false;
  }
  if (!pStateIn.data) {
    return false;
  }
  if (cNEmbdState == 0) {
    return false;  // this context does not take state embeddings
  }

  const size_t cNTotal = pStateIn.vNRows * pStateIn.vNEmbd;
  if (cNTotal != cNEmbdState) {
    DGS_LOG_ERROR("%s: state size mismatch, got %zu rows x %zu = %zu, expected %zu\n", __func__, pStateIn.vNRows,
                  pStateIn.vNEmbd, cNTotal, cNEmbdState);
    return false;
  }

  token & vT = lTokens[pIdx];

  if (vT.vHasState) {
    DGS_LOG_ERROR("%s: state for token %d is already set\n", __func__, pIdx);
    return false;
  }

  vT.vHasState = true;
  vT.vStateOff = lState.size();
  lState.insert(lState.end(), pStateIn.data, pStateIn.data + cNTotal);

  return true;
}

bool dgs_batch_ext::mSetTokenPos(int32_t pIdx, const dgs_pos * pPosIn) {
  if (pIdx < 0 || pIdx >= (int32_t) lTokens.size()) {
    return false;
  }
  if (!pPosIn) {
    return false;
  }

  token & vT = lTokens[pIdx];

  size_t vNPos = vT.vId != DGS_TOKEN_NULL ? 1 : cNPosPerEmbd;
  for (size_t vI = 0; vI < vNPos; ++vI) {
    vT.aPos[vI] = pPosIn[vI];
  }

  return true;
}

bool dgs_batch_ext::mSetOutput(int32_t pIdx, bool pOutputLast) {
  if (pIdx < 0 || pIdx >= (int32_t) lTokens.size()) {
    return false;
  }
  lTokens[pIdx].vOutput = pOutputLast;
  return true;
}

bool dgs_batch_ext::mSetDecisionOrder(int32_t pIdx, int32_t pOrder) {
  if (pIdx < 0 || pIdx >= (int32_t) lTokens.size()) {
    return false;
  }
  lTokens[pIdx].vDecisionOrder = pOrder;
  return true;
}

// dgs_batch_ext C API

dgs_batch_ext * fDgsBatchExtInit(dgs_context * pCtx) {
  return new dgs_batch_ext(pCtx);
}

void fDgsBatchExtFree(dgs_batch_ext * pBatch) {
  delete pBatch;
}

void fDgsBatchExtClear(dgs_batch_ext * pBatch) {
  pBatch->clear();
}

int32_t fDgsBatchExtAdd(dgs_batch_ext * pBatch, dgs_seq_id pSeqId) {
  return pBatch->mAddToken(pSeqId);
}

int32_t fDgsBatchExtAddToken(dgs_batch_ext * pBatch, dgs_seq_id pSeqId, dgs_token pId) {
  int32_t vIdx = pBatch->mAddToken(pSeqId);
  if (vIdx < 0) {
    return vIdx;
  }
  if (!pBatch->mSetTokenId(vIdx, pId)) {
    return -2;
  }
  return vIdx;
}

int32_t fDgsBatchExtAddEmbd(dgs_batch_ext * pBatch, dgs_seq_id pSeqId, dgs_embd pEmbd) {
  int32_t vIdx = pBatch->mAddToken(pSeqId);
  if (vIdx < 0) {
    return vIdx;
  }
  if (!pBatch->mSetTokenEmbd(vIdx, pEmbd)) {
    return -2;
  }
  return vIdx;
}

bool fDgsBatchExtAddSeq(dgs_batch_ext * pBatch, int32_t pIdx, dgs_seq_id pSeqId) {
  return pBatch->mAddSeq(pIdx, pSeqId);
}

bool fDgsBatchExtSetPos(dgs_batch_ext * pBatch, int32_t pIdx, const dgs_pos * pPos) {
  return pBatch->mSetTokenPos(pIdx, pPos);
}

bool fDgsBatchExtSetEmbdToken(dgs_batch_ext * pBatch, int32_t pIdx, dgs_embd pEmbd) {
  return pBatch->mSetTokenEmbd(pIdx, pEmbd);
}

bool fDgsBatchExtSetEmbdState(dgs_batch_ext * pBatch, int32_t pIdx, dgs_embd pEmbd) {
  return pBatch->mSetTokenState(pIdx, pEmbd);
}

bool fDgsBatchExtSetOutputEmbd(dgs_batch_ext * pBatch, int32_t pIdx, bool value) {
  return pBatch->mSetOutput(pIdx, value);
}

bool fDgsBatchExtSetOutputLogits(dgs_batch_ext * pBatch, int32_t pIdx, bool value) {
  return pBatch->mSetOutput(pIdx, value);
}

bool fDgsBatchExtSetDecisionOrder(dgs_batch_ext * pBatch, int32_t pIdx, dgs_decision_order pOrder) {
  return pBatch->mSetDecisionOrder(pIdx, pOrder);
}

// dgs_batch_compat

void dgs_batch_compat::init(dgs_batch_ext & pDst, const dgs_batch & pBatchInp, size_t pNEmbdRow) {
  dgs_batch_ext * vBatchExt = &pDst;

  if (pNEmbdRow == 0) {
    pNEmbdRow = vBatchExt->cNEmbdInp;
  }

  // a batch can carry both, for example the MTP hook batches
  const bool cHasToken = pBatchInp.token != nullptr;
  const bool cHasEmbd  = pBatchInp.vEmbd != nullptr;

  static const dgs_seq_id cDefaultSeqId  = 0;
  static const int32_t    cDefaultNSeqId = 1;

  // auto-generates positions locally when batch_inp.pos is null, continuing from memory
  std::vector<dgs_pos> lPosNext(vBatchExt->cNSeqMax);
  for (dgs_seq_id vS = 0; vS < (dgs_seq_id) vBatchExt->cNSeqMax; ++vS) {
    lPosNext[vS] = fDgsMemorySeqPosMax(vBatchExt->vMem, vS) + 1;  // assume next pos
  }

  for (int32_t vI = 0; vI < pBatchInp.vNTokens; ++vI) {
    const int32_t      cNSid = pBatchInp.vNSeqId ? pBatchInp.vNSeqId[vI] : cDefaultNSeqId;
    const dgs_seq_id * vSids = pBatchInp.vSeqId ? pBatchInp.vSeqId[vI] : &cDefaultSeqId;

    dgs_batch_ext::token vT;

    // seq_ids
    for (int32_t vS = 0; vS < cNSid; ++vS) {
      vT.sSeqIds.insert(vSids[vS]);
    }

    // position(s)
    if (pBatchInp.vPos) {
      if (cHasToken) {
        // token batch: one position per token
        vT.aPos[0] = pBatchInp.vPos[vI];
      } else {
        // embedding batch (M-RoPE): section-major layout pos[j*n_tokens + i]
        for (uint32_t vJ = 0; vJ < vBatchExt->cNPosPerEmbd; ++vJ) {
          vT.aPos[vJ] = pBatchInp.vPos[(int32_t) vJ * pBatchInp.vNTokens + vI];
        }
      }
    } else {
      // auto-generate position from the first seq_id
      vT.aPos[0] = lPosNext[vSids[0]]++;
    }

    // token id and/or embeddings
    if (cHasToken) {
      vT.vId = pBatchInp.token[vI];
    }

    // MTP batches carry the hidden state next to the token ids
    if (cHasEmbd && cHasToken && vBatchExt->cNEmbdState > 0) {
      vT.vHasState       = true;
      vT.vStateOff       = vBatchExt->lState.size();
      const float * vSrc = pBatchInp.vEmbd + (size_t) vI * vBatchExt->cNEmbdState;
      vBatchExt->lState.insert(vBatchExt->lState.end(), vSrc, vSrc + vBatchExt->cNEmbdState);
    } else if (cHasEmbd) {
      vT.vHasEmbd        = true;
      vT.vEmbdOff        = vBatchExt->lEmbd.size();
      const float * vSrc = pBatchInp.vEmbd + (size_t) vI * pNEmbdRow;
      vBatchExt->lEmbd.insert(vBatchExt->lEmbd.end(), vSrc, vSrc + pNEmbdRow);
      vBatchExt->vNEmbd = pNEmbdRow;
    }

    // output flag
    // if no logits array is given, default to only the last token being an output
    vT.vOutput = pBatchInp.vLogits ? (pBatchInp.vLogits[vI] != 0) : (vI == pBatchInp.vNTokens - 1);

    vBatchExt->lTokens.push_back(vT);
  }
}

dgs_batch_compat::dgs_batch_compat(dgs_context * pCtx, const dgs_batch & pBatchInp, size_t pNEmbdRow) {
  vBatchExt = new dgs_batch_ext(pCtx);
  init(*vBatchExt, pBatchInp, pNEmbdRow);
}

dgs_batch_compat::~dgs_batch_compat() {
  delete vBatchExt;
}
