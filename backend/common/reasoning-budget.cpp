#include "reasoning-budget.h"

#include "common.h"
#include "log.h"
#include "trie.h"
#include "unicode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
  std::vector<dgs_tokens> lSeqs;
  common_aho_corasick     vAc;
  size_t                  vState = 0;

  token_matcher(const std::vector<dgs_tokens> & pSeqs) : lSeqs(mCollect(pSeqs)), vAc(mBuildTrie(this->lSeqs)) {}

  static std::vector<dgs_tokens> mCollect(const std::vector<dgs_tokens> & pSeqs) {
    std::vector<dgs_tokens> lRes;
    for (const auto & lSeq : pSeqs) {
      if (!lSeq.empty() && std::find(lRes.begin(), lRes.end(), lSeq) == lRes.end()) {
        lRes.push_back(lSeq);
      }
    }
    return lRes;
  }

  static common_trie mBuildTrie(const std::vector<dgs_tokens> & pSeqs) {
    common_trie vT;
    for (const auto & lSeq : pSeqs) {
      vT.insert(std::vector<uint32_t>(lSeq.begin(), lSeq.end()));
    }
    return vT;
  }

  // returns the index into seqs of the longest sequence ending at this token, or -1
  int32_t mAdvance(dgs_token token) {
    vState           = vAc.mNext(vState, (uint32_t) token);
    const int32_t cP = vAc.mMatchPattern(vState);
    if (cP >= 0) {
      vState = 0;
    }
    return cP;
  }

  void reset() { vState = 0; }
};

struct common_reasoning_budget_ctx {
  const dgs_vocab * vVocab;

  token_matcher vStartMatcher;
  token_matcher vEndMatcher;
  dgs_tokens    lForcedTokens;

  int32_t vBudget;     // maximum tokens in reasoning block
  int32_t vRemaining;  // tokens remaining in budget

  common_reasoning_budget_state vState;

  // for forcing
  size_t vForcePos;   // next position in forced_tokens to force

  int32_t vEndMatch;  // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none
};

static const char * fCommonReasoningBudgetName(const struct dgs_sampler * /*smpl*/) {
  return "reasoning-budget";
}

static void fCommonReasoningBudgetAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  auto * vCtx = (common_reasoning_budget_ctx *) pSmpl->vCtx;

  switch (vCtx->vState) {
    case cReasoningBudgetIdle:
      {
        if (vCtx->vStartMatcher.mAdvance(token) >= 0) {
          vCtx->vState     = cReasoningBudgetCounting;
          vCtx->vRemaining = vCtx->vBudget;
          COM_TRC("activated, budget=%d tokens\n", vCtx->vBudget);

          if (vCtx->vRemaining <= 0) {
            vCtx->vState    = cReasoningBudgetForcing;
            vCtx->vForcePos = 0;
            COM_TRC("%s", "budget=0, forcing immediately\n");
          }
        }
        break;
      }
    case cReasoningBudgetCounting:
    case cReasoningBudgetWaitingUtf8:
      {
        const int32_t cMatch = vCtx->vEndMatcher.mAdvance(token);
        if (cMatch >= 0) {
          vCtx->vState    = cReasoningBudgetDone;
          vCtx->vEndMatch = cMatch;
          COM_TRC("%s", "deactivated (natural end)\n");
          break;
        }

        bool vUtf8Complete = true;
        if (vCtx->vVocab != nullptr) {
          const std::string cPiece = fCommonTokenToPiece(vCtx->vVocab, token, false);
          vUtf8Complete            = fCommonUtf8IsComplete(cPiece);
        }

        if (vCtx->vState == cReasoningBudgetWaitingUtf8) {
          if (vUtf8Complete) {
            vCtx->vState    = cReasoningBudgetForcing;
            vCtx->vForcePos = 0;
            vCtx->vEndMatcher.reset();
            COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
          }
        } else if (vCtx->vState == cReasoningBudgetCounting) {
          vCtx->vRemaining--;
          if (vCtx->vRemaining <= 0) {
            if (vUtf8Complete) {
              vCtx->vState    = cReasoningBudgetForcing;
              vCtx->vForcePos = 0;
              vCtx->vEndMatcher.reset();
              COM_TRC("%s", "budget exhausted, forcing end sequence\n");
            } else {
              vCtx->vState = cReasoningBudgetWaitingUtf8;
              vCtx->vEndMatcher.reset();
              COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
            }
          }
        }
        break;
      }
    case cReasoningBudgetForcing:
      {
        // track the end sequence within forced_tokens so it is also reported on DONE
        const int32_t cMatch = vCtx->vEndMatcher.mAdvance(token);
        vCtx->vForcePos++;
        if (vCtx->vForcePos >= vCtx->lForcedTokens.size()) {
          vCtx->vState    = cReasoningBudgetDone;
          vCtx->vEndMatch = cMatch;
          COM_TRC("%s", "forced sequence complete, done\n");
        }
        break;
      }
    case cReasoningBudgetDone:
      // Re-arm on a new start tag: some models emit multiple <think> blocks
      // per response, and each should get a fresh budget window.
      if (vCtx->vStartMatcher.mAdvance(token) >= 0) {
        vCtx->vState     = cReasoningBudgetCounting;
        vCtx->vRemaining = vCtx->vBudget;
        vCtx->vEndMatcher.reset();
        vCtx->vEndMatch = -1;
        COM_TRC("re-activated on new start tag, budget=%d tokens\n", vCtx->vBudget);

        if (vCtx->vRemaining <= 0) {
          vCtx->vState    = cReasoningBudgetForcing;
          vCtx->vForcePos = 0;
          COM_TRC("%s", "budget=0, forcing immediately\n");
        }
      }
      break;
  }
}

static void fCommonReasoningBudgetApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (common_reasoning_budget_ctx *) pSmpl->vCtx;

  if (vCtx->vState != cReasoningBudgetForcing) {
    // passthrough — don't modify logits
    return;
  }

  if (vCtx->vForcePos >= vCtx->lForcedTokens.size()) {
    return;
  }

  const dgs_token cForced = vCtx->lForcedTokens[vCtx->vForcePos];

  // set all logits to -inf except the forced token
  for (size_t vI = 0; vI < pCurP->size; vI++) {
    if (pCurP->data[vI].vId != cForced) {
      pCurP->data[vI].vLogit = -INFINITY;
    }
  }
}

static void fCommonReasoningBudgetReset(struct dgs_sampler * pSmpl) {
  auto * vCtx      = (common_reasoning_budget_ctx *) pSmpl->vCtx;
  vCtx->vState     = cReasoningBudgetIdle;
  vCtx->vRemaining = vCtx->vBudget;
  vCtx->vStartMatcher.reset();
  vCtx->vEndMatcher.reset();
  vCtx->vForcePos = 0;
  vCtx->vEndMatch = -1;
}

static struct dgs_sampler * fCommonReasoningBudgetInitState(const struct dgs_vocab *        pVocab,
                                                            const std::vector<dgs_tokens> & pStartSeqs,
                                                            const std::vector<dgs_tokens> & pEndSeqs,
                                                            const dgs_tokens &              pForcedTokens,
                                                            int32_t                         pBudget,
                                                            common_reasoning_budget_state   pInitialState);

static struct dgs_sampler * fCommonReasoningBudgetClone(const struct dgs_sampler * pSmpl);

static void fCommonReasoningBudgetFree(struct dgs_sampler * pSmpl) {
  delete (common_reasoning_budget_ctx *) pSmpl->vCtx;
}

static struct dgs_sampler_i vCommonReasoningBudgetI = {
  /* .name              = */ fCommonReasoningBudgetName,
  /* .accept            = */ fCommonReasoningBudgetAccept,
  /* .apply             = */ fCommonReasoningBudgetApply,
  /* .reset             = */ fCommonReasoningBudgetReset,
  /* .clone             = */ fCommonReasoningBudgetClone,
  /* .free              = */ fCommonReasoningBudgetFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

static struct dgs_sampler * fCommonReasoningBudgetClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const common_reasoning_budget_ctx *) pSmpl->vCtx;

  return fDgsSamplerInit(
    /* .iface = */ &vCommonReasoningBudgetI,
    /* .ctx   = */ new common_reasoning_budget_ctx(*vCtx));
}

static struct dgs_sampler * fCommonReasoningBudgetInitState(const struct dgs_vocab *        pVocab,
                                                            const std::vector<dgs_tokens> & pStartSeqs,
                                                            const std::vector<dgs_tokens> & pEndSeqs,
                                                            const dgs_tokens &              pForcedTokens,
                                                            int32_t                         pBudget,
                                                            common_reasoning_budget_state   pInitialState) {
  // promote COUNTING with budget <= 0 to FORCING
  if (pInitialState == cReasoningBudgetCounting && pBudget <= 0) {
    pInitialState = cReasoningBudgetForcing;
  }

  return fDgsSamplerInit(
    /* .iface = */ &vCommonReasoningBudgetI,
    /* .ctx   = */ new common_reasoning_budget_ctx{
      /* .vocab         = */ pVocab,
      /* .start_matcher = */ token_matcher(pStartSeqs),
      /* .end_matcher   = */ token_matcher(pEndSeqs),
      /* .forced_tokens = */ pForcedTokens,
      /* .budget        = */ pBudget,
      /* .remaining     = */ pBudget,
      /* .state         = */ pInitialState,
      /* .force_pos     = */ 0,
      /* .end_match     = */ -1,
    });
}

struct dgs_sampler * fCommonReasoningBudgetInit(const struct dgs_vocab *        pVocab,
                                                const std::vector<dgs_tokens> & pStartSeqs,
                                                const std::vector<dgs_tokens> & pEndSeqs,
                                                const dgs_tokens &              pForcedTokens,
                                                int32_t                         pBudget,
                                                common_reasoning_budget_state   pInitialState) {
  return fCommonReasoningBudgetInitState(pVocab, pStartSeqs, pEndSeqs, pForcedTokens, pBudget, pInitialState);
}

common_reasoning_budget_state fCommonReasoningBudgetGetState(const struct dgs_sampler * pSmpl) {
  if (!pSmpl) {
    return cReasoningBudgetIdle;
  }
  return ((const common_reasoning_budget_ctx *) pSmpl->vCtx)->vState;
}

const dgs_tokens * fCommonReasoningBudgetGetEndMatch(const struct dgs_sampler * pSmpl) {
  if (!pSmpl) {
    return nullptr;
  }

  const auto * vCtx = (const common_reasoning_budget_ctx *) pSmpl->vCtx;
  if (vCtx->vEndMatch < 0) {
    return nullptr;
  }

  return &vCtx->vEndMatcher.lSeqs[vCtx->vEndMatch];
}

bool fCommonReasoningBudgetForce(struct dgs_sampler * pSmpl) {
  if (!pSmpl) {
    return false;
  }

  auto * vCtx = (common_reasoning_budget_ctx *) pSmpl->vCtx;

  // only a sampler that is actively counting down the budget may be forced;
  // any other state (idle, already forcing/waiting, or done) is left untouched
  if (vCtx->vState != cReasoningBudgetCounting) {
    return false;
  }

  vCtx->vState    = cReasoningBudgetForcing;
  vCtx->vForcePos = 0;
  vCtx->vEndMatcher.reset();
  COM_TRC("%s", "forced into forcing state (manual transition)\n");

  return true;
}
