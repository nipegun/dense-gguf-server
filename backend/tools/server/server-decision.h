#pragma once

#include "server-common.h"
#include "server-task.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API)
// the model answers each question in one forward pass, no token is generated

enum server_decision_question_type {
  cServerDecisionQuestionChoice,
  cServerDecisionQuestionScore,
  cServerDecisionQuestionNoul,
};

struct server_decision_option {
  std::string vKey;
  json        vDescription;  // null if not provided
};

struct server_decision_question {
  std::string                         vId;
  server_decision_question_type       type;
  json                                vInstructions;
  std::vector<server_decision_option> lOptions;  // in the order of the model outputs
};

struct server_decision_context {
  common_decision_type type = cCommonDecisionTypeNone;

  // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
  void init(const dgs_model * pModel);

  // true if the questions of a request start with the same tokens, and the model can continue from them
  bool mCanSharePrompt() const {
    switch (type) {
      case cCommonDecisionTypeOpenjev:
      case cCommonDecisionTypeLev:
      case cCommonDecisionTypeKev:
      case cCommonDecisionTypeNimble:
      case cCommonDecisionTypePplxDecider:
      case cCommonDecisionTypeVllmDecision2:
      case cCommonDecisionTypeLfm2D1:
        return true;
      default:
        return false;
    }
  }

  // true if all the questions of a request go in one prompt, see fill_task_joint()
  bool mIsJoint() const { return type == cCommonDecisionTypeClef; }

  // true if the prompt of the model has a place for images
  bool mCanUseImages() const {
    switch (type) {
      case cCommonDecisionTypeOpenjev:
      case cCommonDecisionTypeClef:
      case cCommonDecisionTypePplxDecider:
      case cCommonDecisionTypeVllmD3:
      case cCommonDecisionTypeLfm2D1:
      case cCommonDecisionTypeLfm2D1Omni:
        return true;
      default:
        return false;
    }
  }

  // throw std::invalid_argument on bad input
  std::vector<server_decision_question> mParseQuestions(const json & pBody) const;

  // returns the state without its images, they are appended to files in order
  // images come from "files" (alias "images") and from the image_url and input_audio parts of a state made of chat messages
  // an image can be an audio clip if the model supports it
  json mParseState(const json & pBody, std::vector<raw_buffer> & pFiles) const;

  // number of prompts that are evaluated to answer this question, each one shows the options in a different order
  size_t mNVariants(const server_decision_question & pQuestion) const;

  // set the prompt of one variant of this question, and where to read its result
  // mctx is only used if there are files
  void mFillTask(const json &                                  pState,
                 const std::vector<server_decision_question> & pQuestions,
                 const server_decision_question &              pQuestion,
                 size_t                                        pVariant,
                 const std::vector<raw_buffer> &               pFiles,
                 mtmd_context *                                pMctx,
                 const mtmd_helper_init_opt &                  pInitOpt,
                 server_task &                                 pTask) const;

  // set the prompt of all the questions, the result has the scores of all their options, in order
  // mctx is only used if there are files
  void mFillTaskJoint(const json &                                  pState,
                      const std::vector<server_decision_question> & pQuestions,
                      const std::vector<raw_buffer> &               pFiles,
                      mtmd_context *                                pMctx,
                      const mtmd_helper_init_opt &                  pInitOpt,
                      server_task &                                 pTask) const;

  // scores: the raw model outputs of each variant
  json mFormatAnswer(const server_decision_question &        pQuestion,
                     const std::vector<std::vector<float>> & pScores,
                     bool                                    pHasMedia = false) const;

private:
  const dgs_vocab *                           vVocab = nullptr;
  std::shared_ptr<const common_chat_template> vTmpl;   // the "systemone" template

  std::map<std::string, float>         dTemperatures;  // "<type>" or "<type>.<n_options bucket>"
  std::map<size_t, std::vector<float>> dScoreBias;     // added to the scores of a score question, by number of levels
  size_t                               vNOptionsMax = 0;
  bool vNoulTrueFirst                               = false;  // noul options are [true, false] instead of [false, true]
  bool vNoulCriteriaOrder = false;  // noul options are in the order of the criteria keys, if given
  bool vChoiceSorted      = false;  // choice options are in the order of their keys
  bool vStrictPayload     = false;  // state, instructions and descriptions must be text, an object or an array
  bool vCanonicalLegend   = false;  // non-string score descriptions are returned as canonical JSON text

  // OPENJEV, LEV, NIMBLE, PPLX_DECIDER, VLLM_D3
  std::vector<dgs_token>   lLabels;
  std::vector<std::string> lLabelTexts;  // only if the label of an option is given to the template

  // LAYA, KEV
  dgs_token   vTokenMarker = DGS_TOKEN_NULL;
  dgs_token   vTokenSep    = DGS_TOKEN_NULL;
  std::string vTextMarker;
  size_t      vMaxHeadTokens   = 0;  // question + options
  size_t      vMaxOptionTokens = 48;

  std::string mRender(const json &                                  pState,
                      const std::vector<server_decision_question> & pQuestions,
                      const server_decision_question &              pQuestion,
                      size_t                                        pVariant,
                      size_t                                        pNImages,
                      bool                                          pIsAudio = false) const;
  json        mRenderOptions(const server_decision_question & pQuestion, size_t pVariant) const;
  size_t      mNOutputs(const server_decision_question & pQuestion) const;
  // LFM2_D1: label text and tokens of each option
  void        mD1Labels(const server_decision_question & pQuestion,
                        std::vector<std::string> &       pTexts,
                        std::vector<dgs_tokens> &        pGroups) const;
  void mFillTaskLaya(dgs_tokens & pTokens, const server_decision_question & pQuestion, server_task & pTask) const;
  void mFillTaskD1omni(const json &                                  pState,
                       const std::vector<server_decision_question> & pQuestions,
                       const server_decision_question &              pQuestion,
                       size_t                                        pVariant,
                       const std::vector<raw_buffer> &               pFiles,
                       mtmd_context *                                pMctx,
                       const mtmd_helper_init_opt &                  pInitOpt,
                       server_task &                                 pTask) const;

  float  mGetTemperature(const server_decision_question & pQuestion) const;
  double mGetDecisionConfidence(server_decision_question_type type, const std::vector<double> & pProbs) const;
};

// group the tasks so that the common prefix of their prompts is evaluated only once
// each group is one parent and its children, it takes at most n_slots slots
// note: the order of the tasks is preserved
std::vector<server_task> fServerDecisionGroupTasks(std::vector<server_task> && pTasks, size_t pNSlots);
