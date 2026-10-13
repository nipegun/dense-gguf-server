#pragma once

#include "clip.h"
#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

// Internal header for clip.cpp

#define MTMD_INTERNAL_HEADER

#define KEY_FTYPE             "general.file_type"
#define KEY_NAME              "general.name"
#define KEY_DESCRIPTION       "general.description"
#define KEY_PROJ_TYPE         "clip.projector_type"
#define KEY_HAS_AUDIO_ENC     "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC    "clip.has_vision_encoder"
#define KEY_HAS_GEN_AUDIO_ENC "clip.has_gen_audio_encoder"
#define KEY_USE_GELU          "clip.use_gelu"
#define KEY_USE_SILU          "clip.use_silu"

#define KEY_N_EMBD         "clip.%s.embedding_length"
#define KEY_N_FF           "clip.%s.feed_forward_length"
#define KEY_N_BLOCK        "clip.%s.block_count"
#define KEY_PROJ_DIM       "clip.%s.projection_dim"
#define KEY_N_HEAD         "clip.%s.attention.head_count"
#define KEY_N_HEAD_KV      "clip.%s.attention.head_count_kv"
#define KEY_N_EMBD_HEAD    "clip.%s.attention.head_dim"
#define KEY_LAYER_NORM_EPS "clip.%s.attention.layer_norm_epsilon"
#define KEY_FEATURE_LAYERS "clip.%s.feature_layer"

// vision-specific
#define KEY_VISION_PROJ_TYPE        "clip.vision.projector_type"  // for models with mixed modalities
#define KEY_IMAGE_SIZE              "clip.vision.image_size"
#define KEY_IMAGE_MIN_PIXELS        "clip.vision.image_min_pixels"
#define KEY_IMAGE_MAX_PIXELS        "clip.vision.image_max_pixels"
#define KEY_PREPROC_MIN_TILES       "clip.vision.preproc_min_tiles"
#define KEY_PREPROC_MAX_TILES       "clip.vision.preproc_max_tiles"
#define KEY_PREPROC_IMAGE_SIZE      "clip.vision.preproc_image_size"
#define KEY_PATCH_SIZE              "clip.vision.patch_size"
#define KEY_IMAGE_MEAN              "clip.vision.image_mean"
#define KEY_IMAGE_STD               "clip.vision.image_std"
#define KEY_IMAGE_RESIZE_ALGO       "clip.vision.image_resize_algo"
#define KEY_PROJ_SCALE_FACTOR       "clip.vision.projector.scale_factor"
#define KEY_PROJ_SAMPLE_QUERY_SIDE  "clip.vision.projector.query_side"
#define KEY_PROJ_SAMPLE_WINDOW_SIDE "clip.vision.projector.window_side"
#define KEY_PROJ_SPATIAL_OFFSETS    "clip.vision.projector.spatial_offsets"
#define KEY_SPATIAL_MERGE_SIZE      "clip.vision.spatial_merge_size"
#define KEY_SWIGLU_CLAMP            "clip.vision.swiglu_clamp"

#define KEY_MM_PATCH_MERGE_TYPE    "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS   "clip.vision.image_grid_pinpoints"
#define KEY_MAX_SLICE_NUMS         "clip.vision.max_slice_nums"
#define KEY_WIN_ATTN_PATTERN       "clip.vision.n_wa_pattern"
#define KEY_WIN_ATTN_LAYER_INDEXES "clip.vision.wa_layer_indexes"
#define KEY_WA_PATTERN_MODE        "clip.vision.wa_pattern_mode"
#define KEY_ATTN_WINDOW_SIZE       "clip.vision.window_size"
#define KEY_MINICPMV_VERSION       "clip.minicpmv_version"
#define KEY_MINICPMV_QUERY_NUM     "clip.minicpmv_query_num"
#define KEY_SAM_N_HEAD             "clip.vision.sam.head_count"
#define KEY_SAM_N_BLOCK            "clip.vision.sam.block_count"
#define KEY_SAM_N_EMBD             "clip.vision.sam.embedding_length"
#define KEY_VISION_N_EXPERT_USED   "clip.vision.expert_used_count"
// audio-specific
#define KEY_AUDIO_PROJ_TYPE        "clip.audio.projector_type"  // for models with mixed modalities
#define KEY_A_NUM_MEL_BINS         "clip.audio.num_mel_bins"
#define KEY_A_PROJ_STACK_FACTOR    "clip.audio.projector.stack_factor"
#define KEY_A_CHUNK_SIZE           "clip.audio.chunk_size"
#define KEY_A_CONV_KERNEL_SIZE     "clip.audio.conv_kernel_size"
#define KEY_A_MAX_POS_EMB          "clip.audio.max_pos_emb"
#define KEY_A_PROJ_WINDOW_SIZE     "clip.audio.projector.window_size"
#define KEY_A_PROJ_DOWNSAMPLE_RATE "clip.audio.projector.downsample_rate"
#define KEY_A_PROJ_HEAD_COUNT      "clip.audio.projector.head_count"
#define KEY_A_RVQ_NUM_QUANTIZERS   "clip.audio.rvq.num_quantizers"  // mimo-audio-tokenizer
#define KEY_A_RVQ_CODEBOOK_SIZE    "clip.audio.rvq.codebook_size"   // mimo-audio-tokenizer: per-quantizer bin count
#define KEY_A_WA_PATTERN_MODE  "clip.audio.wa_pattern_mode"  // mimo-audio-tokenizer, per-layer -1 (full) / 0 (windowed)
#define KEY_A_ATTN_WINDOW_SIZE "clip.audio.window_size"      // mimo-audio-tokenizer: sliding-window radius
#define KEY_A_LOCAL_BLOCK_COUNT  "clip.audio.local_block_count"  // mimo-v2.5: input_local_transformer layer count
#define KEY_A_LOCAL_GROUP_SIZE   "clip.audio.local_group_size"   // mimo-v2.5: input_local_transformer grouping size
// audio generation (gen-audio)-specific
#define KEY_GEN_AUDIO_PROJ_TYPE  "clip.gen.audio.projector_type"  // for models with mixed modalities
// name of the weight variant, for settings that are not in the checkpoint
#define KEY_GEN_AUDIO_VARIANT    "clip.gen.audio.model_variant"
#define KEY_AUDIO_SUBSMPL_FACTOR "clip.audio.subsampling_factor"

//
// tensor name constants
//

#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_CLASS_EMBD      "v.class_embd"
#define TN_PATCH_EMBD      "v.patch_embd.weight"  // current GGUF patch embedding name
#define TN_PATCH_EMBD_1    "v.patch_embd.weight.1"
#define TN_PATCH_BIAS      "v.patch_embd.bias"
#define TN_NORM_EMBD       "v.norm_embd.%s"
#define TN_PATCH_NORM      "v.patch_norm.%d.%s"
#define TN_ATTN_QKV        "%s.blk.%d.attn_qkv.%s"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_ATTN_SINKS      "%s.blk.%d.attn_sinks"
#define TN_ATTN_K_NORM     "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM     "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_FFN_GATE_INP    "%s.blk.%d.ffn_gate_inp.%s"  // MoE router (dots3note)
#define TN_FFN_GATE_EXPS   "%s.blk.%d.ffn_gate_exps.%s"
#define TN_FFN_UP_EXPS     "%s.blk.%d.ffn_up_exps.%s"
#define TN_FFN_DOWN_EXPS   "%s.blk.%d.ffn_down_exps.%s"
#define TN_FFN_EXP_PROBS_B "%s.blk.%d.exp_probs_b.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s"             // layer norm
#define TN_LN_2            "%s.blk.%d.ln2.%s"             // layer norm
#define TN_LS_1            "%s.blk.%d.ls1.%s"             // layer scale
#define TN_LS_2            "%s.blk.%d.ls2.%s"             // layer scale
#define TN_LS_OUT          "%s.blk.%d.out_scale.%s"       // layer out scale (gemma4)
#define TN_ATTN_POST_NORM  "%s.blk.%d.attn_post_norm.%s"  // post-attn norm (gemma4)
#define TN_FFN_POST_NORM   "%s.blk.%d.ffn_post_norm.%s"   // post-FFN norm (gemma4)
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_LLAVA_PROJ      "mm.%d.%s"
#define TN_MM_UP           "mm.up.%s"
#define TN_MM_GATE         "mm.gate.%s"
#define TN_MM_DOWN         "mm.down.%s"
#define TN_MM_POST_NORM    "mm.post_norm.%s"
#define TN_MVLM_PROJ_MLP   "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG   "mm.model.peg.%d.%s"
#define TN_IMAGE_NEWLINE   "v.image_newline"
#define TN_IMAGE_SEPERATOR "v.view_seperator"
#define TN_TOK_EMBD_SEP    "v.tok_embd_sep"
#define TN_MM_INP_NORM     "mm.input_norm.weight"
#define TN_MM_INP_NORM_B   "mm.input_norm.bias"
#define TN_MM_INP_PROJ     "mm.input_projection.weight"  // gemma3
#define TN_MM_SOFT_EMB_N   "mm.soft_emb_norm.weight"     // gemma3
#define TN_MM_PROJECTOR    "mm.model.fc.%s"              // idefics3, deepseekocr
#define TN_MM_PATCH_MERGER "mm.patch_merger.%s"          // mistral small 3.1, glm4v
#define TN_MM_MERGER_FC1   "mm.merger.fc1.%s"            // minimax-m3 patch-merge MLP
#define TN_MM_MERGER_FC2   "mm.merger.fc2.%s"
#define TN_TOK_IMG_BREAK   "v.token_embd.img_break"      // pixtral
#define TN_TOK_IMG_START   "v.token_embd.img_start"      // deepseek4v
#define TN_TOK_IMG_END     "v.token_embd.img_end"        // deepseek4v
#define TN_TOK_IMG_PAD     "v.token_embd.img_pad"        // deepseek4v
#define TN_TOK_GLM_BOI     "adapter.boi"                 // glm-edge (these embeddings are not in text model)
#define TN_TOK_GLM_EOI     "adapter.eoi"                 // glm-edge (these embeddings are not in text model)
#define TN_DEEPSTACK_NORM  "v.deepstack.%d.norm.%s"      // qwen3vl deepstack
#define TN_DEEPSTACK_FC1   "v.deepstack.%d.fc1.%s"       // qwen3vl deepstack
#define TN_DEEPSTACK_FC2   "v.deepstack.%d.fc2.%s"       // qwen3vl deepstack

// mimicpmv
#define TN_MINICPMV_POS_EMBD_K "resampler.pos_embed_k"
#define TN_MINICPMV_QUERY      "resampler.query"
#define TN_MINICPMV_PROJ       "resampler.proj.weight"
#define TN_MINICPMV_KV_PROJ    "resampler.kv.weight"
#define TN_MINICPMV_ATTN       "resampler.attn.%s.%s"
#define TN_MINICPMV_LN         "resampler.ln_%s.%s"

// MiniCPM-V 4.6 ViT merger (window attention + MLP downsample),
// matching the upstream `vit_merger` module name in transformers.
#define TN_VIT_MERGER_LN1     "v.vit_merger.ln1.%s"
#define TN_VIT_MERGER_ATTN_Q  "v.vit_merger.attn_q.%s"
#define TN_VIT_MERGER_ATTN_K  "v.vit_merger.attn_k.%s"
#define TN_VIT_MERGER_ATTN_V  "v.vit_merger.attn_v.%s"
#define TN_VIT_MERGER_ATTN_O  "v.vit_merger.attn_out.%s"
#define TN_VIT_MERGER_DS_LN   "v.vit_merger.ds_ln.%s"
#define TN_VIT_MERGER_DS_UP   "v.vit_merger.ds_ffn_up.%s"
#define TN_VIT_MERGER_DS_DOWN "v.vit_merger.ds_ffn_down.%s"

#define TN_GLM_ADAPER_CONV      "adapter.conv.%s"
#define TN_GLM_ADAPTER_LINEAR   "adapter.linear.linear.%s"
#define TN_GLM_ADAPTER_NORM_1   "adapter.linear.norm1.%s"
#define TN_GLM_ADAPTER_D_H_2_4H "adapter.linear.dense_h_to_4h.%s"
#define TN_GLM_ADAPTER_GATE     "adapter.linear.gate.%s"
#define TN_GLM_ADAPTER_D_4H_2_H "adapter.linear.dense_4h_to_h.%s"

// ultravox
#define TN_CONV1D       "a.conv1d.%d.%s"
#define TN_CONV2D       "a.conv2d.%d.%s"
#define TN_CONV_OUT     "a.conv_out.%s"
#define TN_MM_AUDIO_MLP "mm.a.mlp.%d.%s"
#define TN_MM_AUDIO_FC  "mm.a.fc.%s"  // fully connected layer
#define TN_MM_NORM_PRE  "mm.a.norm_pre.%s"
#define TN_MM_NORM_MID  "mm.a.norm_mid.%s"

// mimo-audio-tokenizer
#define TN_A_DOWNSAMPLE_CONV   "a.downsample.conv.%s"
#define TN_A_DOWNSAMPLE_NORM   "a.downsample.norm.%s"
#define TN_A_RVQ_CODEBOOK      "a.rvq.codebook.%s"
// mimo-v2.5: text-side RVQ code embedding ("text codebook")
#define TN_MM_A_CODE_EMBD      "mm.a.code_embd.%s"
// mimo-v2.5: LLM-side connector (input_local_transformer)
#define TN_MM_A_LOCAL_ATTN_Q   "mm.a.local_blk.%d.attn_q.%s"
#define TN_MM_A_LOCAL_ATTN_K   "mm.a.local_blk.%d.attn_k.%s"
#define TN_MM_A_LOCAL_ATTN_V   "mm.a.local_blk.%d.attn_v.%s"
#define TN_MM_A_LOCAL_ATTN_OUT "mm.a.local_blk.%d.attn_out.%s"
#define TN_MM_A_LOCAL_FFN_GATE "mm.a.local_blk.%d.ffn_gate.%s"
#define TN_MM_A_LOCAL_FFN_UP   "mm.a.local_blk.%d.ffn_up.%s"
#define TN_MM_A_LOCAL_FFN_DOWN "mm.a.local_blk.%d.ffn_down.%s"
#define TN_MM_A_LOCAL_LN1      "mm.a.local_blk.%d.ln1.%s"
#define TN_MM_A_LOCAL_LN2      "mm.a.local_blk.%d.ln2.%s"
#define TN_MM_A_LOCAL_NORM     "mm.a.local_norm.%s"

// qwen3tts speaker encoder (ECAPA-TDNN)
#define TN_A_SE_CONV1  "a.blk.%d.se_conv1.%s"
#define TN_A_SE_CONV2  "a.blk.%d.se_conv2.%s"
#define TN_A_CONV_RES2 "a.blk.%d.res2.%d.%s"
#define TN_A_ASP_ATTN  "a.asp_attn.%s"
#define TN_A_ASP_TDNN  "a.asp_tdnn.%s"

// qwen3tts code_predictor
#define TN_A_GEN_CODE_PROJ_IN  "a.gen.code.proj_in.%s"
#define TN_A_GEN_CODE_EMBD     "a.gen.code.embd.%s"
#define TN_A_GEN_CODE_HEAD     "a.gen.code.head.%s"
#define TN_A_GEN_CODE_OUT_EMBD "a.gen.code.out_embd.%s"
#define TN_A_GEN_CODE_NORM     "a.gen.code.output_norm.%s"

// qwen3tts code2wav (RVQ codes -> raw PCM)
// pre_transformer layers use the generic TN_ATTN_*/TN_FFN_*/TN_LN_*/TN_LS_* macros, prefix "a.gen.wav.tfm"
#define TN_A_GEN_WAV_QUANT_FIRST_IN  "a.gen.wav.quant.first.in_proj.%s"
#define TN_A_GEN_WAV_QUANT_FIRST_OUT "a.gen.wav.quant.first.out_proj.%s"
#define TN_A_GEN_WAV_QUANT_FIRST_CB  "a.gen.wav.quant.first.codebook.%s"
#define TN_A_GEN_WAV_QUANT_REST_IN   "a.gen.wav.quant.rest.in_proj.%s"
#define TN_A_GEN_WAV_QUANT_REST_OUT  "a.gen.wav.quant.rest.out_proj.%s"
#define TN_A_GEN_WAV_QUANT_REST_CB   "a.gen.wav.quant.rest.codebook.%s"
#define TN_A_GEN_WAV_PRE_CONV        "a.gen.wav.pre_conv.%s"
#define TN_A_GEN_WAV_TFM_IN_PROJ     "a.gen.wav.tfm.in_proj.%s"
#define TN_A_GEN_WAV_TFM_OUT_PROJ    "a.gen.wav.tfm.out_proj.%s"
#define TN_A_GEN_WAV_TFM_OUT_NORM    "a.gen.wav.tfm.output_norm.%s"
#define TN_A_GEN_WAV_UP_CONV         "a.gen.wav.up.blk.%d.conv.%s"
#define TN_A_GEN_WAV_UP_DWCONV       "a.gen.wav.up.blk.%d.dwconv.%s"
#define TN_A_GEN_WAV_UP_NORM         "a.gen.wav.up.blk.%d.norm.%s"
#define TN_A_GEN_WAV_UP_PW1          "a.gen.wav.up.blk.%d.pw1.%s"
#define TN_A_GEN_WAV_UP_PW2          "a.gen.wav.up.blk.%d.pw2.%s"
#define TN_A_GEN_WAV_UP_GAMMA        "a.gen.wav.up.blk.%d.gamma"
#define TN_A_GEN_WAV_DAC_ENTRY       "a.gen.wav.dac.entry.%s"
#define TN_A_GEN_WAV_DAC_SNAKE       "a.gen.wav.dac.blk.%d.snake.%s"
#define TN_A_GEN_WAV_DAC_CONV        "a.gen.wav.dac.blk.%d.conv.%s"
#define TN_A_GEN_WAV_DAC_RES_ACT1    "a.gen.wav.dac.blk.%d.res.%d.act1.%s"
#define TN_A_GEN_WAV_DAC_RES_CONV1   "a.gen.wav.dac.blk.%d.res.%d.conv1.%s"
#define TN_A_GEN_WAV_DAC_RES_ACT2    "a.gen.wav.dac.blk.%d.res.%d.act2.%s"
#define TN_A_GEN_WAV_DAC_RES_CONV2   "a.gen.wav.dac.blk.%d.res.%d.conv2.%s"
#define TN_A_GEN_WAV_DAC_POST_SNAKE  "a.gen.wav.dac.post_snake.%s"
#define TN_A_GEN_WAV_DAC_POST_CONV   "a.gen.wav.dac.post_conv.%s"

// pocket-tts
#define TN_A_SEANET_CONV_IN            "a.seanet.conv_in.%s"
#define TN_A_SEANET_CONV_OUT           "a.seanet.conv_out.%s"
#define TN_A_SEANET_RES_CONV1          "a.seanet.blk.%d.res_conv1.%s"
#define TN_A_SEANET_RES_CONV2          "a.seanet.blk.%d.res_conv2.%s"
#define TN_A_SEANET_SCALE_CONV         "a.seanet.blk.%d.scale_conv.%s"
#define TN_A_SPEAKER_PROJ              "a.speaker_proj.%s"
#define TN_A_DOWNSAMPLE_CONV           "a.downsample.conv.%s"
#define TN_A_GEN_FLOW_INPUT_PROJ       "a.gen.flow.input_proj.%s"
#define TN_A_GEN_FLOW_COND_EMBD        "a.gen.flow.cond_embd.%s"
#define TN_A_GEN_FLOW_TIME_FREQS       "a.gen.flow.time.%d.freqs"
#define TN_A_GEN_FLOW_TIME_UP          "a.gen.flow.time.%d.up.%s"
#define TN_A_GEN_FLOW_TIME_DOWN        "a.gen.flow.time.%d.down.%s"
#define TN_A_GEN_FLOW_TIME_NORM        "a.gen.flow.time.%d.norm"
#define TN_A_GEN_FLOW_BLK_NORM         "a.gen.flow.blk.%d.norm.%s"
#define TN_A_GEN_FLOW_BLK_UP           "a.gen.flow.blk.%d.up.%s"
#define TN_A_GEN_FLOW_BLK_DOWN         "a.gen.flow.blk.%d.down.%s"
#define TN_A_GEN_FLOW_BLK_ADA          "a.gen.flow.blk.%d.ada.%s"
#define TN_A_GEN_FLOW_FINAL_ADA        "a.gen.flow.final.ada.%s"
#define TN_A_GEN_FLOW_FINAL_PROJ       "a.gen.flow.final.proj.%s"
#define TN_A_GEN_OUT_EOS               "a.gen.out_eos.%s"
#define TN_A_GEN_INPUT_LINEAR          "a.gen.input_linear.%s"
#define TN_A_GEN_EMB_MEAN              "a.gen.emb_mean"
#define TN_A_GEN_EMB_STD               "a.gen.emb_std"
#define TN_A_GEN_WAV_QUANT_OUT         "a.gen.wav.quant_out.%s"
#define TN_A_GEN_WAV_UPSAMPLE          "a.gen.wav.upsample.%s"
#define TN_A_GEN_WAV_SEANET_CONV_IN    "a.gen.wav.seanet.conv_in.%s"
#define TN_A_GEN_WAV_SEANET_CONV_OUT   "a.gen.wav.seanet.conv_out.%s"
#define TN_A_GEN_WAV_SEANET_RES_CONV1  "a.gen.wav.seanet.blk.%d.res_conv1.%s"
#define TN_A_GEN_WAV_SEANET_RES_CONV2  "a.gen.wav.seanet.blk.%d.res_conv2.%s"
#define TN_A_GEN_WAV_SEANET_SCALE_CONV "a.gen.wav.seanet.blk.%d.scale_conv.%s"

// cogvlm
#define TN_MM_POST_FC_NORM "mm.post_fc_norm.%s"
#define TN_MM_H_TO_4H      "mm.up.%s"
#define TN_MM_GATE         "mm.gate.%s"
#define TN_MM_4H_TO_H      "mm.down.%s"
#define TN_TOK_BOI         "v.boi"
#define TN_TOK_EOI         "v.eoi"

// hunyuanvl (shared GGUF tensor names)
#define TN_MM_PRE_NORM  "mm.pre_norm.%s"
#define TN_MM_IMG_BEGIN "mm.image_begin"  // GGUF image boundary tensor name
#define TN_MM_IMG_END   "mm.image_end"    // GGUF image boundary tensor name

// deepseek-ocr
#define TN_SAM_POS_EMBD         "v.sam.pos_embd.%s"
#define TN_SAM_PATCH_EMBD       "v.sam.patch_embd.%s"
#define TN_SAM_PRE_NORM         "v.sam.blk.%d.pre_ln.%s"
#define TN_SAM_POST_NORM        "v.sam.blk.%d.post_ln.%s"
#define TN_SAM_ATTN_POS_H       "v.sam.blk.%d.attn.pos_h.%s"
#define TN_SAM_ATTN_POS_W       "v.sam.blk.%d.attn.pos_w.%s"
#define TN_SAM_ATTN_QKV         "v.sam.blk.%d.attn.qkv.%s"
#define TN_SAM_ATTN_OUT         "v.sam.blk.%d.attn.out.%s"
#define TN_SAM_FFN_UP           "v.sam.blk.%d.mlp.lin1.%s"
#define TN_SAM_FFN_DOWN         "v.sam.blk.%d.mlp.lin2.%s"
#define TN_SAM_NECK             "v.sam.neck.%d.%s"
#define TN_SAM_NET              "v.sam.net_%d.%s"
// deepseek-ocr-2
#define TN_RESMPL_QUERY         "v.resample_query_%d.%s"
// (conformer) lfm2
#define TN_PRE_ENCODE_OUT       "a.pre_encode.out.%s"
#define TN_FFN_NORM             "%s.blk.%d.ffn_norm.%s"
#define TN_FFN_NORM_1           "%s.blk.%d.ffn_norm_1.%s"
#define TN_FFN_UP_1             "%s.blk.%d.ffn_up_1.%s"
#define TN_FFN_DOWN_1           "%s.blk.%d.ffn_down_1.%s"
#define TN_POS_BIAS_U           "%s.blk.%d.pos_bias_u"
#define TN_POS_BIAS_V           "%s.blk.%d.pos_bias_v"
#define TN_NORM_CONV            "%s.blk.%d.norm_conv.%s"
#define TN_LINEAR_POS           "%s.blk.%d.linear_pos.%s"
#define TN_CONV_DW              "%s.blk.%d.conv_dw.%s"
#define TN_CONV_NORM            "%s.blk.%d.conv_norm.%s"
#define TN_CONV_PW1             "%s.blk.%d.conv_pw1.%s"
#define TN_CONV_PW2             "%s.blk.%d.conv_pw2.%s"
#define TN_INP_PROJ             "a.input_projection.%s"
#define TN_CTC_OUT              "a.enc_ctc_out.%s"
#define TN_CTC_OUT_MID          "a.enc_ctc_out_mid.%s"
#define TN_ATTN_REL_POS_EMB     "%s.blk.%d.attn_rel_pos_emb"
// qformer projector
#define TN_QF_PROJ_QUERY        "%s.proj_query"
#define TN_QF_PROJ_NORM         "%s.proj_norm.%s"
#define TN_QF_PROJ_LINEAR       "%s.proj_linear.%s"
#define TN_QF_SELF_ATTN_Q       "%s.proj_blk.%d.self_attn_q.%s"
#define TN_QF_SELF_ATTN_K       "%s.proj_blk.%d.self_attn_k.%s"
#define TN_QF_SELF_ATTN_V       "%s.proj_blk.%d.self_attn_v.%s"
#define TN_QF_SELF_ATTN_O       "%s.proj_blk.%d.self_attn_out.%s"
#define TN_QF_SELF_ATTN_N       "%s.proj_blk.%d.self_attn_norm.%s"
#define TN_QF_CROSS_ATTN_Q      "%s.proj_blk.%d.cross_attn_q.%s"
#define TN_QF_CROSS_ATTN_K      "%s.proj_blk.%d.cross_attn_k.%s"
#define TN_QF_CROSS_ATTN_V      "%s.proj_blk.%d.cross_attn_v.%s"
#define TN_QF_CROSS_ATTN_O      "%s.proj_blk.%d.cross_attn_out.%s"
#define TN_QF_CROSS_ATTN_N      "%s.proj_blk.%d.cross_attn_norm.%s"
#define TN_QF_FFN_UP            "%s.proj_blk.%d.ffn_up.%s"
#define TN_QF_FFN_DOWN          "%s.proj_blk.%d.ffn_down.%s"
#define TN_QF_FFN_NORM          "%s.proj_blk.%d.ffn_norm.%s"
// multi-projector qformer (bid => projector ID)
#define TN_MULTI_PROJ_IMG_POS   "v.proj_blk.%d.img_pos"
#define TN_MULTI_PROJ_QUERY     "%s.proj_blk.%d.query"
#define TN_MULTI_PROJ_LINEAR    "%s.proj_blk.%d.linear.%s"
#define TN_MULTI_PROJ_NORM      "%s.proj_blk.%d.norm.%s"
#define TN_MULTI_PROJ_POST_NORM "%s.proj_blk.%d.post_norm.%s"

// gemma4 audio conformer
#define TN_A_MM_INP_PROJ     "mm.a.input_projection.%s"
#define TN_A_MM_SOFT_EMB_N   "mm.a.soft_emb_norm.%s"
#define TN_A_INP_PROJ        "a.input_projection.%s"
#define TN_A_CONV1D          "a.conv1d.%d.%s"
#define TN_A_CONV1D_NORM     "a.conv1d.%d.norm.%s"
#define TN_A_OUT_PROJ        "a.pre_encode.out.%s"
#define TN_A_ATTN_PRE_NORM   "%s.blk.%d.attn_pre_norm.%s"
#define TN_A_ATTN_POST_NORM  "%s.blk.%d.attn_post_norm.%s"
#define TN_A_ATTN_K_REL      "%s.blk.%d.attn_k_rel.%s"
#define TN_A_PER_DIM_SCALE   "%s.blk.%d.per_dim_scale.%s"
#define TN_A_PER_DIM_K_SCALE "%s.blk.%d.per_dim_k_scale.%s"
#define TN_A_FFN_POST_NORM   "%s.blk.%d.ffn_post_norm.%s"
#define TN_A_FFN_POST_NORM_1 "%s.blk.%d.ffn_post_norm_1.%s"

// mobilenetv5 (gemma3n) definitions
#define TN_MNV5_STEM_CONV "v.conv_stem.conv.weight"
#define TN_MNV5_STEM_BIAS "v.conv_stem.conv.bias"
#define TN_MNV5_STEM_BN   "v.conv_stem.bn.weight"

// Stage 0 Block (Edge Residual)
#define TN_MNV5_BLK_S0_EXP_W "v.blk.%d.%d.conv_exp.weight"
#define TN_MNV5_BLK_S0_BN1_W "v.blk.%d.%d.bn1.weight"
#define TN_MNV5_BLK_S0_PWL_W "v.blk.%d.%d.conv_pwl.weight"
#define TN_MNV5_BLK_S0_BN2_W "v.blk.%d.%d.bn2.weight"

// Stage 1+ Block (Universal Inverted Residual)
#define TN_MNV5_BLK_DW_START_W  "v.blk.%d.%d.dw_start.conv.weight"
#define TN_MNV5_BLK_DW_START_BN "v.blk.%d.%d.dw_start.bn.weight"
#define TN_MNV5_BLK_DW_MID_W    "v.blk.%d.%d.dw_mid.conv.weight"
#define TN_MNV5_BLK_DW_MID_BN   "v.blk.%d.%d.dw_mid.bn.weight"
#define TN_MNV5_BLK_PW_EXP_W    "v.blk.%d.%d.pw_exp.conv.weight"
#define TN_MNV5_BLK_PW_EXP_BN   "v.blk.%d.%d.pw_exp.bn.weight"
#define TN_MNV5_BLK_PW_PROJ_W   "v.blk.%d.%d.pw_proj.conv.weight"
#define TN_MNV5_BLK_PW_PROJ_BN  "v.blk.%d.%d.pw_proj.bn.weight"
#define TN_MNV5_BLK_LAYER_SCALE "v.blk.%d.%d.layer_scale.gamma"

// Attention Components
#define TN_MNV5_ATTN_Q_W    "v.blk.%d.%d.attn.query.proj.weight"
#define TN_MNV5_ATTN_K_W    "v.blk.%d.%d.attn.key.proj.weight"
#define TN_MNV5_ATTN_V_W    "v.blk.%d.%d.attn.value.proj.weight"
#define TN_MNV5_ATTN_O_W    "v.blk.%d.%d.attn.output.proj.weight"
#define TN_MNV5_ATTN_K_DW   "v.blk.%d.%d.attn.key.down_conv.weight"
#define TN_MNV5_ATTN_K_NORM "v.blk.%d.%d.attn.key.norm.weight"
#define TN_MNV5_ATTN_V_DW   "v.blk.%d.%d.attn.value.down_conv.weight"
#define TN_MNV5_ATTN_V_NORM "v.blk.%d.%d.attn.value.norm.weight"
#define TN_MNV5_ATTN_NORM   "v.blk.%d.%d.norm.weight"  // Block norm used in attn blocks

// MSFA
#define TN_MNV5_MSFA_FFN_EXP_W   "v.msfa.ffn.pw_exp.conv.weight"
#define TN_MNV5_MSFA_FFN_EXP_BN  "v.msfa.ffn.pw_exp.bn.weight"
#define TN_MNV5_MSFA_FFN_PROJ_W  "v.msfa.ffn.pw_proj.conv.weight"
#define TN_MNV5_MSFA_FFN_PROJ_BN "v.msfa.ffn.pw_proj.bn.weight"
#define TN_MNV5_MSFA_NORM        "v.msfa.norm.weight"

// gemma4
#define TN_STD_BIAS  "v.std_bias"
#define TN_STD_SCALE "v.std_scale"

// yasa2
#define TN_YASA_PATCH_LN_W      "v.patch_ln.weight"
#define TN_YASA_PATCH_LN_B      "v.patch_ln.bias"
#define TN_YASA_BACKBONE_LN_W   "v.backbone_ln.weight"
#define TN_YASA_BACKBONE_LN_B   "v.backbone_ln.bias"
#define TN_YASA_POS_EMBD        "v.vision_pos_embed"
#define TN_YASA_STAGE_DOWN_LN   "v.stage.%d.down.ln.%s"
#define TN_YASA_STAGE_DOWN_CONV "v.stage.%d.down.conv.%s"
#define TN_YASA_STAGE_BLK       "v.stage.%d.blk.%d.%s.%s"

// parakeet
#define TN_MEL_FILTERS    "a.mel_filters"
#define TN_WINDOW         "a.window"
#define TN_CONV_NORM_MEAN "%s.blk.%d.conv_norm_mean"
#define TN_CONV_NORM_VAR  "%s.blk.%d.conv_norm_var"

// align x to upper multiple of n
#define CLIP_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

// forward declaration
// TODO: improve this later
struct clip_ctx;

enum projector_type {
  cProjectorTypeMlp,
  cProjectorTypeMlpNorm,
  cProjectorTypeLdp,
  cProjectorTypeLdpv2,
  cProjectorTypeMinicpmv,
  cProjectorTypeGlmEdge,
  cProjectorTypeQwen2vl,
  cProjectorTypeQwen3vl,
  cProjectorTypeLing3vl,
  cProjectorTypeStep3vl,
  cProjectorTypeGemma3,
  cProjectorTypeGemma3nv,
  cProjectorTypeGemma3na,
  cProjectorTypeGemma4v,
  cProjectorTypeGemma4a,
  cProjectorTypeGemma4uv,
  cProjectorTypeGemma4ua,
  cProjectorTypePhi4,
  cProjectorTypeIdefics3,
  cProjectorTypePixtral,
  cProjectorTypeQwen25vl,
  cProjectorTypeUltravox,
  cProjectorTypeInternvl,
  cProjectorTypeDgs4,
  cProjectorTypeQwen2a,
  cProjectorTypeQwen3a,
  cProjectorTypeGlma,
  cProjectorTypeQwen25o,  // will be replaced by QWEN2A or QWEN25VL depending on clip_ctx
  cProjectorTypeVoxtral,
  cProjectorTypeMeralion,
  cProjectorTypeMusicFlamingo,
  cProjectorTypeLfm2,
  cProjectorTypeD1omniV,
  cProjectorTypeKimivl,
  cProjectorTypePaddleocr,
  cProjectorTypeLightonocr,
  cProjectorTypeCogvlm,
  cProjectorTypeJanusPro,
  cProjectorTypeDotsOcr,
  cProjectorTypeDots3noteV,
  cProjectorTypeDots3noteA,
  cProjectorTypeDeepseekocr,
  cProjectorTypeDeepseekocr2,
  cProjectorTypeDeepseek4v,
  cProjectorTypeLfm2a,
  cProjectorTypeD1omniA,
  cProjectorTypeGlm4v,
  cProjectorTypeGlm5v,
  cProjectorTypeYoutuvl,
  cProjectorTypeYasa2,
  cProjectorTypeKimik25,
  cProjectorTypeNemotronV2Vl,
  cProjectorTypeHunyuanvl,
  cProjectorTypeParakeet,
  cProjectorTypeEXAONE4Point5,
  cProjectorTypeMINICPMV4Point6,
  cProjectorTypeMINICPMV4Point7,
  cProjectorTypeGraniteSpeech,
  cProjectorTypeMimovl,
  cProjectorTypeMinimaxM3,
  cProjectorTypeGranite4Vision,
  cProjectorTypeMimoAudio,
  cProjectorTypeQwen3ttsSpkenc,
  cProjectorTypeQwen3ttsGen,
  cProjectorTypePocketttsSpkenc,
  cProjectorTypePocketttsGen,
  cProjectorTypeMuseGlimmer,
  cProjectorTypeCohere2v,
  cProjectorTypeUnknown,
};

static std::map<projector_type, std::string> dProjectorTypeNames = {
  { cProjectorTypeQwen3vl, "qwen3vl_merger" },
};

static projector_type fClipProjectorTypeFromString(const std::string & str) {
  for (const auto & vPair : dProjectorTypeNames) {
    if (vPair.second == str) {
      return vPair.first;
    }
  }
  return cProjectorTypeUnknown;
}

// RGB uint8 image
struct clip_image_u8 {
  clip_image_size mGetSize() const { return { vNx, vNy }; }

  void mSetSize(clip_image_size size, bool pIsPlaceholder) {
    vNx = size.vWidth;
    vNy = size.vHeight;
    if (pIsPlaceholder) {
      lBuf.clear();
    } else {
      lBuf.resize((size_t) vNx * (size_t) vNy * 3);
    }
  }

  void mCpyBuf(const std::vector<uint8_t> & pNewBuf) { lBuf = pNewBuf; }

  const std::vector<uint8_t> & mGetRoBuf() const {
    if (mIsPlaceholder()) {
      throw std::runtime_error("this clip_image_u8 is a placeholder");
    }
    return lBuf;
  }

  // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern. always use get_pixel / set_pixel for buffer manipulation

  bool mIsPlaceholder() const { return lBuf.empty(); }

  std::array<uint8_t, 3> mGetPixel(int pX, int pY) const {
    if (mIsPlaceholder()) {
      throw std::runtime_error("Cannot read pixels from a placeholder image");
    }
    size_t vIdx = ((size_t) pY * (size_t) vNx + (size_t) pX) * 3;
    return { lBuf[vIdx], lBuf[vIdx + 1], lBuf[vIdx + 2] };
  }

  void mSetPixel(int pX, int pY, const std::array<uint8_t, 3> & pRgb) {
    if (mIsPlaceholder()) {
      return;  // no-op
    }
    size_t vIdx    = ((size_t) pY * (size_t) vNx + (size_t) pX) * 3;
    lBuf[vIdx]     = pRgb[0];
    lBuf[vIdx + 1] = pRgb[1];
    lBuf[vIdx + 2] = pRgb[2];
  }

  size_t mNElements() const { return mNPixels() * 3; }

private:
  std::vector<uint8_t> lBuf;
  int                  vNx = 0;
  int                  vNy = 0;

  size_t mNPixels() const { return (size_t) vNx * (size_t) vNy; }
};

struct mtmd_serialization;  // forward declaration

// separators appended after the image tokens of one entry, as rows of v.tok_embd_sep
enum clip_suffix_type : int32_t {
  cClipSuffixNone = 0,
  // MiniCPM-V 4.7 tiles
  cClipSuffixMinicpmvOv,       // </image>
  cClipSuffixMinicpmvOvSlice,  // </image><slice>
  cClipSuffixMinicpmvSlice,    // </slice><slice>
  cClipSuffixMinicpmvRowEnd,   // </slice>\n<slice>
  cClipSuffixMinicpmvLast,     // </slice>
  cClipSuffixCount,
};

// rows of v.tok_embd_sep for each suffix type
// MiniCPM-V 4.7 rows (set by the converter): 0 = </image>, 1 = <slice>, 2 = </slice>, 3 = \n
static inline const std::vector<int> & fClipSuffixRows(clip_suffix_type type) {
  static const std::vector<int> lNone;
  static const std::vector<int> lMinicpmvOv      = { 0 };
  static const std::vector<int> lMinicpmvOvSlice = { 0, 1 };
  static const std::vector<int> lMinicpmvSlice   = { 2, 1 };
  static const std::vector<int> lMinicpmvRowEnd  = { 2, 3, 1 };
  static const std::vector<int> lMinicpmvLast    = { 2 };
  switch (type) {
    case cClipSuffixNone:
      return lNone;
    case cClipSuffixMinicpmvOv:
      return lMinicpmvOv;
    case cClipSuffixMinicpmvOvSlice:
      return lMinicpmvOvSlice;
    case cClipSuffixMinicpmvSlice:
      return lMinicpmvSlice;
    case cClipSuffixMinicpmvRowEnd:
      return lMinicpmvRowEnd;
    case cClipSuffixMinicpmvLast:
      return lMinicpmvLast;
    default:
      GGML_ABORT("invalid suffix type");
  }
}

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For seq, buf.size() == nx*ny*3*nt
//     Memory layout: RGBRGB...RGBRGB... (nt times)
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct clip_image_f32 {
  // marks the global view in e.g., DeepSeek-OCR Models
  bool             vAddViewsep = false;
  // appends a learned newline (or EOI) token after the image
  // no model uses it now (Granite4 Vision moved to anyres), kept for future models
  bool             vAddNewline = false;
  // deepseek4v: number of leading IMAGE_PAD embeddings, aligns IMAGE_START to the LLM compressor ratio
  // depends on the chunk position, set at tokenize time (see mtmd_tokenizer::add_media)
  int32_t          vLeadPad    = 0;
  // separators appended after the image tokens
  clip_suffix_type vSuffixType = cClipSuffixNone;

  // tile grid of the image group this entry belongs to
  // llava-next "anyres" (Granite4 Vision): the whole grid is encoded and assembled in a single graph
  // MiniCPM-V 4.7: set on the overview entry, the decoder positions of all tiles are derived from it
  struct anyres_info {
    int vGridX  = 0;  // tiles per row, 0 means the image is not tiled
    int vGridY  = 0;  // tiles per column
    int vOrigNx = 0;  // size of the source image, used to drop the padding tokens
    int vOrigNy = 0;

    bool mIsTiled() const { return vGridX > 0 && vGridY > 0; }
  };

  anyres_info vAnyres;

  clip_image_size mGetSize() const { return { vNx, vNy }; }

  int mNx() const { return vNx; }

  int mNy() const { return vNy; }

  void mSetSize(clip_image_size size, bool pIsPlaceholder, bool pIsAudio) {
    vNx = size.vWidth;
    vNy = size.vHeight;
    if (pIsPlaceholder) {
      lBuf.clear();
    } else {
      if (pIsAudio) {
        lBuf.resize((size_t) vNx * (size_t) vNy);
      } else {
        lBuf.resize((size_t) vNx * (size_t) vNy * 3);
      }
    }
  }

  void mCpyBuf(const std::vector<float> & pNewBuf) { lBuf = pNewBuf; }

  void mFromU8(const clip_image_u8 & pImg) {
    auto size = pImg.mGetSize();
    vNx       = size.vWidth;
    vNy       = size.vHeight;
    if (pImg.mIsPlaceholder()) {
      lBuf.clear();
      return;  // no-op
    }
    lBuf.resize(pImg.mNElements());
    const auto & lU8Buf = pImg.mGetRoBuf();
    for (size_t vI = 0; vI < pImg.mNElements(); ++vI) {
      lBuf[vI] = (float) lU8Buf[vI] / 255.0f;
    }
  }

  size_t mNElements() const { return mNPixels() * 3; }

  void mNormalize(const float pMean[3], const float pStd[3]) {
    if (mIsPlaceholder()) {
      return;  // no-op
    }
    for (size_t vI = 0; vI < mNPixels(); ++vI) {
      lBuf[vI * 3 + 0] = (lBuf[vI * 3 + 0] - pMean[0]) / pStd[0];
      lBuf[vI * 3 + 1] = (lBuf[vI * 3 + 1] - pMean[1]) / pStd[1];
      lBuf[vI * 3 + 2] = (lBuf[vI * 3 + 2] - pMean[2]) / pStd[2];
    }
  }

  const std::vector<float> & mGetRoBuf() const {
    if (mIsPlaceholder()) {
      throw std::runtime_error("this clip_image_f32 is a placeholder");
    }
    return lBuf;
  }

  // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern

  bool mIsPlaceholder() const { return lBuf.empty(); }

  void serialize(struct mtmd_serialization & pSer) const;
  void deserialize(struct mtmd_serialization & pSer);

private:
  std::vector<float> lBuf;
  int                vNx = 0;
  int                vNy = 0;

  size_t mNPixels() const { return (size_t) vNx * (size_t) vNy; }
};

// token area kept after removing the padding added by the anyres resize
// ref: https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/llava_next/modeling_llava_next.py#L109
static inline void
fClipAnyresUnpad(int pCurW, int pCurH, int pOrigW, int pOrigH, int & pOffX, int & pOffY, int & pOutW, int & pOutH) {
  pOffX = 0;
  pOffY = 0;
  pOutW = pCurW;
  pOutH = pCurH;
  if ((float) pOrigW / pOrigH > (float) pCurW / pCurH) {
    const int cNewH = (int) std::floor((double) pOrigH * pCurW / pOrigW + 1e-7);
    pOffY           = (pCurH - cNewH) / 2;
    pOutH           = pCurH - 2 * pOffY;
  } else {
    const int cNewW = (int) std::floor((double) pOrigW * pCurH / pOrigH + 1e-7);
    pOffX           = (pCurW - cNewW) / 2;
    pOutW           = pCurW - 2 * pOffX;
  }
}

// deepseek4v: layout of the LLM token block built from the aligner grid
struct dsv4_block_layout {
  int vRows;     // grid rows, padded to an even count
  int vRowLen;   // grid width + 1 newline
  int vPadLast;  // trailing pads
  int vNOut;     // total block size, including lead pads and the start/end sentinels
};

static inline dsv4_block_layout fDsv4GetBlockLayout(int pNLlmW, int pNLlmH, int pLeadPad) {
  dsv4_block_layout vBl;
  vBl.vRows    = pNLlmH + (pNLlmH % 2);
  vBl.vRowLen  = pNLlmW + 1;
  vBl.vPadLast = (vBl.vRows / 2 * vBl.vRowLen) % 2 * 2;
  vBl.vNOut    = pLeadPad + 1 + vBl.vRows * vBl.vRowLen + vBl.vPadLast + 1;
  return vBl;
}

//
// logging
//

static void fClipLogCallbackDefault(enum ggml_log_level pLevel, const char * pText, void * pUserData) {
  (void) pLevel;
  (void) pUserData;
  fputs(pText, stderr);
  fflush(stderr);
}

struct clip_logger_state {
  ggml_log_callback vLogCallback;
  void *            vLogCallbackUserData;
};

extern struct clip_logger_state vGLoggerState;

static void fClipLogInternalV(enum ggml_log_level pLevel, const char * format, va_list pArgs) {
  if (format == NULL) {
    return;
  }
  va_list vArgsCopy;
  va_copy(vArgsCopy, pArgs);
  char aBuffer[128];
  int  vLen = vsnprintf(aBuffer, 128, format, pArgs);
  if (vLen < 128) {
    vGLoggerState.vLogCallback(pLevel, aBuffer, vGLoggerState.vLogCallbackUserData);
  } else {
    char * vBuffer2 = (char *) calloc(vLen + 1, sizeof(char));
    vsnprintf(vBuffer2, vLen + 1, format, vArgsCopy);
    vBuffer2[vLen] = 0;
    vGLoggerState.vLogCallback(pLevel, vBuffer2, vGLoggerState.vLogCallbackUserData);
    free(vBuffer2);
  }
  va_end(vArgsCopy);
}

static void fClipLogInternal(enum ggml_log_level pLevel, const char * format, ...) {
  va_list vArgs;
  va_start(vArgs, format);
  fClipLogInternalV(pLevel, format, vArgs);
  va_end(vArgs);
}

#define LOG_TRC(...) fClipLogInternal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_DBG(...) fClipLogInternal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) fClipLogInternal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WRN(...) fClipLogInternal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LOG_ERR(...) fClipLogInternal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_CNT(...) fClipLogInternal(GGML_LOG_LEVEL_CONT, __VA_ARGS__)

//
// cpp wrappers
//

struct clip_image_f32_batch {
  std::vector<clip_image_f32> lEntries;
  bool                        vIsAudio = false;

  clip_image_f32_batch clone() const {
    clip_image_f32_batch vNewBatch{
      /* entries  */ {},
      /* is_audio */ vIsAudio,
    };
    vNewBatch.lEntries.reserve(lEntries.size());
    for (const auto & vEntry : lEntries) {
      vNewBatch.lEntries.emplace_back(vEntry);  // copy
    }
    return vNewBatch;
  }

  void serialize(struct mtmd_serialization & pSer) const;
  void deserialize(struct mtmd_serialization & pSer);
};

//
// common utils
//

#ifdef _WIN32
static std::ifstream open_ifstream_binary(const std::string & fname) {
  int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
  if (!wlen) {
    throw std::runtime_error("failed to convert filename to UTF-16: " + fname);
  }
  std::vector<wchar_t> wfname(wlen);
  (void) MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wfname.data(), wlen);
  return std::ifstream(wfname.data(), std::ios::binary);
}
#else
static std::ifstream fOpenIfstreamBinary(const std::string & pFname) {
  return std::ifstream(pFname, std::ios::binary);
}
#endif

// in test-mtmd-impl, we include woth common.h and this file, and these functions are duplicated
// this is a quick fix to avoid compilation errors
#ifndef DIRECTORY_SEPARATOR
static std::string fStringFormat(const char * pFmt, ...) {
  va_list vAp;
  va_list vAp2;
  va_start(vAp, pFmt);
  va_copy(vAp2, vAp);
  int size = vsnprintf(NULL, 0, pFmt, vAp);
  GGML_ASSERT(size >= 0 && size < INT_MAX);  // NOLINT
  std::vector<char> lBuf(size + 1);
  int               vSize2 = vsnprintf(lBuf.data(), size + 1, pFmt, vAp2);
  GGML_ASSERT(vSize2 == size);
  va_end(vAp2);
  va_end(vAp);
  return std::string(lBuf.data(), lBuf.size());
}

static void fStringReplaceAll(std::string & pS, const std::string & pSearch, const std::string & pReplace) {
  if (pSearch.empty()) {
    return;
  }
  std::string vBuilder;
  vBuilder.reserve(pS.length());
  size_t vPos     = 0;
  size_t vLastPos = 0;
  while ((vPos = pS.find(pSearch, vLastPos)) != std::string::npos) {
    vBuilder.append(pS, vLastPos, vPos - vLastPos);
    vBuilder.append(pReplace);
    vLastPos = vPos + pSearch.length();
  }
  vBuilder.append(pS, vLastPos, std::string::npos);
  pS = std::move(vBuilder);
}

// split string by a `std::string delim` instead of `char delim`
static std::vector<std::string> fStringSplitStr(std::string pS, const std::string & pDelimiter) {
  std::vector<std::string> lTokens;
  size_t                   vPos = 0;
  std::string              token;
  while ((vPos = pS.find(pDelimiter)) != std::string::npos) {
    token = pS.substr(0, vPos);
    lTokens.push_back(token);
    pS.erase(0, vPos + pDelimiter.length());
  }
  lTokens.push_back(pS);
  return lTokens;
}

// remove when moving to c++20
inline bool fStringStartsWith(std::string_view str, std::string_view pPrefix) {
  return str.size() >= pPrefix.size() && str.compare(0, pPrefix.size(), pPrefix) == 0;
}

// remove when moving to c++20
inline bool fStringEndsWith(std::string_view str, std::string_view pSuffix) {
  return str.size() >= pSuffix.size() && str.compare(str.size() - pSuffix.size(), pSuffix.size(), pSuffix) == 0;
}
#endif

//
// gguf utils
//

static std::string fGgufDataToStr(enum gguf_type type, const void * data, int pI) {
  switch (type) {
    case GGUF_TYPE_UINT8:
      return std::to_string(((const uint8_t *) data)[pI]);
    case GGUF_TYPE_INT8:
      return std::to_string(((const int8_t *) data)[pI]);
    case GGUF_TYPE_UINT16:
      return std::to_string(((const uint16_t *) data)[pI]);
    case GGUF_TYPE_INT16:
      return std::to_string(((const int16_t *) data)[pI]);
    case GGUF_TYPE_UINT32:
      return std::to_string(((const uint32_t *) data)[pI]);
    case GGUF_TYPE_INT32:
      return std::to_string(((const int32_t *) data)[pI]);
    case GGUF_TYPE_UINT64:
      return std::to_string(((const uint64_t *) data)[pI]);
    case GGUF_TYPE_INT64:
      return std::to_string(((const int64_t *) data)[pI]);
    case GGUF_TYPE_FLOAT32:
      return std::to_string(((const float *) data)[pI]);
    case GGUF_TYPE_FLOAT64:
      return std::to_string(((const double *) data)[pI]);
    case GGUF_TYPE_BOOL:
      return ((const int8_t *) data)[pI] != 0 ? "true" : "false";
    default:
      return fStringFormat("unknown type %d", type);
  }
}

static std::string fGgufKvToStr(const struct gguf_context * pCtxGguf, int pI) {
  const enum gguf_type type = gguf_get_kv_type(pCtxGguf, pI);

  switch (type) {
    case GGUF_TYPE_STRING:
      return gguf_get_val_str(pCtxGguf, pI);
    case GGUF_TYPE_ARRAY:
      {
        const enum gguf_type cArrType = gguf_get_arr_type(pCtxGguf, pI);
        int                  vArrN    = gguf_get_arr_n(pCtxGguf, pI);
        const void *         data     = cArrType == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(pCtxGguf, pI);
        std::stringstream    vSs;
        vSs << "[";
        for (int vJ = 0; vJ < vArrN; vJ++) {
          if (cArrType == GGUF_TYPE_STRING) {
            std::string vVal = gguf_get_arr_str(pCtxGguf, pI, vJ);
            // escape quotes
            fStringReplaceAll(vVal, "\\", "\\\\");
            fStringReplaceAll(vVal, "\"", "\\\"");
            vSs << '"' << vVal << '"';
          } else if (cArrType == GGUF_TYPE_ARRAY) {
            vSs << "???";
          } else {
            vSs << fGgufDataToStr(cArrType, data, vJ);
          }
          if (vJ < vArrN - 1) {
            vSs << ", ";
          }
        }
        vSs << "]";
        return vSs.str();
      }
    default:
      return fGgufDataToStr(type, gguf_get_val_data(pCtxGguf, pI), 0);
  }
}

//
// debugging
//

static void fPrintTensorShape(ggml_tensor * pT) {
  printf("%s.shape = [", pT->name);
  for (int vI = 0; vI < ggml_n_dims(pT); ++vI) {
    printf("%" PRId64, pT->ne[vI]);
    if (vI < ggml_n_dims(pT) - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

static void fPrintTensorData(ggml_tensor * pT, uint8_t * data, int64_t n) {
  ggml_type type = pT->type;
  int64_t * vNe  = pT->ne;
  size_t *  vNb  = pT->nb;
  for (int64_t vI3 = 0; vI3 < vNe[3]; vI3++) {
    printf("%s.data: [\n", pT->name);
    for (int64_t vI2 = 0; vI2 < vNe[2]; vI2++) {
      if (vI2 == n && vNe[2] > 2 * n) {
        printf("     ..., \n");
        vI2 = vNe[2] - n;
      }
      printf("     [\n");
      for (int64_t vI1 = 0; vI1 < vNe[1]; vI1++) {
        if (vI1 == n && vNe[1] > 2 * n) {
          printf("      ..., \n");
          vI1 = vNe[1] - n;
        }
        printf("      [");
        for (int64_t vI0 = 0; vI0 < vNe[0]; vI0++) {
          if (vI0 == n && vNe[0] > 2 * n) {
            printf("..., ");
            vI0 = vNe[0] - n;
          }
          size_t vI = vI3 * vNb[3] + vI2 * vNb[2] + vI1 * vNb[1] + vI0 * vNb[0];
          float  vV;
          if (type == GGML_TYPE_F16) {
            vV = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[vI]);
          } else if (type == GGML_TYPE_F32) {
            vV = *(float *) &data[vI];
          } else if (type == GGML_TYPE_I32) {
            vV = (float) *(int32_t *) &data[vI];
          } else if (type == GGML_TYPE_I16) {
            vV = (float) *(int16_t *) &data[vI];
          } else if (type == GGML_TYPE_I8) {
            vV = (float) *(int8_t *) &data[vI];
          } else {
            GGML_ABORT("fatal error");
          }
          printf("%8.4f", vV);
          if (vI0 < vNe[0] - 1) {
            printf(", ");
          }
        }
        printf("],\n");
      }
      printf("     ],\n");
    }
    printf("    ]\n");
  }
}

//
// API used internally with mtmd
//

projector_type fClipGetProjectorType(const struct clip_ctx * pCtx);
void           fClipSetDebugOutputEmbeddings(struct clip_ctx * pCtx, bool pDebug);
