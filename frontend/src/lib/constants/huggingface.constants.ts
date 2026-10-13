/**
 * HuggingFace Hub constants.
 *
 * URLs, parsing regexes and formatting units for the HuggingFaceService.
 * Reference: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
 */

// API endpoints

export const cHfBaseUrl = 'https://huggingface.co';
export const cHfApiModelsUrl = `${cHfBaseUrl}/api/models`;
export const cHfAvatarsUrl = `${cHfBaseUrl}/api/avatars`;

// Query params

export const cHfFullDetailParam = 'full=true';
export const cHfRecursiveTreeParam = 'recursive=true';
/** Search filter that restricts results to repos containing GGUF files. */
export const cHfGgufFilter = 'gguf';
/** Repeatable `expand` query param selecting fields on the list endpoint. */
export const cHfExpandParam = 'expand';
/**
 * Fields the model list endpoint omits by default but the discover list rows
 * render: `gguf` (chat template, context length, param count) drives the
 * reasoning / tool-use icons and the context badge, `siblings` the vision and
 * draft-sidecar badges. Without them those parts of a row stay empty.
 */
export const aHfModelListExpand: readonly string[] = [
  'author',
  'downloads',
  'gguf',
  'lastModified',
  'likes',
  'pipeline_tag',
  'siblings',
  // `base_model:` tags, so search rows can show the base org's avatar as the
  // main avatar with the quant org as the corner badge, like catalog rows.
  'tags'
];

// Repo file conventions

export const cHfMainBranch = 'main';
export const cHfReadmeFilename = 'README.md';
export const cHfRawPath = 'raw';
export const cHfTreePath = 'tree';

// Pagination

export const cHfLinkNextRegex = /<([^>]+)>;\s*rel="next"/;
/** `Link` response header carrying the next page URL for cursor pagination. */
export const cHfLinkHeader = 'Link';

// Fetch retry

export const cHfRetryAttempts = 3;
export const cHfRetryDelayMs = 1000;
export const cHfHttpNotFound = 404;
export const cHfHttpServerErrorMin = 500;

// Search limits

export const cHfDefaultLimit = 50;
/** Safety cap on `/tree` pagination: more pages means a misbehaving endpoint. */
export const cHfTreeMaxPages = 10;
export const cHfMaxLimit = 100;

// GGUF shard files

/** Matches a split-shard GGUF file name, e.g. `Model-00001-of-00015.gguf`. */
export const cHfShardRegex = /-(\d{5})-of-(\d{5})\.gguf$/i;
/** Index (1-based) of the first shard in a split-shard set. */
export const cHfFirstShard = 1;
/** Zero-padded width of the shard index in a split-shard file name. */
export const cHfShardPadWidth = 5;

// Quantization tokens

/** `UD-` (Unsloth Dynamic) custom quantization prefix, e.g. `UD-Q4_K_XL`. */
export const cHfUdQuantPrefix = 'UD';
export const cHfUdQuantPrefixRegex = /^UD-/i;
/**
 * Segment marking an Unsloth `shared-` draft head that borrows the target
 * model's embedding/output weights, e.g. `...-shared-Q4_K_M.gguf`.
 */
export const cHfSharedDraftToken = 'shared';
/**
 * Extracts the leading precision digits from a quant token, e.g.
 * `Q4_K_XL` -> 4, `IQ2_XXS` -> 2, `TQ1_0` -> 1, `BF16` -> 16.
 */
export const cHfQuantPrecisionRegex = /^(?:I?Q|TQ|BF|F|MXFP)?(\d+)/i;

// Model card tags

/** Matches the `base_model:` tag (plain or `quantized:`), capturing the repo id. */
export const cHfBaseModelTagRegex = /^base_model:(?:quantized:)?(.+)$/;
export const cHfLicenseTagPrefix = 'license:';
export const cHfGatedTag = 'gated';
export const cHfGgufTag = 'gguf';
export const cHfSafetensorsTag = 'safetensors';

// Pipeline tasks (logic use only - matching `pipeline_tag` values against tags)

/**
 * `pipeline_tag` values grouped by the input/output modality they imply, used
 * to derive a discover row's modality icons. A tag in more than one group (e.g.
 * `image-to-video`) lights up each modality it belongs to.
 */
export const cHfModalityPipelineTags: Readonly<
  Record<'audio' | 'video' | 'vision', readonly string[]>
> = {
  audio: [
    'audio-classification',
    'audio-to-audio',
    'automatic-speech-recognition',
    'text-to-speech',
    'voice-activity-detection'
  ],
  video: ['text-to-video', 'image-to-video', 'video-to-video'],
  vision: ['image-text-to-text', 'image-to-text', 'text-to-image', 'image-to-video']
};

/** Filename token marking an mmproj sidecar sibling (unlocks vision / audio). */
export const cHfMmprojFilenameToken = 'mmproj';

export const aHfTaskTags: readonly string[] = [
  'audio-classification',
  'audio-to-audio',
  'automatic-speech-recognition',
  'conversational',
  'depth-estimation',
  'feature-extraction',
  'fill-mask',
  'image-classification',
  'image-feature-extraction',
  'image-segmentation',
  'image-text-to-text',
  'image-to-text',
  'image-to-video',
  'object-detection',
  'question-answering',
  'reinforcement-learning',
  'robotics',
  'sentence-similarity',
  'summarization',
  'text2text-generation',
  'text-classification',
  'text-generation',
  'text-to-image',
  'text-to-speech',
  'text-to-video',
  'token-classification',
  'translation',
  'video-to-video',
  'voice-activity-detection',
  'zero-shot-classification'
];

// Formatting

export const cByte = 1;
export const cKilobyte = 1_000;
export const cMegabyte = 1_000_000;
export const cGigabyte = 1_000_000_000;
export const cTerabyte = 1_000_000_000_000;

/**
 * Matches a human size string (`177GB`, `1.2 TB`, `500MB`), capturing the
 * numeric value and its unit suffix. Used by `parseSizeBytes`.
 */
export const cHfSizeStringRegex = /^\s*([\d.]+)\s*([a-z]+)\s*$/i;

/**
 * Byte multiplier for a size suffix (`k` kilobyte, `m` megabyte, ...) as used by
 * the dgs.app catalog `size` strings, whose suffix is lowercase.
 */
export const cHfSizeSuffixBytes: Readonly<Record<string, number>> = {
  b: cByte,
  g: cGigabyte,
  k: cKilobyte,
  m: cMegabyte,
  t: cTerabyte
};

export const cByteLabel = 'B';
export const cKilobyteLabel = 'KB';
export const cMegabyteLabel = 'MB';
export const cGigabyteLabel = 'GB';

/** Count suffixes for compact number formatting, e.g. `1.5K`, `2.0M`. */
export const cKiloLabel = 'K';
export const cMegaLabel = 'M';
export const cGigaLabel = 'B';

// Relative time

export const cMsPerDay = 1000 * 60 * 60 * 24;
export const cDaysPerWeek = 7;
/** Rough month length in days, used to bucket relative timestamps. */
export const cDaysPerMonth = 30;
export const cDaysPerYear = 365;

export const cTodayLabel = 'Today';
export const cYesterdayLabel = 'Yesterday';
export const cDaysAgoLabel = 'days ago';
export const cWeeksAgoLabel = 'weeks ago';
export const cMonthsAgoLabel = 'months ago';
export const cYearsAgoLabel = 'years ago';

// Cache paths

/**
 * Matches a local HF cache file path
 * (`.../models--<org>--<name>/snapshots/<sha>/<file>`), capturing the repo
 * directory name and the repo-relative file path.
 */
export const cHfCachePathRegex = /models--(.+?)\/snapshots\/[^/]+\/(.+)$/;
/** Separator between org and name segments in an HF cache directory name. */
export const cHfCacheDirSeparator = '--';

// README

/** Matches a leading YAML frontmatter block (--- ... ---) in a markdown document. */
export const cHfFrontmatterRegex = /^---\r?\n[\s\S]*?\r?\n---\r?\n?/;

// Param counts

/**
 * Best-effort parameter count token in a model id/name, e.g. `27B` from
 * `Qwen3.8-27B-GGUF` or `300M` from `embeddinggemma-300M-GGUF`.
 */
export const cHfParamCountRegex = /(?:^|[^a-z0-9])(\d+(?:[._]\d+)?)\s*([bm])(?![a-z0-9])/i;
