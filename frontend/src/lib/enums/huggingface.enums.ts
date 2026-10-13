/**
 * HuggingFace Hub enums.
 *
 * Values mirror the strings used by the HF REST API
 * (https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
 * so they can be sent and compared directly.
 */

/** Sort field for /api/models search queries. */
export enum HfModelSort {
  cCreatedAt = 'createdAt',
  cDownloads = 'downloads',
  cLastModified = 'lastModified',
  cLikes = 'likes',
  cTrendingScore = 'trendingScore'
}

/**
 * Where the sidecar token (`mtp` / `dflash` / `mmproj` / ...) sits in the
 * filename.
 * - `prefix`  sidecar file that lives next to the main weights, e.g. `mtp-Q4_0.gguf`
 * - `suffix`  embedded draft baked into the main weights, e.g. `Hy3-IQ1_M-mtp.gguf`
 * - `infix`   standalone sidecar named between head and quant, e.g. `model-mtp-Q8_0.gguf`
 */
export enum SidecarForm {
  cInfix = 'infix',
  PREFIX = 'prefix',
  SUFFIX = 'suffix'
}

/** Entry type in a model repository file tree (`/tree` responses). */
export enum HfEntryType {
  cDirectory = 'directory',
  FILE = 'file'
}
