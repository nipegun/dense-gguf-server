export enum ModelModality {
  AUDIO = 'AUDIO',
  TEXT = 'TEXT',
  VIDEO = 'VIDEO',
  VISION = 'VISION'
}

export enum ModelCapability {
  REASONING = 'reasoning',
  cToolUse = 'tools'
}

/**
 * Speculative-decoding draft sidecars (server spec-type draft-*).
 * Filenames use the lowercase token, e.g. `mtp-<name>.gguf` or `-mtp` suffix.
 */
export enum ModelDraftSidecar {
  /** DFlash block-diffusion draft (spec-type draft-dflash). */
  cDflash = 'dflash',
  /** DSpark block-diffusion draft (spec-type draft-dspark). */
  cDspark = 'dspark',
  /** EAGLE-3 speculative draft (spec-type draft-eagle3). */
  cEagle3 = 'eagle3',
  /** Multi-token-prediction draft head (spec-type draft-mtp). */
  cMtp = 'mtp'
}

/**
 * Non-draft sidecar file types. A sidecar is any auxiliary GGUF file
 * accompanying the main model weights.
 */
export enum ModelAuxSidecar {
  /** Importance-matrix data used to build imatrix quants; not loaded at serve time. */
  cImatrix = 'imatrix',
  /** Multimodal projector: unlocks vision and/or audio input modalities. */
  cMmproj = 'mmproj'
}

/**
 * Why an in-flight download is being stopped, so the terminal `download_failed`
 * feed event can be attributed: a user pause (resumable) or a user cancel
 * (discard). Distinguishes these from a genuine download failure.
 */
export enum ModelDownloadStopRequest {
  cCancel = 'cancel',
  cPause = 'pause'
}

/**
 * Destructive download action the user is asked to confirm: stop and discard an
 * in-flight download, or delete an already-downloaded model from disk. Both
 * resolve through the same store removal call, differing only in the copy.
 */
export enum ModelDownloadConfirmAction {
  cCancel = 'cancel',
  DELETE = 'delete'
}

/** What a table row group folds: the quants of one repo, or its variants. */
export enum ModelGroupKind {
  cQuants = 'quants',
  cVariants = 'variants'
}

/** Kind of one collapsible block of the models manager table. */
export enum ModelsTableGroupKind {
  cDownloading = 'downloading',
  cFavorites = 'favorites',
  cHidden = 'hidden',
  cLoaded = 'loaded',
  cLocal = 'local'
}

/** Download state a manager row reports in place of its load state. */
export enum ModelRowDownloadState {
  cDownloading = 'downloading',
  cPaused = 'paused'
}

/** Column the models manager table can be ordered by. */
export enum ModelsTableSortKey {
  CONTEXT = 'context',
  NAME = 'name'
}
