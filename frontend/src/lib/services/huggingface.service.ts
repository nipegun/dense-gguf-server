import { fTranslate } from '$lib/i18n';
import { cPathSeparator, cSettingsKeys } from '$lib/constants';
import {
  cByte,
  cByteLabel,
  cDaysAgoLabel,
  cDaysPerMonth,
  cDaysPerWeek,
  cDaysPerYear,
  cGigaLabel,
  cGigabyte,
  cGigabyteLabel,
  cHfApiModelsUrl,
  cHfAvatarsUrl,
  cHfBaseModelTagRegex,
  cHfBaseUrl,
  cHfCacheDirSeparator,
  cHfCachePathRegex,
  cHfDefaultLimit,
  cHfFirstShard,
  cHfFrontmatterRegex,
  cHfFullDetailParam,
  cHfGatedTag,
  cHfGgufFilter,
  cHfGgufTag,
  cHfHttpNotFound,
  cHfHttpServerErrorMin,
  cHfLicenseTagPrefix,
  cHfLinkHeader,
  cHfLinkNextRegex,
  cHfMainBranch,
  cHfMaxLimit,
  aHfModelListExpand,
  cHfParamCountRegex,
  cHfQuantPrecisionRegex,
  cHfRawPath,
  cHfReadmeFilename,
  cHfRecursiveTreeParam,
  cHfRetryAttempts,
  cHfRetryDelayMs,
  cHfSafetensorsTag,
  cHfShardPadWidth,
  cHfShardRegex,
  cHfSharedDraftToken,
  cHfSizeStringRegex,
  cHfSizeSuffixBytes,
  aHfTaskTags,
  cHfTreeMaxPages,
  cHfTreePath,
  cHfUdQuantPrefix,
  cHfUdQuantPrefixRegex,
  cKiloLabel,
  cKilobyte,
  cKilobyteLabel,
  cMegaLabel,
  cMegabyte,
  cMegabyteLabel,
  cMonthsAgoLabel,
  cMsPerDay,
  cTodayLabel,
  cWeeksAgoLabel,
  cYearsAgoLabel,
  cYesterdayLabel
} from '$lib/constants';
import { MODEL_ID, type ModelSidecar } from '$lib/constants';
import { HfEntryType, HfModelSort, SidecarForm } from '$lib/enums';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import type {
  HfModelDetailInfo,
  HfModelInfo,
  HfModelSearchParams,
  HfModelSibling
} from '$lib/types/huggingface';
import { fSidecarFromFileToken } from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';

/** Fetch failure carrying the HTTP status, so retry logic tests the code instead of the message. */
class HfHttpStatusError extends Error {
  status: number;

  constructor(pStatus: number, pStatusText: string) {
    super(`API request failed: ${pStatus} ${pStatusText}`);

    this.status = pStatus;
  }
}

export class HuggingFaceService {
  private static readonly vBaseUrl = cHfApiModelsUrl;

  // Cached base model lookups keyed by repo id, so repeated selector opens
  // never re-hit the HF API for the same repo.
  private static dBaseModelCache = new Map<string, { org: string; name: string } | null>();

  private static dBaseModelPending = new Map<
    string,
    Promise<{ org: string; name: string } | null>
  >();

  // Model details fetched this session, keyed by repo id: the Hub rate limits
  // aggressively, so each repo costs at most one request per load. Failures stay
  // uncached so the next mount can retry.
  private static vDetailsCache = new SvelteMap<string, HfModelDetailInfo | null>();

  private static dDetailsPending = new Map<string, Promise<HfModelDetailInfo | null>>();

  /**
   * Map of quant token to its average bit-depth in bits-per-weight (bpw).
   */
  private static readonly dQuantBitDepth: Record<string, number> = {
    BF16: 16,
    F16: 16,
    IQ1_M: 1,
    IQ1_S: 1,
    IQ1_XS: 1,
    IQ1_XXS: 1,
    IQ2_M: 2,
    IQ2_S: 2,
    IQ2_XS: 2,
    IQ2_XXS: 2,
    IQ3_M: 3,
    IQ3_S: 3,
    IQ3_XS: 3,
    IQ3_XXS: 3,
    Q2_K: 2,
    Q2_K_M: 2,
    Q2_K_S: 2,
    Q3_K: 3,
    Q3_K_L: 3,
    Q3_K_M: 3,
    Q3_K_S: 3,
    Q4_0: 4,
    Q4_1: 4,
    Q4_K: 4,
    Q4_K_M: 4,
    Q4_K_S: 4,
    Q5_0: 5,
    Q5_1: 5,
    Q5_K: 5,
    Q5_K_M: 5,
    Q5_K_S: 5,
    Q6_K: 6,
    Q8_0: 8
  };

  /** Details already fetched, for a caller that cannot await (a table sorting its rows). */
  static mCachedDetails(pModelId: string): HfModelDetailInfo | null | undefined {
    // dgs.cpp model ids carry the quant tag after a colon
    const [vHfRepoId] = pModelId.split(MODEL_ID.QUANTIZATION_SEPARATOR);

    return HuggingFaceService.vDetailsCache.get(vHfRepoId);
  }

  /**
   * Collapse split GGUF shard sets (`-00001-of-00015.gguf`, ...) to their first
   * shard, summing every shard's size so the kept entry reflects the whole
   * quant. Downloads are tag-based (`repo:quant`), so the first shard is
   * enough to represent the set.
   */
  static mCollapseGgufShards(pSiblings: HfModelSibling[]): HfModelSibling[] {
    const dSizeByPath = new Map(pSiblings.map((pF) => [pF.path, pF.size ?? 0]));
    const aResult: HfModelSibling[] = [];

    for (const file of pSiblings) {
      const cMatch = cHfShardRegex.exec(file.path);

      if (!cMatch) {
        aResult.push(file);

        continue;
      }

      if (Number(cMatch[1]) !== cHfFirstShard) continue;

      const cTotal = Number(cMatch[2]);
      const cStem = file.path.slice(0, file.path.length - cMatch[0].length);

      let vSize = 0;

      for (let vI = cHfFirstShard; vI <= cTotal; vI++) {
        const cShard = HuggingFaceService.mShardPath(cStem, vI, cTotal);

        vSize += dSizeByPath.get(cShard) ?? 0;
      }

      aResult.push({ ...file, size: vSize });
    }

    return aResult;
  }

  // GGUF Model Browsing

  /**
   * Extract the GGUF quantization token (e.g. `Q4_K_M`) and any sidecar type
   * (`mtp`, `dflash`, `mmproj`, ...) from a `.gguf` filename.
   *
   * `sidecarForm` records which side of the filename the sidecar token sat on
   * so callers can render badges differently. `quant` and `sidecar` are `null`
   * when absent; returns `null` for non-GGUF filenames.
   */
  static mExtractQuantMeta(pFilename: string): {
    quant: string | null;
    /** Draft-head-only variant borrowing embed/output weights from the target model. */
    shared: boolean;
    sidecar: ModelSidecar | null;
    sidecarForm: SidecarForm | null;
  } | null {
    if (!MODEL_ID.WEIGHT_EXTENSION_REGEX.test(pFilename)) return null;

    // HF repos may nest sidecars in a folder (e.g. `MTP/mtp-Model-Q4_0.gguf`);
    // parse the file name only, the folder adds no quant information.
    let vSource = (pFilename.split(cPathSeparator).pop() ?? pFilename).replace(
      MODEL_ID.WEIGHT_EXTENSION_REGEX,
      ''
    );
    let vSidecar: ModelSidecar | null = null;
    let vSidecarForm: SidecarForm | null = null;

    // A file named just the sidecar token (`imatrix.gguf`) is the sidecar
    // itself: no name or quant segments to parse.
    const cBareSidecar = fSidecarFromFileToken(vSource.toLowerCase());

    if (cBareSidecar) {
      return { quant: null, shared: false, sidecar: cBareSidecar, sidecarForm: SidecarForm.PREFIX };
    }

    const cPrefixMatch = vSource.match(MODEL_ID.SIDECAR_PREFIX_REGEX);

    if (cPrefixMatch) {
      vSidecar = fSidecarFromFileToken(cPrefixMatch[1].toLowerCase());
      vSidecarForm = SidecarForm.PREFIX;
      vSource = cPrefixMatch[2];
    } else {
      const cSuffixMatch = vSource.match(MODEL_ID.SIDECAR_SUFFIX_REGEX);

      if (cSuffixMatch) {
        // Take the suffix sidecar even when the head carries no quant:
        // embedded drafts end in one (`Hy3-IQ1_M-mtp`), standalone sidecar
        // files do not (`Model-mtp-draft`, `Model-imatrix`).
        vSidecar = fSidecarFromFileToken(cSuffixMatch[2].toLowerCase());
        vSidecarForm = SidecarForm.SUFFIX;
        vSource = cSuffixMatch[1];
      } else {
        const cInfixMatch = vSource.match(MODEL_ID.SIDECAR_INFIX_REGEX);

        if (cInfixMatch) {
          vSidecar = fSidecarFromFileToken(cInfixMatch[2].toLowerCase());
          vSidecarForm = SidecarForm.cInfix;
          vSource = `${cInfixMatch[1]}-${cInfixMatch[3]}`;
        }
      }
    }

    // Scan dash-separated segments left-to-right for the first quant match.
    // - For sidecars like `mtp-Q4_0-180MB.gguf` the quant is `Q4_0`.
    // - For embedded MTP like `Hy3-IQ1_M-mtp.gguf` we have `Hy3-IQ1_M` and `IQ1_M` matches.
    // - For main files like `Qwen3.5-4B-Q4_K_M.gguf` we land on the trailing quant.
    const aSegments = vSource.split(MODEL_ID.SEGMENT_SEPARATOR);
    const cQuantIdx = aSegments.findIndex((pSeg) => MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(pSeg));
    // Unsloth ships draft heads in two layouts: `shared-` files borrow the
    // embedding/output weights from the target model, others are self-contained.
    const cShared = aSegments.some((pSeg) => pSeg.toLowerCase() === cHfSharedDraftToken);

    let vQuant = cQuantIdx >= 0 ? aSegments[cQuantIdx].toUpperCase() : null;

    // Recombine a `UD-` (Unsloth Dynamic) prefix, e.g. `...-UD-Q4_K_XL.gguf`.
    // The prefix must be the whole previous segment, matching the server's
    // `UD-<quant>` custom-quant convention (e.g. not `-mtp-Q4_K_M`).
    const cUdPrefixIdx = cQuantIdx - 1;

    if (vQuant && cQuantIdx > 0 && aSegments[cUdPrefixIdx].toUpperCase() === cHfUdQuantPrefix) {
      vQuant = `${cHfUdQuantPrefix}-${vQuant}`;
    }

    return { quant: vQuant, shared: cShared, sidecar: vSidecar, sidecarForm: vSidecarForm };
  }

  static mFilterByExtension(pSiblings: HfModelSibling[], pExt: string): HfModelSibling[] {
    return pSiblings
      .filter((pF) => pF.path.toLowerCase().endsWith(pExt.toLowerCase()) && (pF.size ?? 0) > 0)
      .sort((pA, pB) => (pB.size ?? 0) - (pA.size ?? 0));
  }

  static mFormatDownloads(pDownloads: number): string {
    if (pDownloads >= cGigabyte) {
      return `${(pDownloads / cGigabyte).toFixed(1)}${cGigaLabel}`;
    }

    if (pDownloads >= cMegabyte) {
      return `${(pDownloads / cMegabyte).toFixed(1)}${cMegaLabel}`;
    }

    if (pDownloads >= cKilobyte) {
      return `${(pDownloads / cKilobyte).toFixed(1)}${cKiloLabel}`;
    }

    return pDownloads.toString();
  }

  static mFormatFileSize(pBytes: number): string {
    if (pBytes >= cGigabyte) {
      return `${(pBytes / cGigabyte).toFixed(1)} ${cGigabyteLabel}`;
    }

    if (pBytes >= cMegabyte) {
      return `${(pBytes / cMegabyte).toFixed(1)} ${cMegabyteLabel}`;
    }

    if (pBytes >= cKilobyte) {
      return `${(pBytes / cKilobyte).toFixed(1)} ${cKilobyteLabel}`;
    }

    return `${pBytes} ${cByteLabel}`;
  }

  static mFormatLikes(pLikes: number): string {
    if (pLikes >= cKilobyte) {
      return `${(pLikes / cKilobyte).toFixed(1)}${cKiloLabel}`;
    }

    return pLikes.toString();
  }

  static mFormatRelativeTime(pTimestamp: string): string {
    const cDate = new Date(pTimestamp);
    const cNow = new Date();
    const cDiffMs = cNow.getTime() - cDate.getTime();
    // timestamps can lie in the future (clock skew); clamp so they read as today
    const cDiffDays = Math.max(0, Math.floor(cDiffMs / cMsPerDay));

    if (cDiffDays === 0) return cTodayLabel;

    if (cDiffDays === 1) return cYesterdayLabel;

    if (cDiffDays < cDaysPerWeek) return `${cDiffDays} ${cDaysAgoLabel}`;

    if (cDiffDays < cDaysPerMonth) {
      return `${Math.floor(cDiffDays / cDaysPerWeek)} ${cWeeksAgoLabel}`;
    }

    if (cDiffDays < cDaysPerYear) {
      return `${Math.floor(cDiffDays / cDaysPerMonth)} ${cMonthsAgoLabel}`;
    }

    return `${Math.floor(cDiffDays / cDaysPerYear)} ${cYearsAgoLabel}`;
  }

  /**
   * Format a min-max size range with one shared unit, e.g. `19.0-28.6 GB`.
   */
  static mFormatSizeRange(pMin: number, pMax: number): string {
    const cUnit =
      pMax >= cGigabyte
        ? cGigabyteLabel
        : pMax >= cMegabyte
          ? cMegabyteLabel
          : pMax >= cKilobyte
            ? cKilobyteLabel
            : cByteLabel;
    const cDiv =
      cUnit === cGigabyteLabel
        ? cGigabyte
        : cUnit === cMegabyteLabel
          ? cMegabyte
          : cUnit === cKilobyteLabel
            ? cKilobyte
            : cByte;
    const fFmt = (pN: number) => (cDiv === cByte ? `${pN}` : `${(pN / cDiv).toFixed(1)}`);

    return `${fFmt(pMin)}-${fFmt(pMax)} ${cUnit}`;
  }

  // Model Details & Files

  /**
   * Avatar URL for an author (org or user). 404s when the author does not
   * exist, so callers should provide a fallback.
   */
  static mGetAvatarUrl(pAuthor: string): string {
    // OpenRouter-style model ids prefix the provider with a tilde
    // (`~openai/gpt-...`); the avatars endpoint only resolves bare names
    return `${cHfAvatarsUrl}${cPathSeparator}${pAuthor.replace(/^~/, '')}`;
  }

  /**
   * Resolve the original (non-GGUF) base model `{ org, name }` for a GGUF repo
   * from its HF card (`cardData.base_model`). Returns null when the card has no
   * base model. A resolved repo is cached for the session.
   */
  static mGetBaseModel(pRepoId: string): Promise<{ org: string; name: string } | null> {
    if (!HuggingFaceService.mIsEnabled()) return Promise.resolve(null);

    // all quants of one repo share the cached lookup, so key it by the repo id
    const [vHfRepoId] = pRepoId.split(MODEL_ID.QUANTIZATION_SEPARATOR);
    const cCached = this.dBaseModelCache.get(vHfRepoId);

    if (cCached !== undefined) return Promise.resolve(cCached);

    const cPending = this.dBaseModelPending.get(vHfRepoId);

    if (cPending) return cPending;

    const cPromise = (async () => {
      const cDetails = await this.mGetDetails(vHfRepoId);
      const cBase = this.mGetBaseModels(cDetails)[0];

      if (!cBase) return null;

      const [vOrg, ...aRest] = cBase.split(cPathSeparator);

      return { name: aRest.join(cPathSeparator), org: vOrg };
    })();

    this.dBaseModelPending.set(vHfRepoId, cPromise);

    cPromise
      // a repo with no base model stays uncached: the details behind it are cached
      // already, so a retry costs nothing and a failed lookup can try again
      .then((pResult) => {
        if (pResult) this.dBaseModelCache.set(vHfRepoId, pResult);
      })
      .finally(() => this.dBaseModelPending.delete(vHfRepoId));

    return cPromise;
  }

  /**
   * Extract the original (non-GGUF) base model ids for a repo, from
   * `cardData.base_model` (string or list) and the `base_model:` tags.
   */
  static mGetBaseModels(pModel: HfModelDetailInfo | null): string[] {
    if (!pModel) return [];

    const cCardBase = pModel.cardData?.base_model;
    const aFromCard: string[] = Array.isArray(cCardBase) ? cCardBase : cCardBase ? [cCardBase] : [];
    const aFromTags = (pModel.tags ?? [])
      .map((pT) => cHfBaseModelTagRegex.exec(pT)?.[1])
      .filter((pV): pV is string => Boolean(pV));

    return Array.from(new Set([...aFromCard, ...aFromTags]));
  }

  /**
   * Look up the average bit-depth for a known GGUF quantization.
   * Returns `null` for unrecognized tokens.
   */
  static mGetBitDepth(pQuant: string): number | null {
    const cBase = pQuant.replace(cHfUdQuantPrefixRegex, '');
    const cDirect = HuggingFaceService.dQuantBitDepth[cBase];

    if (cDirect !== undefined) return cDirect;

    // Fall back to the leading precision digits for variants missing from the
    // map, e.g. `Q4_K_XL` -> 4, `IQ2_XXS` -> 2, `TQ1_0` -> 1, `BF16` -> 16.
    const cMatch = cHfQuantPrecisionRegex.exec(cBase);

    return cMatch ? parseInt(cMatch[1], 10) : null;
  }

  static async mGetByTask(
    pPipelineTag: string,
    pParams: Omit<HfModelSearchParams, 'pipeline_tag'> = {}
  ): Promise<HfModelInfo[]> {
    return this.mSearch({
      ...pParams,
      pipeline_tag: pPipelineTag
    });
  }

  static mGetDetails(pModelId: string): Promise<HfModelDetailInfo | null> {
    if (!HuggingFaceService.mIsEnabled()) return Promise.resolve(null);

    const cCached = HuggingFaceService.vDetailsCache.get(pModelId);

    if (cCached !== undefined) return Promise.resolve(cCached);

    const cPending = HuggingFaceService.dDetailsPending.get(pModelId);

    if (cPending) return cPending;

    const cPromise = (async () => {
      // Do not encode the modelId, it contains slashes for author/name.
      // `full=true` includes cardData (description, base_model) and safetensors.
      const url = `${cHfApiModelsUrl}${cPathSeparator}${pModelId}?${cHfFullDetailParam}`;

      try {
        const cResponse = await fetch(url);

        // a missing model is a definitive answer, cache it
        if (cResponse.status === cHfHttpNotFound) {
          HuggingFaceService.vDetailsCache.set(pModelId, null);

          return null;
        }

        if (!cResponse.ok) throw new Error(`Failed to fetch model details: ${cResponse.status}`);

        // the hub answers with a list of provider entries for one repo; the gguf and
        // card data sit on the entry that carries them
        const cPayload = (await cResponse.json()) as HfModelDetailInfo | HfModelDetailInfo[];
        const entries = Array.isArray(cPayload) ? cPayload : [cPayload];
        const data = entries.find((pEntry) => pEntry?.gguf ?? pEntry?.cardData) ?? entries[0];

        if (!data) return null;

        HuggingFaceService.vDetailsCache.set(pModelId, data);

        return data;
      } catch (error) {
        // not cached: a rate limited or failed fetch retries on the next mount
        // instead of hiding the model for the session
        console.error(`Error fetching details for ${pModelId}:`, error);

        return null;
      } finally {
        HuggingFaceService.dDetailsPending.delete(pModelId);
      }
    })();

    HuggingFaceService.dDetailsPending.set(pModelId, cPromise);

    return cPromise;
  }

  static mGetModelUrl(pModelId: string): string {
    return `${cHfBaseUrl}${cPathSeparator}${pModelId}`;
  }

  static async mGetMostLiked(pLimit: number = cHfDefaultLimit): Promise<HfModelInfo[]> {
    return this.mSearch({ limit: pLimit, sort: HfModelSort.cLikes });
  }

  // Utility Methods

  static async mGetNew(pLimit: number = cHfDefaultLimit): Promise<HfModelInfo[]> {
    return this.mSearch({ limit: pLimit, sort: HfModelSort.cCreatedAt });
  }

  static async mGetPopular(pLimit: number = cHfDefaultLimit): Promise<HfModelInfo[]> {
    return this.mSearch({ limit: pLimit, sort: HfModelSort.cDownloads });
  }

  /**
   * Fetch the raw README.md for a repo, with the YAML frontmatter stripped.
   */
  static async mGetReadme(pModelId: string): Promise<string | null> {
    if (!HuggingFaceService.mIsEnabled()) return null;

    // Do not encode the modelId, it contains slashes for author/name
    const url = `${cHfBaseUrl}${cPathSeparator}${pModelId}${cPathSeparator}${cHfRawPath}${cPathSeparator}${cHfMainBranch}${cPathSeparator}${cHfReadmeFilename}`;

    try {
      const cResponse = await fetch(url);

      if (cResponse.status === cHfHttpNotFound) return null;

      if (!cResponse.ok) throw new Error(`Failed to fetch README: ${cResponse.status}`);

      return HuggingFaceService.mStripFrontmatter(await cResponse.text());
    } catch (error) {
      console.error(`Error fetching README for ${pModelId}:`, error);

      return null;
    }
  }

  /**
   * Get repository file tree to list available GGUF variants. Recursive so
   * repos that keep quants in per-quant subdirectories (e.g. `UD-Q4_K_XL/`)
   * are included; follows cursor pagination for repos over one page.
   */
  static async mGetTree(pModelId: string): Promise<HfModelSibling[]> {
    if (!HuggingFaceService.mIsEnabled()) return [];

    const aFiles: HfModelSibling[] = [];
    const cFirstUrl =
      `${cHfApiModelsUrl}${cPathSeparator}${pModelId}${cPathSeparator}${cHfTreePath}` +
      `${cPathSeparator}${cHfMainBranch}?${cHfRecursiveTreeParam}`;

    let vUrl: string | null = cFirstUrl;

    try {
      for (let vPage = 0; vUrl && vPage < cHfTreeMaxPages; vPage++) {
        const cResponse: Response = await fetch(vUrl);

        if (!cResponse.ok) return aFiles;

        const aData = (await cResponse.json()) as HfModelSibling[];

        aFiles.push(...aData.filter((pF) => pF.type !== HfEntryType.cDirectory));

        vUrl = HuggingFaceService.mParseNextPageUrl(cResponse.headers.get(cHfLinkHeader));
      }
    } catch {
      // Return whatever was fetched before the failure.
    }

    return aFiles;
  }

  static async mGetTrending(pLimit: number = cHfDefaultLimit): Promise<HfModelInfo[]> {
    return this.mSearch({ limit: pLimit, sort: HfModelSort.cTrendingScore });
  }

  /**
   * True when the UI may read model metadata from the Hugging Face Hub. Off by
   * default: with it off, callers only see what the server's /v1/models reports
   * and the org avatars stay hidden.
   */
  static mIsEnabled(): boolean {
    return cSettingsStore.vConfig[cSettingsKeys.USE_HUGGING_FACE_HUB] ?? false;
  }

  /**
   * Parameter count a repo reports: a GGUF repo spells it out in its metadata,
   * a transformers repo reports it in its SafeTensors index.
   */
  static mParameterCount(pDetails?: HfModelDetailInfo | null): number | null {
    return pDetails?.gguf?.total ?? pDetails?.safetensors?.total ?? null;
  }

  /**
   * Parse a local HF cache file path
   * (`.../models--<org>--<name>/snapshots/<sha>/<file>`) into its repo id and
   * repo-relative file path. Returns null when the path is not an HF cache path.
   */
  static mParseCachePath(pPath: string): { repo: string; file: string } | null {
    // the paths come from the server's CLI args, which use native separators
    const cMatch = cHfCachePathRegex.exec(pPath.replace(/\\/g, cPathSeparator));

    if (!cMatch) return null;

    const aParts = cMatch[1].split(cHfCacheDirSeparator);

    if (aParts.length < 2) return null;

    return {
      file: cMatch[2],
      repo: `${aParts[0]}${cPathSeparator}${aParts.slice(1).join(cHfCacheDirSeparator)}`
    };
  }

  /**
   * Best-effort parameter count parsed from a model id/name, e.g. `27B` from
   * `Qwen3.8-27B-GGUF` or `300M` from `embeddinggemma-300M-GGUF`. Returns null
   * when no size token is present.
   */
  static mParseParamCount(pName: string): string | null {
    const cMatch = cHfParamCountRegex.exec(pName);

    if (!cMatch) return null;

    return `${cMatch[1]}${cMatch[2].toUpperCase()}`;
  }

  /**
   * Parse a human size string (`177GB`, `1.2 TB`, `500MB`) to bytes. Returns
   * null when it carries no number or no known suffix, so callers can fall
   * back to another source instead of showing a wrong size.
   */
  static mParseSizeBytes(pSize: string): number | null {
    const cMatch = cHfSizeStringRegex.exec(pSize);

    if (!cMatch) return null;

    const cValue = parseFloat(cMatch[1]);
    const cMultiplier = cHfSizeSuffixBytes[cMatch[2].toLowerCase()];

    if (!Number.isFinite(cValue) || cMultiplier === undefined) return null;

    return cValue * cMultiplier;
  }

  static mParseTags(pTags: string[]): {
    license: string | null;
    isGated: boolean;
    isGguf: boolean;
    isSafetensors: boolean;
    tasks: string[];
  } {
    const cLicense =
      pTags
        .find((pTag) => pTag.startsWith(cHfLicenseTagPrefix))
        ?.replace(cHfLicenseTagPrefix, '') || null;
    const cIsGated = pTags.includes(cHfGatedTag);
    const cIsGguf = pTags.includes(cHfGgufTag);
    const cIsSafetensors = pTags.includes(cHfSafetensorsTag);
    const aTasks = pTags.filter((pTag) => aHfTaskTags.includes(pTag));

    return {
      isGated: cIsGated,
      isGguf: cIsGguf,
      isSafetensors: cIsSafetensors,
      license: cLicense,
      tasks: aTasks
    };
  }

  /**
   * Search GGUF models with various filters and options.
   *
   * Always expands the fields the discover rows render (chat template, context
   * length, siblings, ...) so a search result carries the same badges as a
   * catalog entry; caller-provided `expand` entries are merged in.
   */
  static async mSearch(pParams: HfModelSearchParams = {}): Promise<HfModelInfo[]> {
    if (!HuggingFaceService.mIsEnabled()) return [];

    const { expand: vExpand, limit: vLimit = cHfDefaultLimit, ...vRestParams } = pParams;
    const url = this.mBuildUrl({
      ...vRestParams,
      expand: [...new Set([...aHfModelListExpand, ...(vExpand ?? [])])],
      filter: cHfGgufFilter,
      limit: Math.min(vLimit, cHfMaxLimit)
    });

    return this.mFetchWithRetry(url);
  }

  static async mSearchByQuery(
    pQuery: string,
    pParams: Omit<HfModelSearchParams, 'search'> = {}
  ): Promise<HfModelInfo[]> {
    return this.mSearch({
      ...pParams,
      search: pQuery
    });
  }

  private static mBuildUrl(pParams: HfModelSearchParams): string {
    const url = new URL(this.vBaseUrl);

    Object.entries(pParams).forEach(([vKey, vValue]) => {
      if (vValue !== undefined && vValue !== null && vValue !== '') {
        if (Array.isArray(vValue)) {
          vValue.forEach((pV) => url.searchParams.append(vKey, pV));
        } else {
          url.searchParams.set(vKey, String(vValue));
        }
      }
    });

    return url.toString();
  }

  private static mDelay(pMs: number): Promise<void> {
    return new Promise((pResolve) => setTimeout(pResolve, pMs));
  }

  private static async mFetchWithRetry(pUrl: string, pAttempt: number = 1): Promise<HfModelInfo[]> {
    try {
      const cResponse = await fetch(pUrl);

      if (!cResponse.ok) {
        if (cResponse.status === cHfHttpNotFound) {
          return [];
        }

        if (cResponse.status >= cHfHttpServerErrorMin && pAttempt < cHfRetryAttempts) {
          await this.mDelay(cHfRetryDelayMs * pAttempt);

          return this.mFetchWithRetry(pUrl, pAttempt + 1);
        }

        throw new HfHttpStatusError(cResponse.status, cResponse.statusText);
      }

      const data = await cResponse.json();

      if (Array.isArray(data)) {
        return data as HfModelInfo[];
      }

      if (data && Array.isArray(data.data)) {
        return data.data as HfModelInfo[];
      }

      throw new Error(fTranslate('dgs_d8765e7283ba'));
    } catch (error) {
      const cTransient =
        error instanceof TypeError ||
        (error instanceof HfHttpStatusError && error.status >= cHfHttpServerErrorMin);

      if (cTransient && pAttempt < cHfRetryAttempts) {
        await this.mDelay(cHfRetryDelayMs * pAttempt);

        return this.mFetchWithRetry(pUrl, pAttempt + 1);
      }

      throw error;
    }
  }

  // Internal Methods

  /** Extract the `rel="next"` URL from an RFC 5988 `Link` header, if present. */
  private static mParseNextPageUrl(pLinkHeader: string | null): string | null {
    if (!pLinkHeader) return null;

    const cMatch = cHfLinkNextRegex.exec(pLinkHeader);

    return cMatch ? cMatch[1] : null;
  }

  /** Full path of one shard in a split-shard GGUF set. */
  private static mShardPath(pStem: string, pIndex: number, pTotal: number): string {
    const fPad = (pN: number) => String(pN).padStart(cHfShardPadWidth, '0');

    return `${pStem}-${fPad(pIndex)}-of-${fPad(pTotal)}.gguf`;
  }

  /** Strip a leading YAML frontmatter block (--- ... ---) from a markdown document. */
  private static mStripFrontmatter(pText: string): string {
    const cMatch = pText.match(cHfFrontmatterRegex);

    return cMatch ? pText.slice(cMatch[0].length) : pText;
  }
}
