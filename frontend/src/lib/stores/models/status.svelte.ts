import { fTranslate } from '$lib/i18n';
/**
 * ModelStatusManager - model load/unload operations and the /models/sse feed
 *
 * The feed drives status and progress, replacing any post-operation polling.
 * Created and owned by modelsStore, which also owns the router rows it updates.
 */

import {
  cCliFlags,
  cHfUdQuantPrefixRegex,
  MODEL_ID,
  cPathSeparator,
  cPausedModelDownloadsLocalstorageKey
} from '$lib/constants';
import { ModelDownloadStopRequest, ServerModelsSseEventType, ServerModelStatus } from '$lib/enums';
import { HuggingFaceService } from '$lib/services/huggingface.service';
import { ModelsService } from '$lib/services/models.service';
import type { ModelPropsManager } from '$lib/stores/models/props.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cServerStore } from '$lib/stores/server.svelte';
// explicit type imports: the app.d.ts globals resolve to `any`, so import the real types
import type {
  ApiModelsSseDownloadProgressData,
  ModelDownloadEntry,
  ModelDownloadProgress
} from '$lib/types';
import type { ModelSidecarBadge } from '$lib/types/models';
import { fRepoOf } from '$lib/utils/model-names';
import { fIsAuxSidecar } from '$lib/utils/sidecars';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

/**
 * The slice of modelsStore the manager drives. Kept narrow on purpose so it
 * cannot reach around the host's full surface; modelsStore implements this
 * structurally.
 */
export interface ModelStatusHost {
  error: string | null;
  readonly props: ModelPropsManager;
  /** Router model rows the status feed updates. */
  aRouterModels: ApiModelDataEntry[];
  mFetchRouterModels(): Promise<void>;
  mIsModelLoaded(pModelId: string): boolean;
  mToDisplayName(pId: string): string;
}

/**
 * Comparison key of a `<repo>:<tag>` download identifier: uppercased, with the
 * `UD-` quant prefix stripped. The router derives cached model names from the
 * actual file, which drops the prefix, so `repo:UD-Q4_K_XL` and `repo:Q4_K_XL`
 * must compare equal.
 */
function fDownloadIdKey(pRepoWithTag: string): string {
  const cIdx = pRepoWithTag.indexOf(MODEL_ID.QUANTIZATION_SEPARATOR);
  const cRepo = cIdx === -1 ? pRepoWithTag : pRepoWithTag.slice(0, cIdx);
  const cTag = cIdx === -1 ? '' : pRepoWithTag.slice(cIdx + 1);

  return `${cRepo.toUpperCase()}:${cTag.toUpperCase().replace(cHfUdQuantPrefixRegex, '')}`;
}

export class ModelStatusManager {
  /**
   * Sidecar files pulled by registered models, as `<repo>/<file>` keys.
   * Sidecars are not separate /v1/models entries - the router pulls them as
   * sidecars of a main model and records them in its `--model-draft` /
   * `--mmproj` args.
   */
  private vDownloadedSidecars = $derived.by(() => {
    const cResult = new SvelteSet<string>();

    for (const cM of this.pHost.aRouterModels) {
      const cArgs = cM.status?.args;

      if (!cArgs) continue;

      for (let vI = 0; vI < cArgs.length - 1; vI++) {
        if (
          cArgs[vI] !== cCliFlags.MODEL_DRAFT &&
          cArgs[vI] !== cCliFlags.MODEL_DRAFT_SHORT &&
          cArgs[vI] !== cCliFlags.MMPROJ
        ) {
          continue;
        }

        const cParsed = HuggingFaceService.mParseCachePath(cArgs[vI + 1]);

        if (cParsed) cResult.add(`${cParsed.repo}${cPathSeparator}${cParsed.file}`);
      }
    }

    return cResult;
  });

  private vDownloadProgress = new SvelteMap<string, ModelDownloadProgress>();
  /**
   * Draft sidecars the registered models pull with them, keyed by the repo of the model
   * that pulls them: the router records them in the model's `--model-draft` args.
   */
  private vDraftSidecarsByRepo = $derived.by(() => {
    const cResult = new SvelteMap<string, ModelSidecarBadge[]>();

    for (const cM of this.pHost.aRouterModels) {
      const cArgs = cM.status?.args;

      if (!cArgs) continue;

      const cBaseRepo = fRepoOf(cM.id);

      for (let vI = 0; vI < cArgs.length - 1; vI++) {
        if (cArgs[vI] !== cCliFlags.MODEL_DRAFT && cArgs[vI] !== cCliFlags.MODEL_DRAFT_SHORT) {
          continue;
        }

        const cParsed = HuggingFaceService.mParseCachePath(cArgs[vI + 1]);

        if (!cParsed) continue;

        const cMeta = HuggingFaceService.mExtractQuantMeta(cParsed.file);

        if (!cMeta?.sidecar || fIsAuxSidecar(cMeta.sidecar)) continue;

        const badge: ModelSidecarBadge = {
          kind: cMeta.sidecar,
          quant: cMeta.quant,
          repo: cParsed.repo
        };
        const cKinds = cResult.get(cBaseRepo);

        if (cKinds) {
          if (!cKinds.some((pB) => pB.kind === badge.kind && pB.repo === badge.repo)) {
            cKinds.push(badge);
          }
        } else {
          cResult.set(cBaseRepo, [badge]);
        }
      }
    }

    return cResult;
  });
  /** `<repo>:<tag>` strings whose most recent download attempt failed (download_failed). */
  private vFailedDownloads = new SvelteSet<string>();
  private vLoadingStates = new SvelteMap<string, boolean>();
  private vLoadProgress = new SvelteMap<string, ModelLoadProgress>();
  /** Paused downloads with their last reported progress, or null when none arrived before the pause. */
  private vPausedDownloads = new SvelteMap<string, ModelDownloadProgress | null>();
  // /models/sse feed state, the single source of truth for status and load progress
  private vStatusAbort: AbortController | null = null;
  private vStatusReaderActive = false;
  private vStatusWaiters = new SvelteMap<
    string,
    { target: ServerModelStatus; resolve: () => void; reject: (pE: Error) => void }
  >();
  /** Tags the user asked to stop (pause or cancel); the download_failed the stop triggers is intentional, not a failure. */
  private vStopRequests = new SvelteMap<string, ModelDownloadStopRequest>();

  /**
   * Cancel an in-flight download, or remove a downloaded/failed entry from the
   * cache (ROUTER mode only). The server's remove endpoint covers both: dropping
   * a running download discards its partial files, dropping a cached model
   * deletes the files on disk, so the delete action resolves through this too.
   */
  async mCancelDownload(pRepoWithTag: string): Promise<boolean> {
    if (!cServerStore.isRouterMode) {
      toast.error(fTranslate('dgs_7db82a0d74d0'));

      return false;
    }

    this.mSubscribe();

    // in-flight: the kill triggers download_failed over the feed; mark it as a
    // user cancel so it settles silently instead of toasting a failure
    if (this.vDownloadProgress.has(pRepoWithTag)) {
      this.vStopRequests.set(pRepoWithTag, ModelDownloadStopRequest.cCancel);
    }

    // a downloaded model registers under the name the router derived from the
    // cached file (e.g. the UD- quant prefix is dropped), so resolve the tag to
    // the registered id before asking the server to remove it
    const cRegisteredId =
      this.pHost.aRouterModels.find((pM) => fDownloadIdKey(pM.id) === fDownloadIdKey(pRepoWithTag))
        ?.id ?? pRepoWithTag;

    try {
      const cRes = await ModelsService.mCancelDownload(cRegisteredId);
      const ok = cRes.success === true;

      if (ok) {
        this.vDownloadProgress.delete(pRepoWithTag);
        this.vFailedDownloads.delete(pRepoWithTag);
        this.mDeletePausedDownload(pRepoWithTag);
      }

      return ok;
    } catch (error) {
      toast.error(`Failed to cancel: ${error instanceof Error ? error.message : 'unknown error'}`);

      return false;
    }
  }

  /**
   * The server force-kills a LOADING model on unload, and the feed reports the
   * settled status, so no waiter is registered here.
   */
  async mCancelLoad(pModelId: string): Promise<void> {
    if (!cServerStore.isRouterMode) return;

    this.mSubscribe();

    try {
      await ModelsService.mUnload(pModelId);
      toast.info(`Load cancelled: ${this.pHost.mToDisplayName(pModelId)}`);
    } catch (error) {
      toast.error(`Failed to cancel load: ${this.pHost.mToDisplayName(pModelId)}`);

      throw error;
    }
  }

  constructor(private pHost: ModelStatusHost) {
    // the server has no notion of a paused download, so the ids survive in
    // localStorage; the progress snapshot is stale after a reload and stays null
    try {
      const cRaw = localStorage.getItem(cPausedModelDownloadsLocalstorageKey);
      const cParsed: unknown = JSON.parse(cRaw ?? '[]');

      if (!Array.isArray(cParsed)) return;

      for (const cRepoWithTag of cParsed.filter((pId): pId is string => typeof pId === 'string')) {
        this.vPausedDownloads.set(cRepoWithTag, null);
      }
    } catch {
      // unreadable or corrupt: start without the paused set
    }
  }

  /**
   * POST /models starts the download in the background; the feed reports
   * progress, and models_reload refreshes the list once it finishes.
   * Re-posting a paused tag resumes from the partial files kept on disk.
   */
  async mDownloadModel(pRepoWithTag: string): Promise<void> {
    if (!cServerStore.isRouterMode) {
      toast.error(fTranslate('dgs_7db82a0d74d0'));

      return;
    }

    // the feed must be live so the resulting models_reload event refreshes the list
    this.mSubscribe();

    // resuming a paused download: drop the paused state, and let the server
    // discard its stale DOWNLOADED entry (via the list fetch) before re-posting
    if (this.mDeletePausedDownload(pRepoWithTag) || this.vStopRequests.delete(pRepoWithTag)) {
      await this.pHost.mFetchRouterModels();
    }

    try {
      const cRes = await ModelsService.mDownloadModel(pRepoWithTag);

      if (!cRes.success) {
        throw new Error(cRes.error?.message ?? 'Server rejected the download request');
      }

      // flip the chip to "downloading" right away; the feed refines it with real progress
      this.vDownloadProgress.set(pRepoWithTag, { downloadedBytes: 0, files: {}, totalBytes: 0 });

      toast.success(`Download started: ${this.pHost.mToDisplayName(pRepoWithTag)}`);
    } catch (error) {
      toast.error(`Download failed: ${pRepoWithTag}`);

      throw error;
    }
  }

  async mEnsureLoaded(pModelId: string): Promise<void> {
    if (this.pHost.mIsModelLoaded(pModelId)) return;

    await this.load(pModelId);
  }

  /**
   * Tracked downloads (in flight or paused) for the selector's
   * "Download in progress" section.
   */
  mGetDownloadEntries(): ModelDownloadEntry[] {
    const aInFlight = Array.from(this.vDownloadProgress, ([vRepoWithTag, vProgress]) => ({
      isPaused: false,
      progress: vProgress,
      repoWithTag: vRepoWithTag
    }));
    const aPaused = Array.from(this.vPausedDownloads, ([vRepoWithTag, vProgress]) => ({
      isPaused: true,
      progress: vProgress,
      repoWithTag: vRepoWithTag
    }));

    return [...aInFlight, ...aPaused];
  }

  mGetDownloadProgress(pRepoWithTag: string): ModelDownloadProgress | null {
    return this.vDownloadProgress.get(pRepoWithTag) ?? null;
  }

  /** Draft sidecars a repo pulls with its model, e.g. `mtp` at `Q8_0`. */
  mGetDraftSidecars(pRepoId: string): ModelSidecarBadge[] {
    return this.vDraftSidecarsByRepo.get(pRepoId) ?? [];
  }

  mGetLoadProgress(pModelId: string): ModelLoadProgress | null {
    return this.vLoadProgress.get(pModelId) ?? null;
  }

  mGetPausedDownloadProgress(pRepoWithTag: string): ModelDownloadProgress | null {
    return this.vPausedDownloads.get(pRepoWithTag) ?? null;
  }

  mHasFailedDownload(pRepoWithTag: string): boolean {
    return this.vFailedDownloads.has(pRepoWithTag);
  }

  /** Active while the feed reports download_progress for the tag. */
  mIsDownloadInProgress(pRepoWithTag: string): boolean {
    return this.vDownloadProgress.has(pRepoWithTag);
  }

  mIsDownloadPaused(pRepoWithTag: string): boolean {
    return this.vPausedDownloads.has(pRepoWithTag);
  }

  /**
   * True when the tag is already registered in the /v1/models list; both ids
   * are normalized, see downloadIdKey().
   */
  mIsModelDownloaded(pRepoWithTag: string): boolean {
    const cKey = fDownloadIdKey(pRepoWithTag);

    return this.pHost.aRouterModels.some((pM) => fDownloadIdKey(pM.id) === cKey);
  }

  mIsOperationInProgress(pModelId: string): boolean {
    return this.vLoadingStates.get(pModelId) ?? false;
  }

  mIsSidecarDownloaded(pRepoId: string, pFilePath: string): boolean {
    return this.vDownloadedSidecars.has(`${pRepoId}/${pFilePath}`);
  }

  async load(pModelId: string): Promise<void> {
    if (this.pHost.mIsModelLoaded(pModelId)) return;

    if (this.vLoadingStates.get(pModelId)) return;

    this.vLoadingStates.set(pModelId, true);
    this.pHost.error = null;

    // the feed drives completion, so it must be live before the request
    this.mSubscribe();

    const cReachedLoaded = this.mWaitForStatus(pModelId, ServerModelStatus.cLoaded);

    cReachedLoaded.catch(() => {});

    try {
      await ModelsService.load(pModelId);
      await cReachedLoaded;
      toast.success(`Model loaded: ${this.pHost.mToDisplayName(pModelId)}`);
    } catch (error) {
      this.mRejectStatus(
        pModelId,
        error instanceof Error ? error : new Error(fTranslate('text_f318eb3d47fb'))
      );
      this.pHost.error = error instanceof Error ? error.message : 'Failed to load model';
      toast.error(`Failed to load model: ${this.pHost.mToDisplayName(pModelId)}`);

      throw error;
    } finally {
      this.vLoadingStates.set(pModelId, false);
    }
  }

  /**
   * The server stops the download child but keeps the partial files, so
   * re-posting the tag resumes where it stopped. The feed reports the stop
   * as download_failed; the 'pause' stop request marks it as intentional.
   */
  async mPauseDownload(pRepoWithTag: string): Promise<void> {
    if (!cServerStore.isRouterMode) {
      toast.error(fTranslate('dgs_7db82a0d74d0'));

      return;
    }

    this.mSubscribe();

    this.vStopRequests.set(pRepoWithTag, ModelDownloadStopRequest.cPause);

    try {
      await ModelsService.mUnload(pRepoWithTag);
    } catch {
      this.vStopRequests.delete(pRepoWithTag);
      toast.error(`Failed to pause: ${pRepoWithTag}`);
    }
  }

  /** Open the /models/sse feed with auto reconnect; idempotent, router mode only. */
  mSubscribe(): void {
    if (this.vStatusReaderActive) return;

    if (!cServerStore.isRouterMode) return;

    this.vStatusReaderActive = true;
    this.vStatusAbort = new AbortController();
    void this.mRunStatusReader(this.vStatusAbort.signal);
  }

  async mUnload(pModelId: string): Promise<void> {
    if (!this.pHost.mIsModelLoaded(pModelId)) return;

    if (this.vLoadingStates.get(pModelId)) return;

    this.vLoadingStates.set(pModelId, true);
    this.pHost.error = null;

    this.mSubscribe();

    const cReachedUnloaded = this.mWaitForStatus(pModelId, ServerModelStatus.cUnloaded);

    cReachedUnloaded.catch(() => {});

    try {
      await ModelsService.mUnload(pModelId);
      await cReachedUnloaded;
      toast.info(`Model unloaded: ${this.pHost.mToDisplayName(pModelId)}`);
    } catch (error) {
      this.mRejectStatus(
        pModelId,
        error instanceof Error ? error : new Error(fTranslate('text_2f7941bef771'))
      );
      this.pHost.error = error instanceof Error ? error.message : 'Failed to unload model';
      toast.error(`Failed to unload model: ${this.pHost.mToDisplayName(pModelId)}`);

      throw error;
    } finally {
      this.vLoadingStates.set(pModelId, false);
    }
  }

  mUnsubscribe(): void {
    this.vStatusReaderActive = false;
    this.vStatusAbort?.abort();
    this.vStatusAbort = null;
    this.vLoadProgress.clear();
    this.vDownloadProgress.clear();
    this.vFailedDownloads.clear();
    this.vStopRequests.clear();
  }

  /**
   * A user pause keeps the last progress and stays resumable, a user cancel
   * settles silently; genuine failures are marked so the UI can offer a retry.
   */
  private mApplyDownloadFinished(pEvent: ApiModelsSseEvent): void {
    let vRequest: ModelDownloadStopRequest | undefined;

    if (pEvent.event === ServerModelsSseEventType.cDownloadFailed) {
      vRequest = this.vStopRequests.get(pEvent.model);
      this.vStopRequests.delete(pEvent.model);
    }

    const cProgress = this.vDownloadProgress.get(pEvent.model) ?? null;

    this.vDownloadProgress.delete(pEvent.model);

    if (vRequest === ModelDownloadStopRequest.cCancel) {
      // user cancel: settle silently, the feed's model_remove cleans up the entry
      this.vFailedDownloads.delete(pEvent.model);
      this.mDeletePausedDownload(pEvent.model);

      return;
    }

    if (vRequest === ModelDownloadStopRequest.cPause) {
      this.mSetPausedDownload(pEvent.model, cProgress);
      this.vFailedDownloads.delete(pEvent.model);

      return;
    }

    this.mDeletePausedDownload(pEvent.model);

    const ok = pEvent.event === ServerModelsSseEventType.cDownloadFinished;

    if (ok) {
      this.vFailedDownloads.delete(pEvent.model);

      // the finished download only registers in /v1/models on the next list
      // fetch (the server reloads its model table then), so refetch to flip
      // the quant chips to "downloaded" without waiting for a dialog reopen
      void this.pHost.mFetchRouterModels();

      toast.success(`Download finished: ${this.pHost.mToDisplayName(pEvent.model)}`);
    } else {
      this.vFailedDownloads.add(pEvent.model);
      toast.error(`Download failed: ${this.pHost.mToDisplayName(pEvent.model)}`);
    }
  }

  /** Aggregate per-file progress into downloaded/total byte counts. */
  private mApplyDownloadProgress(pEvent: ApiModelsSseEvent): void {
    const data = pEvent.data;

    if (!data || !('progress' in data)) return;

    const dProgress = (data as ApiModelsSseDownloadProgressData).progress;

    let vDownloaded = 0;
    let vTotal = 0;

    for (const file of Object.values(dProgress)) {
      vDownloaded += file?.done ?? 0;
      vTotal += file?.total ?? 0;
    }

    this.vDownloadProgress.set(pEvent.model, {
      downloadedBytes: vDownloaded,
      files: dProgress,
      totalBytes: vTotal
    });
  }

  private mApplyModelStatus(pEvent: ApiModelsSseEvent): void {
    const model = pEvent.model;
    const data = pEvent.data;

    if (!model || !data || !('status' in data) || !data.status) return;

    const cStatus = data.status;

    this.mSetRouterModelStatus(model, cStatus);

    if (cStatus === ServerModelStatus.LOADING) {
      if (data.progress) this.vLoadProgress.set(model, data.progress);
    } else {
      this.vLoadProgress.delete(model);
    }

    if (cStatus === ServerModelStatus.cLoaded) {
      void this.pHost.props.mUpdateModelModalities(model);
    }

    const failed =
      cStatus === ServerModelStatus.FAILED ||
      (cStatus === ServerModelStatus.cUnloaded && (data.exit_code ?? 0) !== 0);

    if (failed) {
      this.mRejectStatus(model, new Error(`Model failed: ${this.pHost.mToDisplayName(model)}`));

      return;
    }

    this.mSettleStatus(model, cStatus);
  }

  /** Route one feed record by event kind. */
  private mApplyStatusEvent(pEvent: ApiModelsSseEvent): void {
    switch (pEvent.event) {
      case ServerModelsSseEventType.cStatusChange:
      case ServerModelsSseEventType.cModelStatus:
      case ServerModelsSseEventType.cStatusUpdate:
        this.mApplyModelStatus(pEvent);

        break;
      case ServerModelsSseEventType.cModelsReload:
        void this.pHost.mFetchRouterModels();

        break;
      case ServerModelsSseEventType.cModelRemove:
        this.mRemoveRouterModel(pEvent.model);

        break;
      case ServerModelsSseEventType.cDownloadProgress:
        this.mApplyDownloadProgress(pEvent);

        break;
      case ServerModelsSseEventType.cDownloadFinished:
      case ServerModelsSseEventType.cDownloadFailed:
        this.mApplyDownloadFinished(pEvent);

        break;
    }
  }

  private mDeletePausedDownload(pRepoWithTag: string): boolean {
    if (!this.vPausedDownloads.delete(pRepoWithTag)) return false;

    this.mPersistPausedDownloads();

    return true;
  }

  private mPersistPausedDownloads(): void {
    try {
      localStorage.setItem(
        cPausedModelDownloadsLocalstorageKey,
        JSON.stringify(Array.from(this.vPausedDownloads.keys()))
      );
    } catch {
      // storage unavailable: the pauses just do not survive a reload
    }
  }

  private mRejectStatus(pModelId: string, pError: Error): void {
    const cWaiter = this.vStatusWaiters.get(pModelId);

    if (cWaiter) {
      this.vStatusWaiters.delete(pModelId);
      cWaiter.reject(pError);
    }
  }

  private mRemoveRouterModel(pModelId: string): void {
    if (this.pHost.aRouterModels.findIndex((pM) => pM.id === pModelId) === -1) return;

    this.pHost.aRouterModels = this.pHost.aRouterModels.filter((pM) => pM.id !== pModelId);
    this.vLoadProgress.delete(pModelId);
    this.vDownloadProgress.delete(pModelId);
    this.vFailedDownloads.delete(pModelId);
    this.mDeletePausedDownload(pModelId);
    this.vStopRequests.delete(pModelId);
    this.mRejectStatus(
      pModelId,
      new Error(`Model removed: ${this.pHost.mToDisplayName(pModelId)}`)
    );

    // drop the row from the selector options too; they rebuild from the list
    // response, which only a refetch provides
    void this.pHost.mFetchRouterModels();
  }

  private async mRunStatusReader(pSignal: AbortSignal): Promise<void> {
    await ModelsService.mWatchModelEvents(pSignal, (pEvent) => this.mApplyStatusEvent(pEvent));
  }

  private mSetPausedDownload(pRepoWithTag: string, pProgress: ModelDownloadProgress | null): void {
    this.vPausedDownloads.set(pRepoWithTag, pProgress);
    this.mPersistPausedDownloads();
  }

  // reassign the array: mutating an entry in place would not trigger reactivity
  private mSetRouterModelStatus(pModelId: string, pStatus: ServerModelStatus): void {
    const cIdx = this.pHost.aRouterModels.findIndex((pM) => pM.id === pModelId);

    if (cIdx === -1) return;

    const cCurrent = this.pHost.aRouterModels[cIdx];

    if (cCurrent.status.value === pStatus) return;

    const aNext = [...this.pHost.aRouterModels];

    aNext[cIdx] = { ...cCurrent, status: { ...cCurrent.status, value: pStatus } };
    this.pHost.aRouterModels = aNext;
  }

  private mSettleStatus(pModelId: string, pStatus: ServerModelStatus): void {
    const cWaiter = this.vStatusWaiters.get(pModelId);

    if (cWaiter && cWaiter.target === pStatus) {
      this.vStatusWaiters.delete(pModelId);
      cWaiter.resolve();
    }
  }

  // one operation runs per model at a time, so one waiter per model suffices
  private mWaitForStatus(pModelId: string, pTarget: ServerModelStatus): Promise<void> {
    return new Promise((pResolve, pReject) => {
      this.vStatusWaiters.set(pModelId, { reject: pReject, resolve: pResolve, target: pTarget });
    });
  }
}
