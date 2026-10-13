import { fTranslate } from '$lib/i18n';
import { SharedPreferencesService } from '$lib/services/shared-preferences.service';
/**
 * modelsStore - Model management for MODEL and ROUTER modes
 *
 * Owns model lists, selection, favorites and load/unload state. Composes the
 * per-model props cache (modalities, thinking detection) as
 * {@link ModelsStore.props} and the /models/sse status feed as
 * {@link ModelsStore.status}; tracks which conversations use which models.
 */

import { browser } from '$app/environment';
import {
  cFavoriteModelsLocalstorageKey,
  cHiddenModelsLocalstorageKey,
  cModelGroupOpenLocalstorageKey,
  cRecentModelLimit,
  cRecentModelsLocalstorageKey
} from '$lib/constants';
import { ServerModelStatus } from '$lib/enums';
import { HuggingFaceService } from '$lib/services/huggingface.service';
import { ModelsService } from '$lib/services/models.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cConversationsStore } from '$lib/stores/conversations/index.svelte';
import { type ModelPropsHost, ModelPropsManager } from '$lib/stores/models/props.svelte';
import { type ModelStatusHost, ModelStatusManager } from '$lib/stores/models/status.svelte';
import { cServerStore } from '$lib/stores/server.svelte';
import type { ModelSidecarBadge, ParsedModelId } from '$lib/types/models';
import { fGetConversationModel } from '$lib/utils/conversation-utils';
import { fRepoOf } from '$lib/utils/model-names';
import { fIsAuxSidecar } from '$lib/utils/sidecars';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

/** Group open states kept before the oldest ones fall off; the map only ever grows. */
const cMaxGroupOpenEntries = 200;
/**
 * Repos whose Hub details are prefetched on load. The Hub rate limits an
 * unauthenticated client aggressively, so the warm slice stays far below the
 * list window and the rest of the rows fetch as they scroll near the viewport.
 */
const cHubWarmLimit = 12;

/** Union of the draft sidecar badges two sources report for one repo. */
function fMergedDraftSidecars(
  pFromListing: ModelSidecarBadge[] | undefined,
  pFromArgs: ModelSidecarBadge[]
): ModelSidecarBadge[] {
  if (!pFromListing || pFromListing.length === 0) return pFromArgs;

  if (pFromArgs.length === 0) return pFromListing;

  const aMerged = [...pFromListing];

  for (const badge of pFromArgs) {
    if (!aMerged.some((pB) => pB.kind === badge.kind && pB.repo === badge.repo))
      aMerged.push(badge);
  }

  return aMerged;
}

/** Models kept out of the selector. */
function fLoadHiddenModels(): Set<string> {
  if (!browser) return new SvelteSet<string>();

  try {
    const cRaw = SharedPreferencesService.mGetItem(cHiddenModelsLocalstorageKey);

    return cRaw ? new SvelteSet(JSON.parse(cRaw) as string[]) : new SvelteSet<string>();
  } catch {
    return new SvelteSet<string>();
  }
}

/** Open state the user set for a section or family of the model lists. */
function fLoadGroupOpenState(): SvelteMap<string, boolean> {
  if (!browser) return new SvelteMap<string, boolean>();

  try {
    const cRaw = SharedPreferencesService.mGetItem(cModelGroupOpenLocalstorageKey);
    const cParsed = cRaw ? (JSON.parse(cRaw) as unknown) : null;
    const entries =
      cParsed && typeof cParsed === 'object'
        ? Object.entries(cParsed as Record<string, unknown>)
        : [];

    return new SvelteMap(
      entries.filter((pEntry): pEntry is [string, boolean] => typeof pEntry[1] === 'boolean')
    );
  } catch {
    return new SvelteMap<string, boolean>();
  }
}

/** Recently used backend-qualified ids, most recent first. */
function fLoadRecentModels(): string[] {
  if (!browser) return [];

  try {
    const cRaw = SharedPreferencesService.mGetItem(cRecentModelsLocalstorageKey);

    if (!cRaw) return [];

    const cParsed = JSON.parse(cRaw) as unknown;

    return Array.isArray(cParsed)
      ? cParsed.filter((pId): pId is string => typeof pId === 'string').slice(0, cRecentModelLimit)
      : [];
  } catch {
    return [];
  }
}

class ModelsStore implements ModelPropsHost, ModelStatusHost {
  mReloadSharedSettings(): void {
    this.sFavoriteModelIds = this.mLoadFavoritesFromStorage();
    this.vGroupOpenState = fLoadGroupOpenState();
    this.sHiddenModelIds = fLoadHiddenModels();
    this.aRecentModelIds = fLoadRecentModels();
  }
  error = $state<string | null>(null);
  sFavoriteModelIds = $state<Set<string>>(this.mLoadFavoritesFromStorage());
  vGroupOpenState = $state<SvelteMap<string, boolean>>(fLoadGroupOpenState());
  sHiddenModelIds = $state<Set<string>>(fLoadHiddenModels());
  loading = $state(false);
  aModels = $state<ModelOption[]>([]);
  aRecentModelIds = $state<string[]>(fLoadRecentModels());
  aRouterModels = $state<ApiModelDataEntry[]>([]);
  vSelectedModelId = $state<string | null>(null);
  vSelectedModelName = $state<string | null>(null);

  vUpdating = $state(false);

  /** Per-model props cache, modalities and thinking detection, composed here. */
  private _props = new ModelPropsManager(this);

  /** Load/unload operations and the /models/sse status feed, composed here. */
  private _status = new ModelStatusManager(this);

  // every caller awaits the same inflight promise
  // Without this, ?model=<name> URL handler races an in-progress fetch and sees an empty list.
  private vInflightFetch: Promise<void> | null = null;

  /**
   * Model the active conversation view resolves to. Router mode: the user's
   * selection first, then the conversation's own model. Otherwise the single
   * served model, from the models list or the server props as a fallback.
   */
  get activeModelId(): string | null {
    if (!cServerStore.isRouterMode) {
      return this.aModels.length > 0 ? this.aModels[0].model : this.singleModelName;
    }

    if (this.vSelectedModelId) {
      const cSelected = this.aModels.find((pM) => pM.id === this.vSelectedModelId);

      if (cSelected) return cSelected.model;
    }

    const cConversationModel = fGetConversationModel(cConversationsStore.aActiveMessages);

    if (cConversationModel) {
      const model = this.aModels.find((pM) => pM.model === cConversationModel);

      if (model) return model.model;
    }

    return null;
  }

  get loadedModelIds(): string[] {
    return this.aRouterModels
      .filter(
        (pM) =>
          pM.status.value === ServerModelStatus.cLoaded ||
          pM.status.value === ServerModelStatus.cSleeping
      )
      .map((pM) => pM.id);
  }

  get props() {
    return this._props;
  }

  get selectedModel(): ModelOption | null {
    if (!this.vSelectedModelId) return null;

    return this.aModels.find((pM) => pM.id === this.vSelectedModelId) ?? null;
  }

  get selectedModelContextSize(): number | null {
    if (!this.vSelectedModelName) return null;

    return this.props.mGetModelContextSize(this.vSelectedModelName);
  }

  /**
   * Get model name in MODEL mode (single model).
   * Extracts from model_path or model_alias from server props.
   * In ROUTER mode, returns null (model is per-conversation).
   */
  get singleModelName(): string | null {
    if (cServerStore.isRouterMode) return null;

    const cProps = cServerStore.props;

    if (cProps?.model_alias) return cProps.model_alias;

    if (!cProps?.model_path) return null;

    return cProps.model_path.split(/(\\|\/)/).pop() || null;
  }

  get status() {
    return this._status;
  }

  mClearSelection(): void {
    this.vSelectedModelId = null;
    this.vSelectedModelName = null;
  }

  /** Family keys folded away under one section, for a list that restores them. */
  mCollapsedGroupsUnder(pPrefix: string): string[] {
    const cHead = `${pPrefix}-`;

    return [...this.vGroupOpenState]
      .filter(([vId, vOpen]) => !vOpen && vId.startsWith(cHead))
      .map(([vId]) => vId.slice(cHead.length));
  }

  /**
   * Auto-selects the first available model if none is selected.
   * Prioritizes:
   * 1. Model from active conversation's last assistant response (if loaded)
   * 2. Model from active conversation's last assistant response (if not loaded)
   * 3. First loaded model (not from active conversation)
   * 4. A favorite model
   * 5. First available model
   */
  async mEnsureFirstModelSelected(): Promise<void> {
    if (this.vSelectedModelName) return;

    const aAvailableModels = this.mGetVisibleModels();

    if (aAvailableModels.length === 0) return;

    // Try to select model from last assistant response first
    const cLastModel = this.mGetModelFromLastAssistantResponse();

    if (cLastModel) {
      const cLastModelOption = aAvailableModels.find((pM) => pM.model === cLastModel);

      if (cLastModelOption) {
        await this.mSelectModelById(cLastModelOption.id);

        if (this.mIsModelLoaded(cLastModel)) {
          await this.props.mFetchModelProps(cLastModel);
        }

        return;
      }
    }

    // Try a loaded model first
    const cLoadedModel = aAvailableModels.find((pM) => this.mIsModelLoaded(pM.model));

    if (cLoadedModel) {
      await this.mSelectModelById(cLoadedModel.id);
      await this.props.mFetchModelProps(cLoadedModel.model);

      return;
    }

    // favorites are shared across backends, so a stored id may belong to another one
    const cFavorite = this.sFavoriteModelIds.values().next()?.value;
    const cFavoriteOption = cFavorite
      ? aAvailableModels.find((pM) => pM.id === cFavorite || pM.model === cFavorite)
      : undefined;

    if (cFavoriteOption) {
      await this.mSelectModelById(cFavoriteOption.id);

      return;
    }

    // Fall back to the first available model
    await this.mSelectModelById(aAvailableModels[0].id);
  }

  /**
   * Fetch list of models from server and detect server role.
   * Also fetches modalities for MODEL mode (single model).
   */
  async mFetch(pForce = false): Promise<void> {
    if (this.vInflightFetch) return this.vInflightFetch;

    if (this.aModels.length > 0 && !pForce) return;

    this.vInflightFetch = this.mRunFetch();
    try {
      await this.vInflightFetch;
    } finally {
      this.vInflightFetch = null;
    }
  }

  /**
   * Fetch models with full metadata (ROUTER mode only).
   * No-op in MODEL mode - fetch() already calls list() internally.
   * Kept for API compatibility (e.g. handleOpenChange dropdown open handler).
   */
  async mFetchRouterModels(): Promise<void> {
    if (!cServerStore.isRouterMode) return;

    try {
      const cResponse = await ModelsService.mList();

      this.aRouterModels = cResponse.data;
      // keep the selector options in sync: a downloaded / deleted model shows
      // up here too, not only in the router model rows
      this.aModels = this.mBuildModelOptions(cResponse);
      this.mWarmHubDetails();
      await this.props.mFetchModalitiesForLoadedModels();

      const aVisible = this.mGetVisibleModels();

      if (aVisible.length === 1 && this.mIsModelLoaded(aVisible[0].model)) {
        this.mSelectModelById(aVisible[0].id);
      }
    } catch (error) {
      console.warn('Failed to fetch router models:', error);
      this.aRouterModels = [];
    }
  }

  mFindModelById(pModelId: string): ModelOption | null {
    return this.aModels.find((pModel) => pModel.id === pModelId) ?? null;
  }

  mFindModelByName(pModelName: string): ModelOption | null {
    return (
      this.aModels.find(
        (pModel) =>
          pModel.model === pModelName ||
          pModel.id === pModelName ||
          pModel.aliases?.includes(pModelName)
      ) ?? null
    );
  }

  /**
   * Gets the model name from the last assistant message in the active conversation.
   * Used by both the chat page and settings page to maintain model consistency.
   */
  mGetModelFromLastAssistantResponse(): string | null {
    const aMessages = cConversationsStore.aActiveMessages;

    if (!aMessages || aMessages.length === 0) return null;

    for (let vI = aMessages.length - 1; vI >= 0; vI--) {
      if (aMessages[vI].model) {
        return aMessages[vI].model;
      }
    }

    return null;
  }

  mGetModelStatus(pModelId: string): ServerModelStatus | null {
    const model = this.aRouterModels.find((pM) => pM.id === pModelId);

    return (model?.status?.value as ServerModelStatus) ?? null;
  }

  mHasModel(pModelName: string): boolean {
    return this.aModels.some((pModel) => pModel.model === pModelName);
  }

  mIsFavorite(pModelId: string): boolean {
    return this.sFavoriteModelIds.has(pModelId);
  }

  mIsGroupOpen(pId: string, pFallbackOpen: boolean): boolean {
    return this.vGroupOpenState.get(pId) ?? pFallbackOpen;
  }

  mIsHidden(pModelId: string): boolean {
    return this.sHiddenModelIds.has(pModelId);
  }

  mIsModelLoaded(pModelId: string): boolean {
    const cStatus = this.mGetModelStatus(pModelId);

    return cStatus === ServerModelStatus.cLoaded || cStatus === ServerModelStatus.cSleeping;
  }

  /** Loaded, or sleeping, and not mid-operation: what a row shows as running. */
  mIsModelRunning(pModelId: string): boolean {
    return this.mIsModelLoaded(pModelId) && !this._status.mIsOperationInProgress(pModelId);
  }

  /**
   * Select a model. `recordRecent` keeps automatic picks out of the recency list.
   */
  async mSelectModelById(pModelId: string, pOptions?: { recordRecent?: boolean }): Promise<void> {
    if (!pModelId || this.vUpdating) return;

    if (this.vSelectedModelId === pModelId) {
      if (pOptions?.recordRecent) this.mRecordRecentModel(pModelId);

      return;
    }

    const cOption = this.aModels.find((pModel) => pModel.id === pModelId);

    if (!cOption) throw new Error(fTranslate('text_1861f766e6f3'));

    this.vUpdating = true;
    this.error = null;

    try {
      this.vSelectedModelId = cOption.id;
      this.vSelectedModelName = cOption.model;

      if (pOptions?.recordRecent) this.mRecordRecentModel(pModelId);
    } finally {
      this.vUpdating = false;
    }
  }

  /**
   * Select a model by its model name (used for syncing with conversation model).
   */
  mSelectModelByName(pModelName: string): void {
    const cOption = this.aModels.find((pModel) => pModel.model === pModelName);

    if (cOption) {
      // the updating guard inside selectModelById cannot refuse this call: that
      // call is the only one that raises the flag, and it clears it before it returns
      void this.mSelectModelById(cOption.id);
    }
  }

  /**
   * Auto-selects the model from the last assistant response if available and loaded.
   * Returns true if a model was selected, false otherwise.
   */
  async mSelectModelFromLastAssistantResponse(): Promise<boolean> {
    const cLastModel = this.mGetModelFromLastAssistantResponse();

    if (!cLastModel || this.vSelectedModelName === cLastModel) return false;

    const cMatchingModel = this.aModels.find((pOption) => pOption.model === cLastModel);

    if (!cMatchingModel || !this.mIsModelLoaded(cLastModel)) return false;

    try {
      await this.mSelectModelById(cMatchingModel.id);
      console.log(`[modelsStore] Automatically selected model: ${cLastModel} from last message`);

      return true;
    } catch (error) {
      console.warn('[modelsStore] Failed to automatically select model from last message:', error);

      return false;
    }
  }

  /** Add or remove several models at once, e.g. every quant of a family. */
  mSetFavorites(pModelIds: string[], pFavorite: boolean): void {
    const cNext = new SvelteSet(this.sFavoriteModelIds);

    for (const modelId of pModelIds) {
      if (pFavorite) {
        cNext.add(modelId);
      } else {
        cNext.delete(modelId);
      }
    }

    this.sFavoriteModelIds = cNext;

    try {
      SharedPreferencesService.mSetItem(cFavoriteModelsLocalstorageKey, JSON.stringify([...cNext]));
    } catch {
      toast.error(fTranslate('text_3b683426d98c'));
    }
  }

  /** Persist one family of a section, e.g. a folded `Qwen` under the local models. */
  mSetGroupCollapsed(pPrefix: string, pKey: string, pCollapsed: boolean): void {
    this.mSetGroupOpen(`${pPrefix}-${pKey}`, !pCollapsed);
  }

  mSetGroupOpen(pId: string, pOpen: boolean): void {
    const cNext = new SvelteMap(this.vGroupOpenState);

    // re-set moves the id to the end, so the oldest states fall off first
    cNext.delete(pId);
    cNext.set(pId, pOpen);

    while (cNext.size > cMaxGroupOpenEntries) {
      const cOldest = cNext.keys().next();

      if (cOldest.done) break;

      cNext.delete(cOldest.value);
    }

    this.vGroupOpenState = cNext;

    try {
      SharedPreferencesService.mSetItem(
        cModelGroupOpenLocalstorageKey,
        JSON.stringify(Object.fromEntries(cNext))
      );
    } catch {
      toast.error(fTranslate('dgs_c8ad88d0a0a7'));
    }
  }

  mToDisplayName(pId: string): string {
    const aSegments = pId.split(/\\|\//);
    const cCandidate = aSegments.pop();

    return cCandidate && cCandidate.trim().length > 0 ? cCandidate : pId;
  }

  mToggleFavorite(pModelId: string): void {
    this.mSetFavorites([pModelId], !this.sFavoriteModelIds.has(pModelId));
  }

  /** Models hidden from the selector stay in the manager, flagged and unhideable. */
  mToggleHidden(pModelId: string): void {
    const cNext = new SvelteSet(this.sHiddenModelIds);

    if (cNext.has(pModelId)) {
      cNext.delete(pModelId);
    } else {
      cNext.add(pModelId);
    }

    this.sHiddenModelIds = cNext;

    try {
      SharedPreferencesService.mSetItem(cHiddenModelsLocalstorageKey, JSON.stringify([...cNext]));
    } catch {
      toast.error(fTranslate('dgs_0b2332518b76'));
    }
  }

  /**
   * Warm the Hub record of the local repos, so a list can read the context and size
   * the server reports only once a model is loaded.
   */
  mWarmHubDetails(): void {
    if (!HuggingFaceService.mIsEnabled()) return;

    const aRepos: string[] = [];

    for (const cOption of this.aModels) {
      const cRepo = fRepoOf(cOption.model);

      if (cRepo?.includes('/') && !aRepos.includes(cRepo)) aRepos.push(cRepo);
    }

    // the Hub rate limits aggressively and an unauthenticated burst of requests
    // leaves every detail uncached for a retry on the next mount, so warm only the
    // rows the lists show at rest and let the rest arrive near the viewport
    for (const cRepo of aRepos.slice(0, cHubWarmLimit)) {
      void HuggingFaceService.mGetDetails(cRepo).catch(() => {});
    }
  }

  /**
   * Build ModelOption[] from an API response.
   * Both MODEL and ROUTER modes share the same mapping logic;
   * they differ only in which endpoint is called.
   */
  private mBuildModelOptions(pResponse: ApiModelsListResponse): ModelOption[] {
    const entries: {
      details?: ApiModelsListResponse['models'][number];
      item: ApiModelDataEntry;
      parsed: ParsedModelId;
    }[] = pResponse.data.map((pItem: ApiModelDataEntry, pIndex: number) => ({
      details: pResponse.models?.[pIndex],
      item: pItem,
      parsed: ModelsService.mParseModelId(pItem.id)
    }));
    // sidecar entries mark downloaded sidecar files, not loadable models, and the router
    // lists them under their own `<repo>:<quant>-<sidecar>` id: pair each draft with its
    // repo here and drop the entries below
    const cDraftSidecarsByRepo = new SvelteMap<string, ModelSidecarBadge[]>();

    for (const { item: vItem, parsed: vParsed } of entries) {
      if (!vParsed.sidecar || fIsAuxSidecar(vParsed.sidecar)) continue;

      const cRepo = fRepoOf(vItem.id);
      // the entry id's quant is the sidecar file's own quant
      const badge: ModelSidecarBadge = {
        kind: vParsed.sidecar,
        quant: vParsed.quantization,
        repo: cRepo
      };
      const cKinds = cDraftSidecarsByRepo.get(cRepo);

      if (cKinds) {
        if (!cKinds.some((pB) => pB.kind === badge.kind && pB.repo === badge.repo)) {
          cKinds.push(badge);
        }
      } else {
        cDraftSidecarsByRepo.set(cRepo, [badge]);
      }
    }

    return (
      entries
        // the router also lists projector files by filename, which carry the
        // sidecar token inside the id rather than after a colon
        .filter(
          ({ item: vItem, parsed: vParsed }) =>
            !vParsed.sidecar && !ModelsService.mIsSidecarEntry(vItem.id)
        )
        // in-flight downloads are not usable models yet; the selector tracks
        // them in its "Download in progress" section instead
        .filter(({ item: vItem }) => vItem.status?.value !== ServerModelStatus.cDownloading)
        .map(({ details: vDetails, item: vItem }) => {
          const cRawCapabilities = Array.isArray(vDetails?.capabilities)
            ? vDetails?.capabilities
            : [];
          const cDisplayNameSource =
            vDetails?.name && vDetails.name.trim().length > 0 ? vDetails.name : vItem.id;
          const modelId = vDetails?.model || vItem.id;
          const cRepo = fRepoOf(modelId);

          return {
            aliases: vItem.aliases ?? [],
            capabilities: cRawCapabilities.filter((pValue: unknown): pValue is string =>
              Boolean(pValue)
            ),
            // 0 is the server's way of leaving the trained context unknown; a
            // model-mode listing reports it as meta.n_ctx_train instead
            contextLength:
              vItem.context_length ||
              (typeof vItem.meta?.n_ctx_train === 'number' && vItem.meta.n_ctx_train > 0
                ? vItem.meta.n_ctx_train
                : undefined),
            description: vDetails?.description,
            details: vDetails?.details,
            draftSidecars: fMergedDraftSidecars(
              cDraftSidecarsByRepo.get(cRepo),
              // models the router serves with a draft record it in their
              // --model-draft args instead of a listing entry
              this._status.mGetDraftSidecars(cRepo)
            ),
            id: vItem.id,
            meta: vItem.meta ?? null,
            modalities: this.props.mBuildArchitectureModalities(vItem.architecture),
            model: modelId,
            name: this.mToDisplayName(cDisplayNameSource),
            parsedId: ModelsService.mParseModelId(modelId),
            tags: vItem.tags ?? []
          };
        })
    );
  }

  /** Fetch models in MODEL mode (single model, standard OpenAI-compatible). */
  private async mFetchModelModeInternal(): Promise<ModelOption[]> {
    const cResponse = await ModelsService.mList();

    return this.mBuildModelOptions(cResponse);
  }
  /**
   * Filter to models visible in the UI (ui !== false).
   */
  private mGetVisibleModels(): ModelOption[] {
    return this.aModels.filter((pOption) => this.props.mGetModelProps(pOption.model)?.ui !== false);
  }

  private mLoadFavoritesFromStorage(): Set<string> {
    try {
      const cRaw = SharedPreferencesService.mGetItem(cFavoriteModelsLocalstorageKey);

      return cRaw ? new Set(JSON.parse(cRaw) as string[]) : new Set();
    } catch {
      toast.error(fTranslate('text_38e39f3f3976'));

      return new Set();
    }
  }

  /** Move a model to the front of the recently used list. */
  private mRecordRecentModel(pQualifiedId: string): void {
    this.aRecentModelIds = [
      pQualifiedId,
      ...this.aRecentModelIds.filter((pId) => pId !== pQualifiedId)
    ].slice(0, cRecentModelLimit);

    if (!browser) return;

    try {
      SharedPreferencesService.mSetItem(
        cRecentModelsLocalstorageKey,
        JSON.stringify(this.aRecentModelIds)
      );
    } catch {
      console.warn('[ModelsStore] Failed to persist the recently used models');
    }
  }

  private async mRunFetch(): Promise<void> {
    this.loading = true;
    this.error = null;

    try {
      if (!cServerStore.props) {
        await cServerStore.mFetch();
      }

      const router = cServerStore.isRouterMode;

      if (router) {
        const cResponse = await ModelsService.mList();

        this.aRouterModels = cResponse.data;
        this.aModels = this.mBuildModelOptions(cResponse);

        await this.props.mFetchModalitiesForLoadedModels();

        const aVisible = this.mGetVisibleModels();

        if (aVisible.length === 1 && this.mIsModelLoaded(aVisible[0].model)) {
          this.mSelectModelById(aVisible[0].id);
        }
      } else {
        this.aModels = await this.mFetchModelModeInternal();
      }
    } catch (error) {
      this.aModels = [];
      this.error = error instanceof Error ? error.message : 'Failed to load models';

      throw error;
    } finally {
      this.loading = false;
    }
  }
}

export const cModelsStore = new ModelsStore();
