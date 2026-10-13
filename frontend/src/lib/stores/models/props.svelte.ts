/**
 * ModelPropsManager - Per-model props cache, modalities and thinking detection
 *
 * Owns the /props?model=<id> cache with TTL, the modality views over it,
 * and chat-template thinking detection. Created and owned by modelsStore;
 * the host owns the model lists that fetched modalities are mirrored onto.
 *
 * **API Inconsistency Workaround:**
 * In MODEL mode, `/props` returns modalities for the single model.
 * In ROUTER mode, `/props` has no modalities - must use `/props?model=<id>` per model.
 */

import { cModelPropsCache } from '$lib/constants';
import { FileTypeCategory, ModelModality } from '$lib/enums';
import { PropsService } from '$lib/services/props.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cServerStore } from '$lib/stores/server.svelte';
// deep imports, not the '$lib/utils' barrel: it re-exports modules that reach back
// into the stores, and going through it here would read a half-built module
import { TTLCache } from '$lib/utils/cache-ttl';
import { fDetectThinkingSupport } from '$lib/utils/chat-template-thinking-detector';
import { SvelteSet } from 'svelte/reactivity';

/**
 * The slice of modelsStore the manager reads. Kept narrow on purpose so it
 * cannot reach around the host's full surface; modelsStore implements this
 * structurally.
 */
export interface ModelPropsHost {
  /** Model rows the manager mirrors fetched modalities onto. */
  aModels: ModelOption[];
  readonly vSelectedModelName: string | null;
  readonly loadedModelIds: string[];
  mIsModelLoaded(pModelId: string): boolean;
}

export class ModelPropsManager {
  /** Version counter for the cache - bumped on writes so $derived consumers recompute. */
  vCacheVersion = $state(0);
  /**
   * Model-specific props cache with TTL.
   * Key: modelId, Value: props data including modalities.
   */
  private vCache = new TTLCache<string, ApiDgsServerProps>({
    maxEntries: cModelPropsCache.MAX_ENTRIES,
    ttlMs: cModelPropsCache.TTL_MS
  });
  private vFetching = new SvelteSet<string>();

  /**
   * Whether the selected model's chat template supports thinking/reasoning.
   * Uses heuristic detection on the model's chat_template from /props.
   *
   * - MODEL mode: the global /props already describes the single loaded model,
   *   so its chat_template is used directly and no per-model cache is involved
   * - ROUTER mode: fetches /props?model=<id> for the selected model (cached),
   *   triggering an async fetch if not yet cached
   */
  get supportsThinking(): boolean {
    if (!cServerStore.isRouterMode) {
      return fDetectThinkingSupport(cServerStore.props?.chat_template ?? '');
    }

    const modelId = this.pHost.vSelectedModelName;

    if (!modelId) return false;

    if (!this.vCache.mGet(modelId)) {
      this.mFetchModelProps(modelId);
    }

    const cProps = this.mGetModelProps(modelId);

    return fDetectThinkingSupport(cProps?.chat_template ?? '');
  }

  /** Map the router modalities, the only source available while a model is not loaded. */
  mBuildArchitectureModalities(
    pArchitecture: ApiModelDataEntry['architecture']
  ): ModelModalities | undefined {
    if (!pArchitecture) return undefined;

    const cInputs = pArchitecture.input_modalities;

    return {
      audio: cInputs.includes(FileTypeCategory.AUDIO),
      video: cInputs.includes(FileTypeCategory.VIDEO),
      vision: cInputs.includes(FileTypeCategory.IMAGE)
    };
  }

  /**
   * Check if a specific model supports thinking.
   * In MODEL mode the global /props describes the single loaded model.
   * In ROUTER mode, fetches model props if not cached.
   */
  mCheckModelSupportsThinking(pModelId: string): boolean {
    if (!cServerStore.isRouterMode) {
      return fDetectThinkingSupport(cServerStore.props?.chat_template ?? '');
    }

    if (!pModelId) return false;

    if (!this.vCache.mGet(pModelId)) {
      this.mFetchModelProps(pModelId);
    }

    const cProps = this.mGetModelProps(pModelId);

    return fDetectThinkingSupport(cProps?.chat_template ?? '');
  }

  constructor(private pHost: ModelPropsHost) {}

  /** Fetch modalities for all loaded models from /props endpoint. */
  async mFetchModalitiesForLoadedModels(): Promise<void> {
    const aLoadedModelIds = this.pHost.loadedModelIds;

    if (aLoadedModelIds.length === 0) return;

    const aPropsPromises = aLoadedModelIds.map((pModelId) => this.mFetchModelProps(pModelId));

    try {
      const cResults = await Promise.all(aPropsPromises);

      this.pHost.aModels = this.pHost.aModels.map((pModel) => {
        const cModelIndex = aLoadedModelIds.indexOf(pModel.model);

        if (cModelIndex === -1) return pModel;

        const cProps = cResults[cModelIndex];

        if (!cProps?.modalities) return pModel;

        return { ...pModel, modalities: this.mBuildModalities(cProps.modalities) };
      });

      this.vCacheVersion++;
    } catch (error) {
      console.warn('Failed to fetch modalities for loaded models:', error);
    }
  }

  /**
   * Fetch props for a specific model from /props endpoint.
   * Uses caching to avoid redundant requests.
   *
   * In ROUTER mode, this only fetches props if the model is loaded,
   * since unloaded models return 400 from /props endpoint.
   *
   * @param modelId - Model identifier to fetch props for
   * @returns Props data or null if fetch failed or model not loaded
   */
  async mFetchModelProps(pModelId: string): Promise<ApiDgsServerProps | null> {
    const cCached = this.vCache.mGet(pModelId);

    if (cCached) return cCached;

    if (cServerStore.isRouterMode && !this.pHost.mIsModelLoaded(pModelId)) {
      return null;
    }

    if (this.vFetching.has(pModelId)) return null;

    this.vFetching.add(pModelId);

    try {
      const cProps = await PropsService.mFetchForModel(pModelId);

      this.vCache.mSet(pModelId, cProps);
      this.vCacheVersion++;

      return cProps;
    } catch (error) {
      console.warn(`Failed to fetch props for model ${pModelId}:`, error);

      return null;
    } finally {
      this.vFetching.delete(pModelId);
    }
  }

  mGetModelContextSize(pModelId: string): number | null {
    const cProps = this.mGetModelProps(pModelId);
    const cNCtx = cProps?.default_generation_settings?.n_ctx;

    return typeof cNCtx === 'number' ? cNCtx : null;
  }

  mGetModelModalities(pModelId: string): ModelModalities | null {
    if (!cServerStore.isRouterMode && cServerStore.props?.modalities) {
      return this.mBuildModalities(cServerStore.props.modalities);
    }

    const model = this.pHost.aModels.find((pM) => pM.model === pModelId || pM.id === pModelId);

    if (model?.modalities) {
      return model.modalities;
    }

    const cProps = this.vCache.mGet(pModelId);

    if (cProps?.modalities) {
      return this.mBuildModalities(cProps.modalities);
    }

    return null;
  }

  mGetModelModalitiesArray(pModelId: string): ModelModality[] {
    const cModalities = this.mGetModelModalities(pModelId);

    if (!cModalities) return [];

    const aResult: ModelModality[] = [];

    if (cModalities.vision) aResult.push(ModelModality.VISION);

    if (cModalities.audio) aResult.push(ModelModality.AUDIO);

    if (cModalities.video) aResult.push(ModelModality.VIDEO);

    return aResult;
  }

  mGetModelProps(pModelId: string): ApiDgsServerProps | null {
    return this.vCache.mGet(pModelId);
  }

  mIsModelPropsFetching(pModelId: string): boolean {
    return this.vFetching.has(pModelId);
  }

  mModelSupportsAudio(pModelId: string): boolean {
    return this.mGetModelModalities(pModelId)?.audio ?? false;
  }

  mModelSupportsVideo(pModelId: string): boolean {
    return this.mGetModelModalities(pModelId)?.video ?? false;
  }

  mModelSupportsVision(pModelId: string): boolean {
    return this.mGetModelModalities(pModelId)?.vision ?? false;
  }

  /**
   * Update modalities for a specific model.
   * Called when a model is loaded or when we need fresh modality data.
   */
  async mUpdateModelModalities(pModelId: string): Promise<void> {
    const cProps = await this.mFetchModelProps(pModelId);

    if (!cProps?.modalities) return;

    this.pHost.aModels = this.pHost.aModels.map((pModel) =>
      pModel.model === pModelId
        ? { ...pModel, modalities: this.mBuildModalities(cProps.modalities!) }
        : pModel
    );

    this.vCacheVersion++;
  }

  private mBuildModalities(
    pModalities: NonNullable<ApiDgsServerProps['modalities']>
  ): ModelModalities {
    return {
      audio: pModalities.audio ?? false,
      video: pModalities.video ?? false,
      vision: pModalities.vision ?? false
    };
  }
}
