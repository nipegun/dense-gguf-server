import { cChatInputFocusSelector } from '$lib/constants';
import { cDeviceStore, cModelsStore, cServerStore, cUiStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import {
  fFilterModelOptions,
  fGroupFavoriteOptions,
  fGroupModelOptions,
  type ModelItem
} from '$lib/utils';
import { fNameOf } from '$lib/utils/model-names';
import { onMount } from 'svelte';
import { SvelteSet } from 'svelte/reactivity';

export interface UseModelsSelectorOptions {
  currentModel: () => string | null;
  useGlobalSelection?: () => boolean;
  onModelChange?: () =>
    | ((pModelId: string, pModelName: string) => Promise<boolean> | boolean | void)
    | undefined;
  onOpenChange?: (pOpen: boolean) => void;
}

export interface UseModelsSelectorReturn {
  readonly options: ModelOption[];
  readonly loading: boolean;
  readonly updating: boolean;
  readonly activeId: string | null;
  readonly emptyMessage: string;
  readonly isMultiModel: boolean;
  readonly isRouter: boolean;
  readonly serverModel: string | null;
  readonly isHighlightedCurrentModelActive: boolean;
  readonly isCurrentModelInCache: boolean;
  readonly favoriteItems: ModelItem[];
  readonly loadedItems: ModelItem[];
  readonly filteredOptions: ModelOption[];
  readonly isEmpty: boolean;
  readonly groupedFilteredOptions: ReturnType<typeof fGroupModelOptions>;
  readonly isLoadingModel: boolean;
  readonly searchTerm: string;
  mSetSearchTerm(pValue: string): void;
  mHandleSelect(pModelId: string): Promise<void>;
  mHandleOpenChange(pOpen: boolean): void;
  mIsFavorite(pModel: string): boolean;
  mGetDisplayOption(): ModelOption | undefined;
}

/**
 * Shared reactive state and logic for model selection.
 *
 * Used by the model selector dropdown, which serves the desktop an anchored menu
 * and the phone a bottom drawer, to avoid duplicating store derivations,
 * selection handling, and model loading.
 */
export function fUseModelsSelector(pOpts: UseModelsSelectorOptions): UseModelsSelectorReturn {
  let vIsLoadingModel = $state(false);
  let vSearchTerm = $state('');

  const aOptions = $derived(
    cModelsStore.aModels.filter((pOption) => {
      const cModelProps = cModelsStore.props.mGetModelProps(pOption.model);

      return cModelProps?.ui !== false;
    })
  );
  const loading = $derived(cModelsStore.loading);
  const cUpdating = $derived(cModelsStore.vUpdating);
  const cActiveId = $derived(cModelsStore.vSelectedModelId);
  // a lone dgs.cpp server without a router has nothing to choose from
  const cIsRouter = $derived(cServerStore.isRouterMode);
  const cServerModel = $derived(cModelsStore.singleModelName);
  const cCurrentModel = $derived(pOpts.currentModel());
  const cOnModelChange = $derived(pOpts.onModelChange?.());
  const cIsHighlightedCurrentModelActive = $derived.by(() => {
    if (!cIsRouter || !cCurrentModel) return false;

    const cCurrentOption = aOptions.find((pOption) => pOption.model === cCurrentModel);

    return cCurrentOption ? cCurrentOption.id === cActiveId : false;
  });
  const cIsCurrentModelInCache = $derived.by(() => {
    if (!cIsRouter || !cCurrentModel) return true;

    return aOptions.some((pOption) => pOption.model === cCurrentModel);
  });
  // the search, the rows and the sections all read the visible set; only the current
  // model resolves against `options`, since it can be hidden and still selected
  const aVisibleOptions = $derived(
    aOptions.filter((pOption) => !cModelsStore.mIsHidden(pOption.id))
  );
  // one filter pass feeds the favorites, the loaded rows and the sections alike
  const aFilteredOptions = $derived(fFilterModelOptions(aVisibleOptions, vSearchTerm));
  const aLoadedItems = $derived(
    aFilteredOptions
      .filter((pOption) => cModelsStore.mIsModelLoaded(pOption.model))
      .map((pOption) => ({ option: pOption }))
  );
  const cLoadedIds = $derived(new SvelteSet(aLoadedItems.map((pItem) => pItem.option.id)));
  // loaded models lead the list: their own sections list them once, so a
  // loaded favorite shows there and not twice
  const aFavoriteItems = $derived(
    fGroupFavoriteOptions(
      aFilteredOptions.filter((pOption) => !cLoadedIds.has(pOption.id)),
      cModelsStore.sFavoriteModelIds
    )
  );
  // loaded models and favorites are listed once, at the top: the sections skip both
  const aSectionOptions = $derived(
    aFilteredOptions.filter(
      (pOption) => !cModelsStore.sFavoriteModelIds.has(pOption.model) && !cLoadedIds.has(pOption.id)
    )
  );
  const cGroupedFilteredOptions = $derived(fGroupModelOptions(aSectionOptions));
  const cIsEmpty = $derived(
    aFilteredOptions.length === 0 && aFavoriteItems.length === 0 && aLoadedItems.length === 0
  );
  const cEmptyMessage = $derived(vSearchTerm ? 'No models found.' : 'No models yet.');

  // the manager takes the focus, so the selector closes instead of sitting behind it
  $effect(() => {
    if (!cUiStore.vManageModelsOpen) return;

    pOpts.onOpenChange?.(false);
  });

  onMount(() => {
    cModelsStore.mFetch().catch((pError) => {
      console.error('Unable to load models:', pError);
    });
  });

  function fHandleOpenChange(pOpen: boolean) {
    if (loading || cUpdating) return;

    // a single-model desktop server has no list: the trigger opens the manager
    // instead. a phone has no manager, so its drawer opens with the one model
    if (!cIsRouter && !cDeviceStore.isMobile) {
      if (pOpen) cUiStore.mOpenModelsManager();

      return;
    }

    vSearchTerm = '';

    if (pOpen && cIsRouter) {
      cModelsStore.props.mFetchModalitiesForLoadedModels();
    }

    pOpts.onOpenChange?.(pOpen);
  }

  async function fHandleSelect(pModelId: string) {
    const cOption = aOptions.find((pOpt) => pOpt.id === pModelId);

    if (!cOption) return;

    let vShouldCloseMenu = true;

    if (cOnModelChange) {
      const cResult = await cOnModelChange(cOption.id, cOption.model);

      if (cResult === false) {
        vShouldCloseMenu = false;
      }
    } else {
      await cModelsStore.mSelectModelById(cOption.id, { recordRecent: true });
    }

    if (vShouldCloseMenu) {
      fHandleOpenChange(false);

      requestAnimationFrame(() => {
        const input = document.querySelector<HTMLElement>(cChatInputFocusSelector);

        input?.focus({ preventScroll: true });
      });
    }

    // only the built-in server loads on request, and only in router mode
    if (!cOnModelChange && cIsRouter && !cModelsStore.mIsModelLoaded(cOption.model)) {
      vIsLoadingModel = true;

      cModelsStore.status
        .load(cOption.model)
        .catch((pError) => console.error('Failed to load model:', pError))
        .finally(() => (vIsLoadingModel = false));
    }
  }

  function fGetDisplayOption(): ModelOption | undefined {
    if (!cIsRouter) {
      const cDisplayModel = cServerModel || cCurrentModel;

      if (cDisplayModel) {
        return {
          capabilities: [],
          id: cServerModel ? 'current' : 'offline-current',
          model: cDisplayModel,
          name: fNameOf(cDisplayModel)
        };
      }

      return undefined;
    }

    if (cCurrentModel) {
      if (!cIsCurrentModelInCache) {
        return {
          capabilities: [],
          id: 'not-in-cache',
          model: cCurrentModel,
          name: fNameOf(cCurrentModel)
        };
      }

      return aOptions.find((pOption) => pOption.model === cCurrentModel);
    }

    if (cActiveId) {
      return aOptions.find((pOption) => pOption.id === cActiveId);
    }

    return undefined;
  }

  return {
    get activeId() {
      return cActiveId;
    },

    get emptyMessage() {
      return cEmptyMessage;
    },

    get favoriteItems() {
      return aFavoriteItems;
    },
    get filteredOptions() {
      return aFilteredOptions;
    },

    mGetDisplayOption: fGetDisplayOption,

    get groupedFilteredOptions() {
      return cGroupedFilteredOptions;
    },

    mHandleOpenChange: fHandleOpenChange,

    mHandleSelect: fHandleSelect,

    get isCurrentModelInCache() {
      return cIsCurrentModelInCache;
    },

    get isEmpty() {
      return cIsEmpty;
    },

    mIsFavorite(pModel: string) {
      return cModelsStore.sFavoriteModelIds.has(pModel);
    },

    get isHighlightedCurrentModelActive() {
      return cIsHighlightedCurrentModelActive;
    },

    get isLoadingModel() {
      return vIsLoadingModel;
    },

    get isMultiModel() {
      return cIsRouter;
    },

    get isRouter() {
      return cIsRouter;
    },

    get loadedItems() {
      return aLoadedItems;
    },

    get loading() {
      return loading;
    },

    get options() {
      return aOptions;
    },

    get searchTerm() {
      return vSearchTerm;
    },

    get serverModel() {
      return cServerModel;
    },

    mSetSearchTerm(pValue: string) {
      vSearchTerm = pValue;
    },

    get updating() {
      return cUpdating;
    }
  };
}
