/**
 * View layer over contextStatsStore for the context usage gauge: adds
 * color levels, transient detail formatting, on-demand /props fetching
 * and model loading on top of the store's token stats.
 */

import { fUseProcessingState } from './use-processing-state.svelte';
import { fColorLevelFromPercent } from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/context-gauge';
import { cStatsUnits } from '$lib/constants';
import { ColorLevel } from '$lib/enums';
import { cContextStatsStore, cModelsStore } from '$lib/stores';

export interface UseContextGaugeReturn {
  readonly activeModelId: string | null;
  readonly isActiveModelLoaded: boolean;
  readonly isActiveModelLoading: boolean;
  readonly contextTotal: number | null;
  readonly contextUsed: number;
  readonly contextAvailable: number | null;
  readonly currentRead: number;
  readonly currentFresh: number;
  readonly currentCache: number;
  readonly currentOutput: number;
  readonly kvTotal: number;
  readonly cumulativeRead: number;
  readonly cumulativeOutput: number;
  readonly cumulativeCacheTotal: number;
  readonly averageTokensPerSecond: number | null;
  readonly contextPercent: number | null;
  readonly colorLevel: ColorLevel;
  readonly transientDetails: string[];
  readonly hasAnyUsage: boolean;
  mLoadModel(): Promise<void>;
  mStartMonitoring(): void;
}

const aTransientDetailsExcludedPrefixes = ['Context:', 'Output:'];

function fFilterTransientDetails(pRaw: string[]): string[] {
  return pRaw.filter((pDetail) => {
    if (aTransientDetailsExcludedPrefixes.some((pPrefix) => pDetail.startsWith(pPrefix))) {
      return false;
    }

    return !pDetail.includes(cStatsUnits.TOKENS_PER_SECOND);
  });
}

export function fUseContextGauge(): UseContextGaugeReturn {
  const cProcessingState = fUseProcessingState();

  // Pull /props on demand so n_ctx surfaces before the first chat request.
  $effect(() => {
    const modelId = cContextStatsStore.vActiveModelId;

    if (modelId && cContextStatsStore.vIsActiveModelLoaded) {
      const cCached = cModelsStore.props.mGetModelProps(modelId);

      if (!cCached) {
        void cModelsStore.props.mFetchModelProps(modelId);
      }
    }
  });

  const cColorLevel = $derived(fColorLevelFromPercent(cContextStatsStore.vContextPercent));
  // Drop lines the surrounding Context / Output / speed rows already render.
  const aTransientDetails = $derived(
    fFilterTransientDetails(cProcessingState.mGetTechnicalDetails())
  );
  const cHasAnyUsage = $derived(
    cContextStatsStore.vCumulativeRead > 0 ||
      cContextStatsStore.vCumulativeOutput > 0 ||
      cContextStatsStore.vCurrentRead > 0 ||
      cContextStatsStore.vCurrentOutput > 0 ||
      cContextStatsStore.vAverageTokensPerSecond !== null ||
      aTransientDetails.length > 0
  );

  async function fLoadModel() {
    const modelId = cContextStatsStore.vActiveModelId;

    if (!modelId || cContextStatsStore.vIsActiveModelLoading) return;

    try {
      await cModelsStore.status.load(modelId);
    } catch {
      // toast already surfaced by modelsStore.status.load
    }
  }

  return {
    get activeModelId() {
      return cContextStatsStore.vActiveModelId;
    },
    get averageTokensPerSecond() {
      return cContextStatsStore.vAverageTokensPerSecond;
    },
    get colorLevel() {
      return cColorLevel;
    },
    get contextAvailable() {
      return cContextStatsStore.vContextAvailable;
    },
    get contextPercent() {
      return cContextStatsStore.vContextPercent;
    },
    get contextTotal() {
      return cContextStatsStore.vContextTotal;
    },
    get contextUsed() {
      return cContextStatsStore.vContextUsed;
    },
    get cumulativeCacheTotal() {
      return cContextStatsStore.vCumulativeCacheTotal;
    },
    get cumulativeOutput() {
      return cContextStatsStore.vCumulativeOutput;
    },
    get cumulativeRead() {
      return cContextStatsStore.vCumulativeRead;
    },
    get currentCache() {
      return cContextStatsStore.vCurrentCache;
    },
    get currentFresh() {
      return cContextStatsStore.vCurrentFresh;
    },
    get currentOutput() {
      return cContextStatsStore.vCurrentOutput;
    },
    get currentRead() {
      return cContextStatsStore.vCurrentRead;
    },
    get hasAnyUsage() {
      return cHasAnyUsage;
    },
    get isActiveModelLoaded() {
      return cContextStatsStore.vIsActiveModelLoaded;
    },
    get isActiveModelLoading() {
      return cContextStatsStore.vIsActiveModelLoading;
    },
    get kvTotal() {
      return cContextStatsStore.vKvTotal;
    },
    mLoadModel: fLoadModel,
    mStartMonitoring: () => cProcessingState.mStartMonitoring(),
    get transientDetails() {
      return aTransientDetails;
    }
  };
}
