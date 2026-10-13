import { aReasoningEffortLevels, dReasoningEffortTokens } from '$lib/constants';
import { ReasoningEffort } from '$lib/enums';
import { cConversationsStore, cModelsStore, cServerStore } from '$lib/stores';
import type { ReasoningEffortLevel } from '$lib/types';
import type { DatabaseMessage } from '$lib/types/database';
import { fGetConversationModel } from '$lib/utils';

export interface UseReasoningMenuReturn {
  readonly modelSupportsThinking: boolean;
  readonly thinkingEnabled: boolean;
  readonly isReasoningActive: boolean;
  readonly isOff: boolean;
  readonly currentEffort: ReasoningEffort;
  readonly levels: ReasoningEffortLevel[];
  mIsSelected(pLevel: ReasoningEffortLevel): boolean;
  mTokenLabel(pLevel: ReasoningEffortLevel): string | null;
  mSelect(pLevel: ReasoningEffortLevel): void;
}

/**
 * Shared reactive state and helpers for the reasoning effort menu.
 *
 * Shared by the dropdown footer and the mobile sheet.
 */
export function fUseReasoningMenu(): UseReasoningMenuReturn {
  const cConversationModel = $derived(
    fGetConversationModel(cConversationsStore.aActiveMessages as DatabaseMessage[])
  );
  // a router chat can carry reasoning from an earlier turn before the props
  // cache is primed, so a model that already produced thinking still qualifies
  const cModelSupportsThinkingFromMessages = $derived.by(() => {
    const modelId = cServerStore.isRouterMode
      ? cModelsStore.vSelectedModelName || cConversationModel
      : null;

    if (!modelId) return false;

    return cConversationsStore.aActiveMessages.some(
      (pM) => pM.role === 'assistant' && pM.model === modelId && !!pM.reasoningContent
    );
  });
  const cModelSupportsThinking = $derived.by(() => {
    void cModelsStore.loadedModelIds;
    void cModelsStore.props.vCacheVersion;

    if (cServerStore.isRouterMode) {
      const modelId = cModelsStore.vSelectedModelName || cConversationModel;

      return (
        cModelsStore.props.mCheckModelSupportsThinking(modelId ?? '') ||
        cModelSupportsThinkingFromMessages
      );
    }

    return cModelsStore.props.supportsThinking || cModelSupportsThinkingFromMessages;
  });
  const cCurrentEffort = $derived(cConversationsStore.preferences.mGetReasoningEffort());
  const cThinkingEnabled = $derived(
    cCurrentEffort !== ReasoningEffort.cOff && cCurrentEffort !== ReasoningEffort.cDefault
  );
  // Thinking is effectively on (lightbulb lit) either when an explicit effort
  // is selected, or when the effort is left at "Default" and the model
  // supports thinking.
  const cIsReasoningActive = $derived(
    cThinkingEnabled || (cCurrentEffort === ReasoningEffort.cDefault && cModelSupportsThinking)
  );

  return {
    get currentEffort() {
      return cCurrentEffort;
    },
    get isOff() {
      return cCurrentEffort === ReasoningEffort.cOff;
    },
    get isReasoningActive() {
      return cIsReasoningActive;
    },
    mIsSelected(pLevel: ReasoningEffortLevel): boolean {
      return cCurrentEffort === pLevel.value;
    },
    get levels() {
      return aReasoningEffortLevels;
    },
    get modelSupportsThinking() {
      return cModelSupportsThinking;
    },
    mSelect(pLevel: ReasoningEffortLevel): void {
      cConversationsStore.preferences.mSetReasoningEffort(pLevel.value as ReasoningEffort);
    },
    get thinkingEnabled() {
      return cThinkingEnabled;
    },
    mTokenLabel(pLevel: ReasoningEffortLevel): string | null {
      if (pLevel.value === ReasoningEffort.cDefault) return 'Model default';

      const cTokens = dReasoningEffortTokens[pLevel.value];

      if (cTokens === undefined) return null;

      return cTokens === -1 ? 'Unlimited' : `Max ${cTokens.toLocaleString()} tokens`;
    }
  };
}
