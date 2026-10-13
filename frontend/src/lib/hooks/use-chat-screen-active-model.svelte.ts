/**
 * Active model resolution and capability detection for the ChatScreen.
 *
 * Picks the model that should be used for the current view
 * (router: user-selected or conversation fallback; non-router: first
 * available option), and reactively tracks which modalities (vision /
 * audio / video) it supports — fetching model props from the server on
 * demand if they aren't cached yet.
 */

import { cConversationsStore, cModelsStore, cServerStore } from '$lib/stores';
import { fGetConversationModel } from '$lib/utils';

export function fUseChatScreenActiveModel() {
  const cIsRouter = $derived(cServerStore.isRouterMode);
  const cConversationModel = $derived(
    fGetConversationModel(cConversationsStore.aActiveMessages as DatabaseMessage[])
  );
  const cActiveModelId = $derived(cModelsStore.activeModelId);

  let vModelPropsVersion = $state(0);

  $effect(() => {
    if (cActiveModelId) {
      const cCached = cModelsStore.props.mGetModelProps(cActiveModelId);

      if (!cCached) {
        cModelsStore.props.mFetchModelProps(cActiveModelId).then(() => {
          vModelPropsVersion++;
        });
      }
    }
  });

  const hasAudioModality = $derived.by(() => {
    if (cActiveModelId) {
      void vModelPropsVersion;

      return cModelsStore.props.mModelSupportsAudio(cActiveModelId);
    }

    return false;
  });
  const hasVideoModality = $derived.by(() => {
    if (cActiveModelId) {
      void vModelPropsVersion;

      return cModelsStore.props.mModelSupportsVideo(cActiveModelId);
    }

    return false;
  });
  const hasVisionModality = $derived.by(() => {
    if (cActiveModelId) {
      void vModelPropsVersion;

      return cModelsStore.props.mModelSupportsVision(cActiveModelId);
    }

    return false;
  });

  return {
    get activeModelId() {
      return cActiveModelId;
    },
    get conversationModel() {
      return cConversationModel;
    },
    get hasAudioModality() {
      return hasAudioModality;
    },
    get hasVideoModality() {
      return hasVideoModality;
    },
    get hasVisionModality() {
      return hasVisionModality;
    },
    get isRouter() {
      return cIsRouter;
    }
  };
}
