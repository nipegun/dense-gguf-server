/**
 * Model state for the chat form's action bar.
 *
 * Syncs the conversation's model into the global selection (a conversation
 * opened from history names the model that produced it), falls back to the
 * first loaded model on a router, and reactively tracks which modalities
 * (vision / audio / video) the active model supports - fetching model props
 * from the server on demand if they are not cached yet.
 */

import { cConversationsStore, cModelsStore, cServerStore } from '$lib/stores';
import type { DatabaseMessage } from '$lib/types/database';
import { fGetConversationModel } from '$lib/utils';

export function fUseChatFormModel() {
  const cIsRouter = $derived(cServerStore.isRouterMode);
  const cConversationModel = $derived(
    fGetConversationModel(cConversationsStore.aActiveMessages as DatabaseMessage[])
  );

  let vLastSyncedConversationModel: string | null = null;

  // the conversation's model wins while it names one: opening a history chat
  // points the selection at the model that produced it
  $effect(() => {
    if (cConversationModel && cConversationModel !== vLastSyncedConversationModel) {
      if (cModelsStore.aModels.some((pM) => pM.model === cConversationModel)) {
        cModelsStore.vSelectedModelName = cConversationModel;
        cModelsStore.mSelectModelByName(cConversationModel);
      } else {
        cModelsStore.vSelectedModelName = null;
        cModelsStore.mClearSelection();
      }

      vLastSyncedConversationModel = cConversationModel;
    } else if (
      cIsRouter &&
      !cModelsStore.vSelectedModelId &&
      cModelsStore.loadedModelIds.length > 0 &&
      cConversationsStore.aActiveMessages.length > 0 &&
      !cConversationModel
    ) {
      vLastSyncedConversationModel = null;
      const cFirst = cModelsStore.aModels.find((pM) =>
        cModelsStore.loadedModelIds.includes(pM.model)
      );

      if (cFirst) cModelsStore.mSelectModelById(cFirst.id);
    }
  });

  const cActiveModelId = $derived(cModelsStore.activeModelId);

  let vModelPropsVersion = $state(0); // Used to trigger reactivity after fetch

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
    void vModelPropsVersion;

    return cActiveModelId ? cModelsStore.props.mModelSupportsAudio(cActiveModelId) : false;
  });
  const hasVideoModality = $derived.by(() => {
    void vModelPropsVersion;

    return cActiveModelId ? cModelsStore.props.mModelSupportsVideo(cActiveModelId) : false;
  });
  const hasVisionModality = $derived.by(() => {
    void vModelPropsVersion;

    return cActiveModelId ? cModelsStore.props.mModelSupportsVision(cActiveModelId) : false;
  });
  const cHasModelSelected = $derived(
    !cIsRouter || !!cConversationModel || !!cModelsStore.vSelectedModelId
  );
  const cIsSelectedModelInCache = $derived.by(() => {
    if (!cIsRouter) return true;

    if (cConversationModel) {
      return cModelsStore.aModels.some((pOption) => pOption.model === cConversationModel);
    }

    const cCurrentModelId = cModelsStore.vSelectedModelId;

    if (!cCurrentModelId) return false;

    return cModelsStore.aModels.some((pOption) => pOption.id === cCurrentModelId);
  });

  return {
    get conversationModel() {
      return cConversationModel;
    },
    get hasAudioModality() {
      return hasAudioModality;
    },
    get hasModelSelected() {
      return cHasModelSelected;
    },
    get hasVideoModality() {
      return hasVideoModality;
    },
    get hasVisionModality() {
      return hasVisionModality;
    },
    get isRouter() {
      return cIsRouter;
    },
    get isSelectedModelInCache() {
      return cIsSelectedModelInCache;
    }
  };
}
