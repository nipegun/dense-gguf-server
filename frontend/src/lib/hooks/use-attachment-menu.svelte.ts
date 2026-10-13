import { page } from '$app/state';
import { AttachmentAction } from '$lib/enums';

export interface AttachmentModalityFlags {
  hasVisionModality: boolean;
  hasAudioModality: boolean;
  hasVideoModality: boolean;
}

export interface AttachmentActionCallbacks {
  onFileUpload?: () => void;
  onSystemPromptClick?: () => void;
}

export interface UseAttachmentMenuReturn {
  readonly callbacks: Record<string, () => void>;
  mIsItemEnabled(pEnabledWhen: string | undefined): boolean;
  mGetSystemMessageTooltip(): string;
}

/**
 * useAttachmentMenu - Shared logic for attachment menu components.
 *
 * Encapsulates the modality-flag checks and callback wrapping that is
 * identical across the desktop dropdown (`ChatFormActionAddDropdown`)
 * and the mobile drawer (`ChatFormActionAddDrawer`).
 *
 * @param getFlags   - Getter returning the current modality capability flags.
 * @param getCallbacks - Getter returning the raw action callbacks from props.
 * @param close      - Function that dismisses the hosting UI element (dropdown / sheet).
 */
export function fUseAttachmentMenu(
  pGetFlags: () => AttachmentModalityFlags,
  pGetCallbacks: () => AttachmentActionCallbacks,
  pClose: () => void
): UseAttachmentMenuReturn {
  const cModalityFlags = $derived(pGetFlags());
  const cCallbacks = $derived.by(() => {
    const cCbs = pGetCallbacks();
    const fWrap = (pFn?: () => void) => () => {
      pClose();
      pFn?.();
    };

    return {
      [AttachmentAction.FILE_UPLOAD]: fWrap(cCbs.onFileUpload),
      [AttachmentAction.cSystemPromptClick]: fWrap(cCbs.onSystemPromptClick)
    };
  });

  function fIsItemEnabled(pEnabledWhen: string | undefined): boolean {
    if (!pEnabledWhen || pEnabledWhen === 'always') return true;

    return !!cModalityFlags[pEnabledWhen as keyof AttachmentModalityFlags];
  }

  function fGetSystemMessageTooltip(): string {
    return !page.params.id
      ? 'Add custom system message for a new conversation'
      : 'Inject custom system message at the beginning of the conversation';
  }

  return {
    get callbacks() {
      return cCallbacks;
    },
    mGetSystemMessageTooltip: fGetSystemMessageTooltip,
    mIsItemEnabled: fIsItemEnabled
  };
}
