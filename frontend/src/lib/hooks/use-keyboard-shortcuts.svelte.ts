import { page } from '$app/state';
import { cNewChatTabId } from '$lib/constants';
import { KeyboardKey } from '$lib/enums';
import { cConversationsStore, cSettingsStore, cTabsStore } from '$lib/stores';

interface KeyboardShortcutsCallbacks {
  activateSearchMode?: () => void;
  editActiveConversation?: () => void;
  onSearchActivated?: () => void;
  deleteActiveConversation?: () => void;
  navigateToPrevConversation?: () => void;
  navigateToNextConversation?: () => void;
  navigateToPrevTab?: () => void;
  navigateToNextTab?: () => void;
  toggleSidebar?: () => void;
}

export function fUseKeyboardShortcuts(pCallbacks: KeyboardShortcutsCallbacks) {
  function fHandleKeydown(pEvent: KeyboardEvent) {
    const cIsCmdOrCtrl = pEvent.metaKey || pEvent.ctrlKey;

    if (cIsCmdOrCtrl && pEvent.key === KeyboardKey.cKLower) {
      pEvent.preventDefault();
      pCallbacks.activateSearchMode?.();
      pCallbacks.onSearchActivated?.();
    }

    if (cIsCmdOrCtrl && pEvent.key === KeyboardKey.cBLower) {
      pEvent.preventDefault();
      pCallbacks.toggleSidebar?.();
    }

    if (
      cIsCmdOrCtrl &&
      pEvent.shiftKey &&
      (pEvent.key === KeyboardKey.cOLower || pEvent.key === KeyboardKey.cOUpper)
    ) {
      pEvent.preventDefault();

      void cConversationsStore.mOpenNewChat();
    }

    if (pEvent.shiftKey && cIsCmdOrCtrl && pEvent.key === KeyboardKey.cEUpper) {
      pEvent.preventDefault();
      pCallbacks.editActiveConversation?.();
    }

    if (
      pEvent.shiftKey &&
      cIsCmdOrCtrl &&
      (pEvent.key === KeyboardKey.cXLower || pEvent.key === KeyboardKey.cXUpper)
    ) {
      // several components register this shortcut; only let the first handler
      // act so the synchronous navigation does not cascade-close every tab
      if (pEvent.defaultPrevented) return;

      // close-tab only makes sense with conversation tabs enabled
      if (!cSettingsStore.vConfig.conversationTabs) return;

      pEvent.preventDefault();

      const cActiveId = page.params.id ?? (page.route.id === '/(chat)' ? cNewChatTabId : undefined);

      if (cActiveId) {
        void cTabsStore.mClose(cActiveId, cActiveId);
      }
    }

    if (
      cIsCmdOrCtrl &&
      pEvent.shiftKey &&
      (pEvent.key === KeyboardKey.cDLower || pEvent.key === KeyboardKey.cDUpper)
    ) {
      pEvent.preventDefault();
      pCallbacks.deleteActiveConversation?.();
    }

    if (cIsCmdOrCtrl && pEvent.shiftKey && pEvent.key === KeyboardKey.cArrowUp) {
      pEvent.preventDefault();
      pCallbacks.navigateToPrevConversation?.();
    }

    if (cIsCmdOrCtrl && pEvent.shiftKey && pEvent.key === KeyboardKey.cArrowDown) {
      pEvent.preventDefault();
      pCallbacks.navigateToNextConversation?.();
    }

    if (
      cIsCmdOrCtrl &&
      pEvent.altKey &&
      pEvent.shiftKey &&
      pEvent.code === KeyboardKey.cBracketLeft
    ) {
      pEvent.preventDefault();
      pCallbacks.navigateToPrevTab?.();
    }

    if (
      cIsCmdOrCtrl &&
      pEvent.altKey &&
      pEvent.shiftKey &&
      pEvent.code === KeyboardKey.cBracketRight
    ) {
      pEvent.preventDefault();
      pCallbacks.navigateToNextTab?.();
    }
  }

  return { handleKeydown: fHandleKeydown };
}
