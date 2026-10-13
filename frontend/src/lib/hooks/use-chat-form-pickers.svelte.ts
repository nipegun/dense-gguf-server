import { cPromptTriggerPrefix } from '$lib/constants';
import { ChatFormCommandAction, KeyboardKey } from '$lib/enums';
import type { ChatFormCommand } from '$lib/types';
import { fGetChatCommands } from '$lib/utils';
import {
  type CommandDismissSnapshot,
  fFindCommandToken,
  fFindMentionToken,
  type MentionDismissSnapshot,
  fTakeCommandDismissSnapshot,
  fTakeMentionDismissSnapshot
} from '$lib/utils';

/** Dependencies injected as getters so the hook stays free of store circular imports. */
export interface UseChatFormPickersOptions {
  getValue: () => string;
  /** Also fires the form's onChange. */
  setValue: (pValue: string) => void;
  /** Undefined when unmounted. */
  getCaretOffset: () => number | undefined;
  setCaretOffset: (pOffset: number) => void;
  focusInput: () => void;
  /** Gates `/model`. */
  getShowModelSelector: () => boolean;
  /** Gates `/prompt`. */
  hasPrompts: () => boolean;
  /** Gates `/cwd`. */
  hasCwdTools: () => boolean;
  getCwd: () => string | null;
  /** Mention search fallback scope. */
  getServerHome: () => string | null;
  openModelSelector: () => void;
  /** Delegate a keydown to the mounted pickers component, if any. */
  getPickersRef: () => { handleKeydown(pEvent: KeyboardEvent): boolean } | undefined;
}

/**
 * Chat-form picker state and the `/`+`@` routing that drives them.
 * Owns open/query state, dismiss snapshots and slash-command dispatch;
 * textarea/caret/attachment handling stays in the chat form.
 */
export function fUseChatFormPickers(pOpts: UseChatFormPickersOptions) {
  let vIsCommandPickerOpen = $state(false);
  let vCommandQuery = $state('');
  let vIsPromptPickerOpen = $state(false);
  let vPromptSearchQuery = $state('');
  let vIsMentionPickerOpen = $state(false);
  let vMentionQuery = $state('');
  let vIsWorkingDirectoryPickerOpen = $state(false);
  let vWorkingDirectoryQuery = $state('');
  // Last dismissed `@`-mention token; while intact, the picker does not
  // reopen, so an escaped `@<query>` stays literal until edited.
  let vMentionDismissedSnapshot: MentionDismissSnapshot | null = null;
  // Same dismissal contract for the `/`-command token.
  let vCommandDismissedSnapshot: CommandDismissSnapshot | null = null;

  // Fall back to the server home so the picker still finds matches
  // before a cwd is set.
  const cMentionScopePath = $derived(pOpts.getCwd() ?? pOpts.getServerHome() ?? null);
  const aAvailableCommands = $derived(
    fGetChatCommands({
      hasCwdTools: pOpts.hasCwdTools,
      hasPrompts: pOpts.hasPrompts,
      showModelSelector: pOpts.getShowModelSelector()
    })
  );

  // Dispatch a slash command picked from the list: consume the token and
  // open the target picker, seeding its search with `args`. Runs only on
  // explicit selection (Enter/click), so the buffer is never cleared
  // mid-typing.
  function fDispatchCommand(pCommand: ChatFormCommand, pArgs: string) {
    vIsCommandPickerOpen = false;
    vCommandQuery = '';

    switch (pCommand.action) {
      case ChatFormCommandAction.PROMPT:
        vIsWorkingDirectoryPickerOpen = false;
        pOpts.setValue('');
        vIsPromptPickerOpen = true;
        vPromptSearchQuery = pArgs.trim();

        break;
      case ChatFormCommandAction.CWD: {
        // Keep `/cwd <args>` in the input so the search field and the
        // token stay two-way bound; normalize partial tokens (`/cw foo`).
        const cTrimmed = pArgs.trim();
        const cNewValue = `/cwd ${cTrimmed}`;

        if (pOpts.getValue() !== cNewValue) {
          pOpts.setValue(cNewValue);
          queueMicrotask(() => pOpts.setCaretOffset(cNewValue.length));
        }

        vWorkingDirectoryQuery = cTrimmed;
        vIsWorkingDirectoryPickerOpen = true;

        break;
      }
      case ChatFormCommandAction.MODEL:
        vIsWorkingDirectoryPickerOpen = false;
        pOpts.setValue('');
        pOpts.openModelSelector();

        break;
    }
  }

  function fHandleInput() {
    const cValue = pOpts.getValue();
    const cCursor = pOpts.getCaretOffset() ?? cValue.length;

    if (cValue.startsWith(cPromptTriggerPrefix)) {
      vIsMentionPickerOpen = false;
      vMentionQuery = '';
      vIsPromptPickerOpen = false;
      vPromptSearchQuery = '';

      const cToken = fFindCommandToken(cValue);

      if (!cToken) {
        vIsCommandPickerOpen = false;
        vCommandQuery = '';

        return;
      }

      // While the `/cwd` picker is open the token doubles as its search
      // field: keep the two in sync instead of re-dispatching.
      if (vIsWorkingDirectoryPickerOpen) {
        vIsCommandPickerOpen = false;
        vCommandQuery = '';

        if (cToken.name === 'cwd') {
          vWorkingDirectoryQuery = cToken.args.trim();
        } else {
          vIsWorkingDirectoryPickerOpen = false;
          vWorkingDirectoryQuery = '';
        }

        return;
      }

      // Dismissed token stays literal until it changes.
      const cIsDismissedSticky =
        vCommandDismissedSnapshot !== null &&
        vCommandDismissedSnapshot.name === cToken.name &&
        vCommandDismissedSnapshot.args === cToken.args;

      if (cIsDismissedSticky) {
        vIsCommandPickerOpen = false;
        vCommandQuery = '';

        return;
      }

      // Commands dispatch only on explicit selection (Enter/click),
      // never mid-typing: `/model is broken` is prose until the user
      // picks the command from the list.
      if (aAvailableCommands.length > 0) {
        vIsCommandPickerOpen = true;
        vCommandQuery = cToken.name;
      } else {
        vIsCommandPickerOpen = false;
        vCommandQuery = '';
      }

      return;
    }

    vIsCommandPickerOpen = false;
    vCommandQuery = '';

    if (vCommandDismissedSnapshot !== null) {
      vCommandDismissedSnapshot = null;
    }

    if (vIsWorkingDirectoryPickerOpen) {
      vIsWorkingDirectoryPickerOpen = false;
    }

    const cToken = fFindMentionToken(cValue, cCursor);

    if (cToken) {
      // Dismissed token stays literal: don't reopen until it changes.
      const cIsDismissedSticky =
        vMentionDismissedSnapshot !== null &&
        vMentionDismissedSnapshot.start === cToken.start &&
        vMentionDismissedSnapshot.query === cToken.query;

      if (!cIsDismissedSticky) {
        // Only search once a char follows `@`; a bare `@` is a no-op
        // (otherwise the picker flashes an empty hint on re-type).
        if (cToken.query.length > 0) {
          vMentionDismissedSnapshot = null;
          vIsMentionPickerOpen = true;
          vMentionQuery = cToken.query;
          vIsPromptPickerOpen = false;
          vPromptSearchQuery = '';

          return;
        }
      }
    }

    vIsPromptPickerOpen = false;
    vPromptSearchQuery = '';
    vIsMentionPickerOpen = false;
    vMentionQuery = '';

    // Token gone or changed: reset the snapshot so a fresh `@` reopens.
    if (vMentionDismissedSnapshot !== null && !cToken) {
      vMentionDismissedSnapshot = null;
    }
  }

  function fHandleKeydown(pEvent: KeyboardEvent): boolean {
    if (pOpts.getPickersRef()?.handleKeydown(pEvent)) {
      return true;
    }

    if (pEvent.key === KeyboardKey.cEscape && vIsPromptPickerOpen) {
      vIsPromptPickerOpen = false;
      vPromptSearchQuery = '';

      return true;
    }

    return false;
  }

  function fHandleCommandSelect(pCommand: ChatFormCommand) {
    // Dispatch on the live token so typed args seed the target picker.
    const cToken = fFindCommandToken(pOpts.getValue());

    fDispatchCommand(pCommand, cToken?.args ?? '');
  }

  // Picker dismissed: snapshot the live token so it stays literal until
  // deleted or retyped.
  function fHandleCommandPickerClose() {
    if (vIsCommandPickerOpen) {
      vCommandDismissedSnapshot = fTakeCommandDismissSnapshot(pOpts.getValue());
    }

    vIsCommandPickerOpen = false;
    vCommandQuery = '';

    // Target picker manages its own focus: don't yank it back to the input.
    if (!vIsPromptPickerOpen && !vIsMentionPickerOpen && !vIsWorkingDirectoryPickerOpen) {
      pOpts.focusInput();
    }
  }

  // Same dismissal snapshot for the mention token.
  function fHandleMentionPickerClose() {
    if (vIsMentionPickerOpen) {
      const cCursor = pOpts.getCaretOffset() ?? pOpts.getValue().length;

      vMentionDismissedSnapshot = fTakeMentionDismissSnapshot(pOpts.getValue(), cCursor);
    }

    vIsMentionPickerOpen = false;
    vMentionQuery = '';
    pOpts.focusInput();
  }

  function fHandlePromptPickerClose() {
    vIsPromptPickerOpen = false;
    vPromptSearchQuery = '';
    pOpts.focusInput();
  }

  function fHandleWorkingDirectoryOpen() {
    vWorkingDirectoryQuery = pOpts.getCwd() ?? '';
    vIsWorkingDirectoryPickerOpen = true;
  }

  function fHandleWorkingDirectoryClose() {
    vIsWorkingDirectoryPickerOpen = false;
    vWorkingDirectoryQuery = '';
    pOpts.focusInput();
  }

  // Two-way bind the text after `/cwd ` and the picker search input; the
  // reverse direction is handled by handleInput.
  $effect(() => {
    if (!vIsWorkingDirectoryPickerOpen) return;

    const cValue = pOpts.getValue();
    const cToken = fFindCommandToken(cValue);

    if (!cToken || cToken.name !== 'cwd') return;

    const cNewValue = `/cwd ${vWorkingDirectoryQuery}`;

    if (cNewValue === cValue) return;

    pOpts.setValue(cNewValue);
    queueMicrotask(() => pOpts.setCaretOffset(cNewValue.length));
  });

  return {
    get availableCommands() {
      return aAvailableCommands;
    },
    mClosePromptPicker() {
      vIsPromptPickerOpen = false;
      vPromptSearchQuery = '';
    },
    get commandQuery() {
      return vCommandQuery;
    },
    set commandQuery(pV: string) {
      vCommandQuery = pV;
    },
    dispatchCommand: fDispatchCommand,
    handleCommandPickerClose: fHandleCommandPickerClose,
    handleCommandSelect: fHandleCommandSelect,
    handleInput: fHandleInput,
    // True when a picker consumed the event, so the form skips submit.
    handleKeydown: fHandleKeydown,
    handleMentionPickerClose: fHandleMentionPickerClose,
    handlePromptPickerClose: fHandlePromptPickerClose,
    handleWorkingDirectoryClose: fHandleWorkingDirectoryClose,
    handleWorkingDirectoryOpen: fHandleWorkingDirectoryOpen,
    get isCommandPickerOpen() {
      return vIsCommandPickerOpen;
    },
    set isCommandPickerOpen(pV: boolean) {
      vIsCommandPickerOpen = pV;
    },
    get isMentionPickerOpen() {
      return vIsMentionPickerOpen;
    },
    set isMentionPickerOpen(pV: boolean) {
      vIsMentionPickerOpen = pV;
    },
    get isPromptPickerOpen() {
      return vIsPromptPickerOpen;
    },
    set isPromptPickerOpen(pV: boolean) {
      vIsPromptPickerOpen = pV;
    },
    get isWorkingDirectoryPickerOpen() {
      return vIsWorkingDirectoryPickerOpen;
    },
    set isWorkingDirectoryPickerOpen(pV: boolean) {
      vIsWorkingDirectoryPickerOpen = pV;
    },
    get mentionQuery() {
      return vMentionQuery;
    },
    set mentionQuery(pV: string) {
      vMentionQuery = pV;
    },
    get mentionScopePath() {
      return cMentionScopePath;
    },
    mOpenPromptPicker() {
      vIsPromptPickerOpen = true;
    },
    get promptSearchQuery() {
      return vPromptSearchQuery;
    },
    set promptSearchQuery(pV: string) {
      vPromptSearchQuery = pV;
    },
    get workingDirectoryQuery() {
      return vWorkingDirectoryQuery;
    },
    set workingDirectoryQuery(pV: string) {
      vWorkingDirectoryQuery = pV;
    }
  };
}

export type UseChatFormPickersReturn = ReturnType<typeof fUseChatFormPickers>;
