/**
 * uiStore - Shared UI/layout state
 *
 * Holds cross-component UI state that does not belong to a single component
 * (e.g. the desktop sidebar's expanded/collapsed state, which the sidebar
 * controls and the chat tab bar reacts to).
 */

import type { ModelOption } from '$lib/types/models';

class UiStore {
  /** Set when a flow outside the chat wants the composer focused once it can take it. */
  vComposerFocusRequested = $state(false);

  /** Whether the desktop sidebar is expanded (open). */
  vIsSidebarExpanded = $state(false);
  /** Model the manager reveals when it opens, a qualified id or a raw model name. */
  vManageModelFocus = $state<string | null>(null);
  /** Open state of the models manager, driven from the sidebar and from model rows. */
  vManageModelsOpen = $state(false);
  /** Model the information dialog shows; a phone opens it instead of the manager. */
  vModelInformation = $state<ModelOption | null>(null);

  /** Open the model information dialog, a phone's stand-in for the manager pane. */
  mOpenModelInformation(pOption: ModelOption): void {
    this.vModelInformation = pOption;
  }

  /** Open the models manager, optionally focused on one model. */
  mOpenModelsManager(pFocus?: string): void {
    this.vManageModelFocus = pFocus ?? null;
    this.vManageModelsOpen = true;
  }

  /** Ask the composer to take focus, e.g. after a dialog closes onto the chat. */
  mRequestComposerFocus(): void {
    this.vComposerFocusRequested = true;
  }
}

export const cUiStore = new UiStore();
