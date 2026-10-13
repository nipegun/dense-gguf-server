import { cContextKeyChatFormActions } from '$lib/constants';
import type { ChatFormActionsContext } from '$lib/types';
import { getContext, setContext } from 'svelte';

const cChatFormActionsKey = Symbol.for(cContextKeyChatFormActions);

/**
 * Sets the chat form actions context. Call in the parent component (ChatFormActions.svelte).
 */
export function fSetChatFormActionsContext(ctx: ChatFormActionsContext): ChatFormActionsContext {
  return setContext(cChatFormActionsKey, ctx);
}

/**
 * Gets the chat form actions context. Call in child components.
 */
export function fGetChatFormActionsContext(): ChatFormActionsContext {
  return getContext(cChatFormActionsKey);
}
