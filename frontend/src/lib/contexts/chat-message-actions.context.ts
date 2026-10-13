import { cContextKeyChatMessageActions } from '$lib/constants';
import type { ChatMessageActionsContext } from '$lib/types';
import { getContext, setContext } from 'svelte';

const cChatMessageActionsKey = Symbol.for(cContextKeyChatMessageActions);

/**
 * Sets the per-message actions context. Call this in the parent component (ChatMessage.svelte).
 */
export function fSetChatMessageActionsContext(
  ctx: ChatMessageActionsContext
): ChatMessageActionsContext {
  return setContext(cChatMessageActionsKey, ctx);
}

/**
 * Gets the per-message actions context. Call this in child components.
 */
export function fGetChatMessageActionsContext(): ChatMessageActionsContext {
  return getContext(cChatMessageActionsKey);
}
