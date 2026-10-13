import { cContextKeyChatMessageEdit } from '$lib/constants';
import type { ChatMessageEditContext } from '$lib/types';
import { getContext, setContext } from 'svelte';

const cChatMessageEditKey = Symbol.for(cContextKeyChatMessageEdit);

/**
 * Sets the message edit context. Call this in the parent component (ChatMessage.svelte).
 */
export function fSetChatMessageEditContext(ctx: ChatMessageEditContext): ChatMessageEditContext {
  return setContext(cChatMessageEditKey, ctx);
}

/**
 * Gets the message edit context. Call this in child components.
 */
export function fGetChatMessageEditContext(): ChatMessageEditContext {
  return getContext(cChatMessageEditKey);
}
