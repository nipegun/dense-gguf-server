/** Sentinel tab id for the bare `#/` new-chat screen */
export const cNewChatTabId = 'new-chat';

/** Label shown for the new-chat sentinel tab. */
export const cNewChatLabel = 'New chat';

/** Fallback label for conversations without an auto-generated title. */
export const cUnnamedChatLabel = 'Chat';

/**
 * Tab bar max width so it stays clear of the sidebar strip. The expanded strip
 * is `md:w-72` and the collapsed one `md:w-12`; these hold the fully tuned
 * `max-w-[calc(100vw-?rem)]` classes so the offset has a single source.
 */
export const cChatTabsMaxWidth = {
  COLLAPSED_SIDEBAR: 'max-w-[calc(100vw-5rem)]',
  EXPANDED_SIDEBAR: 'max-w-[calc(100vw-20rem)]'
} as const;
