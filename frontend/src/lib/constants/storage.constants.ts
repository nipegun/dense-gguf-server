/**
 * Storage-related constants (localStorage, IndexedDB).
 *
 * Centralized to ensure consistency across the app and simplify future
 * name changes.
 */

/** Name prefix for all localStorage keys */
export const cStorageAppName = 'DgsUi';

export const cAlwaysAllowedToolsLocalstorageKey = `${cStorageAppName}.alwaysAllowedTools`;

/** Paused model download ids (`<repo>:<tag>`), restored on the next page load. */
export const cPausedModelDownloadsLocalstorageKey = `${cStorageAppName}.pausedModelDownloads`;
export const cConfigLocalstorageKey = `${cStorageAppName}.config`;
export const cDisabledToolsLocalstorageKey = `${cStorageAppName}.disabledTools`;

/** Disabled tools keyed by stable selection identity, no migration from the name based key */
export const cDisabledToolKeysLocalstorageKey = `${cStorageAppName}.disabledToolKeys`;

/** Default disabled tool categories, seeded into newly created conversations */
export const cDisabledToolCategoriesLocalstorageKey = `${cStorageAppName}.disabledToolCategories`;
export const cFavoriteModelsLocalstorageKey = `${cStorageAppName}.favoriteModels`;

/** Open state the user set for a model list section or one of its families, by id. */
export const cModelGroupOpenLocalstorageKey = `${cStorageAppName}.modelGroupOpen`;

/** Recently used model ids, most recent first, backend-qualified. */
export const cRecentModelsLocalstorageKey = `${cStorageAppName}.recentModels`;

/** Model ids kept out of the selector, backend-qualified. */
export const cHiddenModelsLocalstorageKey = `${cStorageAppName}.hiddenModels`;
export const cReasoningEffortDefaultLocalstorageKey = `${cStorageAppName}.reasoningEffortDefault`;
export const cConversationTabsLocalstorageKey = `${cStorageAppName}.conversationTabs`;
export const cUserOverridesLocalstorageKey = `${cStorageAppName}.userOverrides`;
export const cDismissedRecommendedMcpServersLocalstorageKey = `${cStorageAppName}.dismissedRecommendedMcpServers`;

/** Key prefix for per-conversation resumable stream state, conversationId is appended */
export const cStreamResumeLocalstorageKeyPrefix = `${cStorageAppName}.streamResume.`;

/** Build version stored in localStorage for non-PWA update detection */
export const cBuildVersionLocalstorageKey = `${cStorageAppName}.buildVersion`;
