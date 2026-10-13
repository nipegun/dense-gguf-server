/**
 * STORES
 *
 * Reactive Svelte runes state layer. Stores own application state and
 * expose it as plain Svelte 5 runes (`$state`, `$derived`, `$effect`),
 * consumed by components, routes, hooks and services.
 *
 * Import from this barrel in leaf consumers:
 *
 * ```ts
 * import { chatStore, modelsStore } from '$lib/stores';
 * ```
 *
 * Store modules keep direct imports between each other (and from services/
 * utils they depend on) to avoid circular dependency chains.
 *
 * Each store below documents its primary responsibility.
 */

// CHAT / MESSAGING
export { cChatStore } from './chat/index.svelte';

export { cDraftMessagesStore } from './chat/drafts.svelte';

// CONVERSATION TABS
export { cTabsStore } from './tabs.svelte';

// CONTEXT STATS (active conversation context window usage)
export { cContextStatsStore } from './chat/context-stats.svelte';

// AGENTIC (multi-turn tool orchestration)
export { cAgenticStore } from './agentic/index.svelte';

// CONVERSATIONS
export { cConversationsStore } from './conversations/index.svelte';

// MCP
export { cMcpStore } from './mcp/index.svelte';

// MODELS
export { cModelsStore } from './models/index.svelte';

// SERVER
export { cServerStore } from './server.svelte';

// UI / LAYOUT
export { cUiStore } from './ui.svelte';

// SETTINGS / UI PREFERENCES
export { cSettingsStore } from './settings/index.svelte';

export { cPermissionsStore } from './permissions.svelte';

// TOOLS
export { cToolsStore } from './tools.svelte';

// ENVIRONMENT / META
export { cVersionStore } from './version.svelte';

export { cDeviceStore } from './device.svelte';
