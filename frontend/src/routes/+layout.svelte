<script lang="ts">
  import { cProjectAuthor } from '$lib/constants/app.constants';
  import StatusBar from '$lib/components/app/monitor/StatusBar.svelte';
  import DialogLocalShellAskpass from '$lib/components/app/dialogs/DialogLocalShellAskpass.svelte';
  import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
  import { fGetLanguage } from '$lib/i18n';
  import { fApiUrl } from '$lib/api-url';
  import '../app.css';
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import { base } from '$app/paths';
  import { page } from '$app/state';
  import { SidebarNavigation } from '$lib/components/app';
  import { PwaMetaTags, PwaRefreshAlert } from '$lib/components/pwa';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import {
    cFaviconPaths,
    cFaviconSelectors,
    HEADERS,
    cNewChatTabId,
    cRoutes,
    cSettingsKeys,
    cTooltipDelayDuration
  } from '$lib/constants';
  import { fUseKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
  import { fUsePwa } from '$lib/hooks/use-pwa.svelte';
  import { RouterService } from '$lib/services/router.service';
  import {
    cChatStore,
    cConversationsStore,
    cDeviceStore,
    cMcpStore,
    cModelsStore,
    cServerStore,
    cSettingsStore,
    cTabsStore,
    cVersionStore
  } from '$lib/stores';
  import { fInitStores } from '$lib/stores/init';
  import { ModeWatcher } from 'mode-watcher';
  import { untrack } from 'svelte';
  import { onMount } from 'svelte';
  import { Toaster } from 'svelte-sonner';
  import { pwaAssetsHead } from 'virtual:pwa-assets/head';

  let { children } = $props();

  // migrations and store startup, ordered explicitly instead of import side effects
  void fInitStores();

  let vInnerHeight = $state<number | undefined>();
  let vInnerWidth = $state(browser ? window.innerWidth : 0);

  let vChatSidebar:
    | {
        activateSearchMode?: () => void;
        editActiveConversation?: () => void;
      }
    | undefined = $state();

  let vShowBuildVersion = $derived(
    cSettingsStore.vConfig[cSettingsKeys.SHOW_BUILD_VERSION] as boolean
  );

  // Keep the hook object intact: destructuring needRefreshByStorage reads the getter once and freezes it
  const cPwa = fUsePwa();
  const { needRefresh: vNeedRefresh, updateServiceWorker: fUpdateServiceWorker } = cPwa;

  function fUpdateFavicon() {
    const dark = cDeviceStore.vSystemTheme.isDark;

    let vIcoLink = document.querySelector(cFaviconSelectors.ICO_48X48) as HTMLLinkElement | null;

    if (vIcoLink) {
      vIcoLink.href = dark ? cFaviconPaths.ICO_DARK : cFaviconPaths.ICO_LIGHT;
    }

    let vSvgLink = document.querySelector(cFaviconSelectors.SVG_ANY) as HTMLLinkElement | null;

    if (vSvgLink) {
      vSvgLink.href = dark ? cFaviconPaths.SVG_DARK : cFaviconPaths.SVG_LIGHT;
    }
  }

  function fNavigateToTab(pDirection: -1 | 1) {
    // only makes sense with conversation tabs enabled
    if (!cSettingsStore.vConfig.conversationTabs) return;

    const aOpenTabs = cTabsStore.aOpenTabs;

    if (aOpenTabs.length === 0) return;

    const cActiveId = page.params.id ?? cNewChatTabId;
    const cIdx = aOpenTabs.indexOf(cActiveId);
    // active tab not in list (e.g. a non-chat route): start from an edge
    const cTargetIdx =
      cIdx === -1
        ? pDirection === 1
          ? 0
          : aOpenTabs.length - 1
        : (cIdx + pDirection + aOpenTabs.length) % aOpenTabs.length;

    void cTabsStore.mActivate(aOpenTabs[cTargetIdx]);
  }

  function fNavigateToConversation(pDirection: -1 | 1) {
    const aAllConvs = cConversationsStore.aConversations;

    if (aAllConvs.length === 0) return;

    const cCurrentId = page.params.id;

    if (!cCurrentId) {
      goto(RouterService.mChat(aAllConvs[pDirection === 1 ? 0 : aAllConvs.length - 1].id));

      return;
    }

    const cIdx = aAllConvs.findIndex((pC) => pC.id === cCurrentId);

    if (cIdx === -1) return;

    const cTargetIdx = cIdx + pDirection;

    if (cTargetIdx >= 0 && cTargetIdx < aAllConvs.length) {
      goto(RouterService.mChat(aAllConvs[cTargetIdx].id));
    } else {
      cConversationsStore.mOpenNewChat();
    }
  }

  // navigating away from the new-chat screen drops its tab, so it does not
  // linger once the user moves to a real conversation or another route
  let vPreviousChatId = $state<string | undefined>(undefined);

  $effect(() => {
    const cId = page.params.id ?? (page.route.id === '/(chat)' ? cNewChatTabId : undefined);
    const cPrev = untrack(() => vPreviousChatId);

    vPreviousChatId = cId;

    if (
      cId !== cPrev &&
      cPrev &&
      cSettingsStore.vConfig.conversationTabs &&
      cPrev === cNewChatTabId
    ) {
      untrack(() => cTabsStore.mRemoveTabs([cNewChatTabId]));
    }
  });
  // Global keyboard shortcuts
  const { handleKeydown: fHandleKeydown } = fUseKeyboardShortcuts({
    editActiveConversation: () => vChatSidebar?.editActiveConversation?.(),
    navigateToNextConversation: () => fNavigateToConversation(1),
    navigateToNextTab: () => fNavigateToTab(1),
    navigateToPrevConversation: () => fNavigateToConversation(-1),
    navigateToPrevTab: () => fNavigateToTab(-1)
  });

  function fCheckApiKey() {
    const cApiKey = cSettingsStore.vConfig.apiKey;

    // Without a stored key there is nothing to re-validate here; the keyless
    // 401 case is handled by validateApiKey() at navigation time, and the
    // reload below must never fire in a keyless loop.
    if (!cApiKey || cApiKey.trim() === '') {
      return;
    }

    untrack(() => {
      if (
        (page.route.id === '/(chat)' || page.route.id === '/(chat)/chat/[id]') &&
        page.status !== 401 &&
        page.status !== 403
      ) {
        const dHeaders: Record<string, string> = {
          'Content-Type': 'application/json',
          [HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${cApiKey.trim()}`
        };

        fetch(fApiUrl('/props'), { headers: dHeaders })
          .then((pResponse) => {
            if (pResponse.status === 401 || pResponse.status === 403) {
              window.location.reload();
            }
          })
          .catch((pE) => {
            console.error('Error checking API key:', pE);
          });
      }
    });
  }

  onMount(() => {
    document.documentElement.lang = fGetLanguage();
    document.documentElement.dir = fGetLanguage() === 'he-IL' ? 'rtl' : 'ltr';
    fUpdateFavicon();
    // snapshot of every backend running stream on first load, populates the sidebar spinners
    // so the user sees each conv that has a live inference, even ones not opened yet
    void cChatStore.mSyncRemoteRunningStreams();
  });

  // refresh that snapshot when the tab returns to the foreground, a stream may have advanced
  // or ended while it was hidden. snapshot only, no polling
  function fHandleVisibilityChange() {
    if (document.visibilityState !== 'visible') return;

    void cChatStore.mSyncRemoteRunningStreams();
  }

  $effect(() => {
    void cDeviceStore.vSystemTheme.isDark;

    fUpdateFavicon();
  });

  // Initialize server properties on app load (run once)
  $effect(() => {
    // Only fetch if we don't already have props
    if (!cServerStore.props) {
      untrack(() => {
        cServerStore.mFetch();
      });
    }
  });

  // Sync settings when server props are loaded
  $effect(() => {
    const cServerProps = cServerStore.props;

    if (cServerProps && cUiSettingsStore.vIsLoaded && cSettingsStore.vIsInitialized) {
      untrack(() => {
        cSettingsStore.mSyncWithServerDefaults();
        cMcpStore.mSyncShellServer(cServerProps.shell_mcp_server === true);
      });
    }
  });

  $effect(() => {
    if (cUiSettingsStore.vIsLoaded)
      untrack(() => {
        void fInitStores();
      });
    if (browser)
      document.documentElement.style.setProperty(
        '--status-bar-height',
        cSettingsStore.vConfig.showStatusBar
          ? 'calc(1.75rem + env(safe-area-inset-bottom, 0px))'
          : '0px'
      );
  });

  // Inject custom CSS at runtime through an action on the head style node
  // textContent keeps the value as text, never parsed as HTML
  function fCustomCss(node: HTMLStyleElement) {
    $effect(() => {
      node.textContent = (cSettingsStore.vConfig.customCss as string | undefined) ?? '';
    });
  }

  // Fetch router models when in router mode (for status and modalities)
  // Wait for models to be loaded first, run only once
  let vRouterModelsFetched = false;

  $effect(() => {
    const cIsRouter = cServerStore.isRouterMode;
    const cModelsCount = cModelsStore.aModels.length;

    // Only fetch router models once when we have models loaded and in router mode
    if (cIsRouter && cModelsCount > 0 && !vRouterModelsFetched) {
      vRouterModelsFetched = true;

      untrack(() => {
        cModelsStore.mFetchRouterModels();
      });
    }
  });

  // Live model status and load progress via the /models/sse feed (router mode)
  $effect(() => {
    if (!browser) return;

    if (!cServerStore.isRouterMode) return;

    untrack(() => {
      cModelsStore.status.mSubscribe();
    });

    return () => {
      cModelsStore.status.mUnsubscribe();
    };
  });

  // Background MCP server health checks on app load.
  // Health-check every configured server with a URL - including disabled ones -
  // so the /mcp-servers page can display health metadata for servers that are
  // currently turned off. Disabled servers never get promoted to active
  // connections (see runHealthCheck), so their tools/prompts/resources stay
  // out of the chat-side stores.
  // Only IDLE servers are checked; already-resolved (SUCCESS / ERROR) servers
  // keep their existing state, so adding or removing a server does not flash
  // every other card back through skeleton state.
  $effect(() => {
    if (!browser) return;

    const aMcpServers = cMcpStore.mGetServers();
    const aServersWithUrls = aMcpServers.filter((pS) => pS.url.trim());

    if (aServersWithUrls.length > 0) {
      untrack(() => {
        // Run health checks in background (don't await)
        cMcpStore.mRunHealthChecksForServers(aServersWithUrls, true).catch((pError) => {
          console.warn('[layout] MCP health checks failed:', pError);
        });
      });
    }
  });

  // Monitor API key changes and redirect to error page if removed or changed when required
  $effect(() => {
    fCheckApiKey();
  });
</script>

<svelte:head>
  <meta name="author" content={cProjectAuthor} />
  {#if pwaAssetsHead.themeColor}
    <meta content={pwaAssetsHead.themeColor.content} name="theme-color" />
  {/if}

  {#if cSettingsStore.vConfig.customCss}
    <style use:customCss></style>
  {/if}

  {#each pwaAssetsHead.links as vLink (vLink.href)}
    <link {...vLink} />
  {/each}

  <PwaMetaTags />
</svelte:head>

<svelte:window
  bind:innerHeight={vInnerHeight}
  bind:innerWidth={vInnerWidth}
  onkeydown={fHandleKeydown}
/>
<svelte:document onvisibilitychange={fHandleVisibilityChange} />

<Tooltip.fProvider delayDuration={cTooltipDelayDuration}>
  <div class="flex flex-col md:flex-row">
    <SidebarNavigation
      onSearchClick={() => {
        if (cDeviceStore.isMobile) {
          goto(cRoutes.SEARCH);
        } else if (vChatSidebar?.activateSearchMode) {
          vChatSidebar.activateSearchMode();
        }
      }}
    />

    <!-- min-w-0 lets the chat column shrink below its content width, so wide
		     code blocks and tables scroll inside their own containers instead of
		     stretching the page into a horizontal scrollbar -->
    <div class="min-w-0 flex-1">
      {@render children?.()}
    </div>
  </div>

  <ModeWatcher />

  <DialogLocalShellAskpass />
  {#if cSettingsStore.vConfig.showStatusBar}<StatusBar />{/if}
  <Toaster closeButton richColors />
</Tooltip.fProvider>

<!-- PWA update prompt + version -->
<div
  style="bottom: calc(1rem + var(--status-bar-height, 0px))"
  class="fixed right-4 z-9999 flex flex-col items-end gap-1"
>
  {#if vShowBuildVersion && cVersionStore.vBuild}
    <span class="text-[10px] tabular-nums text-muted-foreground">{cVersionStore.vBuild}</span>
  {/if}

  <PwaRefreshAlert
    forceReload={cPwa.needRefreshByStorage}
    needRefresh={$vNeedRefresh || cPwa.needRefreshByStorage}
    updateServiceWorker={fUpdateServiceWorker}
  />
</div>
