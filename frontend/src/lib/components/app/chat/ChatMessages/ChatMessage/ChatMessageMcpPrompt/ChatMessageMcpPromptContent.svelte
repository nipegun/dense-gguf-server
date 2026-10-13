<script lang="ts">
  import { TruncatedText } from '$lib/components/app/misc';
  import { Card } from '$lib/components/ui/card';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { McpPromptVariant } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
  import { SvelteMap } from 'svelte/reactivity';

  interface ContentPart {
    text: string;
    argKey: string | null;
  }

  interface Props {
    class?: string;
    prompt: DatabaseMessageExtraMcpPrompt;
    variant?: McpPromptVariant;
    isLoading?: boolean;
    loadError?: string;
  }

  let {
    class: className = '',
    isLoading = false,
    loadError,
    prompt,
    variant = McpPromptVariant.MESSAGE
  }: Props = $props();

  let vHoveredArgKey = $state<string | null>(null);
  let aArgumentEntries = $derived(Object.entries(prompt.arguments ?? {}));
  let vHasArguments = $derived(prompt.arguments && Object.keys(prompt.arguments).length > 0);
  let vHasContent = $derived(prompt.content && prompt.content.trim().length > 0);

  let aContentParts = $derived.by((): ContentPart[] => {
    if (!prompt.content || !vHasArguments) {
      return [{ argKey: null, text: prompt.content || '' }];
    }

    const aParts: ContentPart[] = [];

    let vRemaining = prompt.content;

    const cValueToKey = new SvelteMap<string, string>();

    for (const [vKey, vValue] of aArgumentEntries) {
      if (vValue && vValue.trim()) {
        cValueToKey.set(vValue, vKey);
      }
    }

    const aSortedValues = [...cValueToKey.keys()].sort((pA, pB) => pB.length - pA.length);

    while (vRemaining.length > 0) {
      let vEarliestMatch: { index: number; value: string; key: string } | null = null;

      for (const cValue of aSortedValues) {
        const cIndex = vRemaining.indexOf(cValue);

        if (cIndex !== -1 && (vEarliestMatch === null || cIndex < vEarliestMatch.index)) {
          vEarliestMatch = { index: cIndex, key: cValueToKey.get(cValue)!, value: cValue };
        }
      }

      if (vEarliestMatch) {
        if (vEarliestMatch.index > 0) {
          aParts.push({ argKey: null, text: vRemaining.slice(0, vEarliestMatch.index) });
        }

        aParts.push({ argKey: vEarliestMatch.key, text: vEarliestMatch.value });
        vRemaining = vRemaining.slice(vEarliestMatch.index + vEarliestMatch.value.length);
      } else {
        aParts.push({ argKey: null, text: vRemaining });

        break;
      }
    }

    return aParts;
  });

  let vShowArgBadges = $derived(vHasArguments && !isLoading && !loadError);
  let vIsAttachment = $derived(variant === McpPromptVariant.ATTACHMENT);
  let vTextSizeClass = $derived(vIsAttachment ? 'text-xs' : 'text-md');
  let vPaddingClass = $derived(vIsAttachment ? 'px-3 py-2' : 'px-3.75 py-2.5');
  let vMaxHeightStyle = $derived(
    vIsAttachment ? 'max-height: 6rem;' : 'max-height: var(--max-message-height);'
  );

  const cServerFavicon = $derived(cMcpStore.mGetServerFavicon(prompt.serverName));
  const cServerDisplayName = $derived(cMcpStore.mGetServerDisplayName(prompt.serverName));
</script>

<div class="flex flex-col gap-2 {className}">
  <div class="flex items-center justify-between gap-2">
    <div class="inline-flex flex-wrap items-center gap-1.25 text-xs text-muted-foreground">
      <Tooltip.fRoot>
        <Tooltip.Trigger>
          {#if cServerFavicon}
            <img
              alt=""
              class="h-3.5 w-3.5 shrink-0 rounded-sm"
              onerror={(pE) => {
                (pE.currentTarget as HTMLImageElement).style.display = 'none';
              }}
              src={cServerFavicon}
            />
          {/if}
        </Tooltip.Trigger>

        <Tooltip.Content>
          <span>{cServerDisplayName}</span>
        </Tooltip.Content>
      </Tooltip.fRoot>

      <TruncatedText text={prompt.name} />
    </div>

    {#if vShowArgBadges}
      <div class="flex flex-wrap justify-end gap-1">
        {#each aArgumentEntries as [vKey, vValue] (vKey)}
          <Tooltip.fRoot>
            <Tooltip.Trigger>
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <span
                class="rounded-sm bg-purple-200/60 px-1.5 py-0.5 text-[10px] leading-none text-purple-700 transition-opacity dark:bg-purple-800/40 dark:text-purple-300 {vHoveredArgKey &&
                vHoveredArgKey !== vKey
                  ? 'opacity-30'
                  : ''}"
                onmouseenter={() => (vHoveredArgKey = vKey)}
                onmouseleave={() => (vHoveredArgKey = null)}
              >
                {vKey}
              </span>
            </Tooltip.Trigger>

            <Tooltip.Content>
              <span class="max-w-xs break-all">{vValue}</span>
            </Tooltip.Content>
          </Tooltip.fRoot>
        {/each}
      </div>
    {/if}
  </div>

  {#if loadError}
    <Card
      class="relative overflow-hidden rounded-[1.125rem] border border-destructive/50 bg-destructive/10 backdrop-blur-md"
    >
      <div
        class="overflow-y-auto {vPaddingClass}"
        style="{vMaxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
      >
        <span class="{vTextSizeClass} text-destructive">{loadError}</span>
      </div>
    </Card>
  {:else if isLoading}
    <Card
      class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 px-1 py-2 backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
    >
      <div
        class="overflow-y-auto {vPaddingClass}"
        style="{vMaxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
      >
        <div class="space-y-2">
          <div class="h-3 w-3/4 animate-pulse rounded bg-foreground/20"></div>

          <div class="h-3 w-full animate-pulse rounded bg-foreground/20"></div>

          <div class="h-3 w-5/6 animate-pulse rounded bg-foreground/20"></div>
        </div>
      </div>
    </Card>
  {:else if vHasContent}
    <Card
      class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 py-0 text-foreground backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
    >
      <div
        class="overflow-y-auto {vPaddingClass}"
        style="{vMaxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
      >
        <span class="{vTextSizeClass} whitespace-pre-wrap">
          <!-- This formatting is needed to keep the text in proper shape -->
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          {#each aContentParts as vPart, vI (vI)}{#if vPart.argKey}<span
                class="rounded-sm bg-purple-300/50 px-0.5 text-purple-900 transition-opacity dark:bg-purple-700/50 dark:text-purple-100 {vHoveredArgKey &&
                vHoveredArgKey !== vPart.argKey
                  ? 'opacity-30'
                  : ''}"
                onmouseenter={() => (vHoveredArgKey = vPart.argKey)}
                onmouseleave={() => (vHoveredArgKey = null)}>{vPart.text}</span
              >{:else}<span class="transition-opacity {vHoveredArgKey ? 'opacity-30' : ''}"
                >{vPart.text}</span
              >{/if}{/each}</span
        >
      </div>
    </Card>
  {/if}
</div>
