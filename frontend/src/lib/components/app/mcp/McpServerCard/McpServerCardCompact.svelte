<script lang="ts">
  import * as Card from '$lib/components/ui/card';
  import type { RecommendedMCPServer } from '$lib/types';
  import { mode } from 'mode-watcher';

  interface Props {
    server: RecommendedMCPServer;
    onClick?: () => void;
    selected?: boolean;
    dimmed?: boolean;
  }

  let { dimmed = false, onClick, selected = false, server }: Props = $props();

  let vActiveIconUrl = $derived.by(() => {
    const cIsDark = mode.current === 'dark';

    if (cIsDark && server.iconUrlDark) return server.iconUrlDark;

    if (!cIsDark && server.iconUrlLight) return server.iconUrlLight;

    return server.iconUrl;
  });
</script>

<Card.Root
  class={`relative gap-3! select-none bg-muted/30 p-4 transition-all ${onClick ? 'cursor-pointer hover:bg-muted/50 hover:opacity-100' : ''} ${selected ? 'bg-muted/30 ring-1 ring-primary/40' : ''} ${dimmed ? 'opacity-50' : ''}`}
  onclick={onClick}
>
  <div class="flex min-w-0 items-center gap-2">
    {#if vActiveIconUrl}
      <img
        alt=""
        class="h-5 w-5 shrink-0 rounded"
        decoding="async"
        loading="lazy"
        src={vActiveIconUrl}
      />
    {/if}

    <h4 class="min-w-0 flex-1 truncate font-medium">{server.name}</h4>
  </div>

  <p class="text-xs text-muted-foreground">{server.description}</p>
</Card.Root>
