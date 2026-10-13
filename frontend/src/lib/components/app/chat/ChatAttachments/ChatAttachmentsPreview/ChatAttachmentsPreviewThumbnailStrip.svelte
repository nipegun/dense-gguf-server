<script lang="ts">
  import { FileText, Music, Video } from '@lucide/svelte';
  import { ScrollCarousel } from '$lib/components/app';
  import { cIconClassDefault, cUiDataAttrs } from '$lib/constants';
  import { ScrollCarouselVariant } from '$lib/enums';

  interface PreviewItem {
    id: string;
    name: string;
    isImage: boolean;
    isAudio: boolean;
    isVideo: boolean;
    preview?: string;
  }

  interface Props {
    items: PreviewItem[];
    currentIndex: number;
    onNavigate: (pIndex: number) => void;
  }

  let { currentIndex, items, onNavigate }: Props = $props();

  function fGetFileExtension(pName: string): string {
    const aParts = pName.split('.');

    if (aParts.length > 1) {
      return aParts.pop()?.toUpperCase() ?? '';
    }

    return '';
  }
</script>

{#if items.length > 1}
  <div class="sticky bottom-0 z-10 mt-4 flex-shrink-0">
    <ScrollCarousel class="max-w-full" variant={ScrollCarouselVariant.CENTER}>
      {#each items as vItem, vIndex (vItem.id)}
        <button
          {...{ [cUiDataAttrs.THUMBNAIL_INDEX]: vIndex }}
          aria-label={`Go to ${vItem.name}`}
          class={[
            'relative flex-shrink-0 cursor-pointer overflow-hidden rounded border-2 bg-black/80 backdrop-blur-sm transition-all hover:opacity-90',
            vIndex === currentIndex ? 'border-white' : 'border-transparent opacity-60',
            '[&:not(:first-child)]:last:mr-4 [&:not(:last-child)]:first:ml-4'
          ]}
          onclick={() => onNavigate(vIndex)}
        >
          {#if vItem.isImage && vItem.preview}
            <img alt={vItem.name} class="h-12 w-12 object-cover" src={vItem.preview} />
          {:else}
            <div
              class="bg-foreground-muted/50 flex h-12 w-12 flex-col items-center justify-center gap-0.5 py-1"
            >
              {#if vItem.isAudio}
                <Music class="{cIconClassDefault} text-white/70" />
              {:else if vItem.isVideo}
                <Video class="{cIconClassDefault} text-white/70" />
              {:else}
                <FileText class="{cIconClassDefault} text-white/70" />
              {/if}

              <span class="font-mono text-[9px] text-white/60">{fGetFileExtension(vItem.name)}</span
              >
            </div>
          {/if}
        </button>
      {/each}
    </ScrollCarousel>
  </div>
{/if}
