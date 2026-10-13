<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Mic, Square } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cIconClassDefault } from '$lib/constants';

  interface Props {
    class?: string;
    disabled?: boolean;
    hasAudioModality?: boolean;
    isLoading?: boolean;
    isRecording?: boolean;
    onMicClick?: () => void;
  }

  let {
    class: className = '',
    disabled = false,
    hasAudioModality = false,
    isLoading = false,
    isRecording = false,
    onMicClick
  }: Props = $props();
</script>

<div class="flex items-center gap-1 {className}">
  <Tooltip.fRoot>
    <Tooltip.Trigger>
      <Button
        class="h-8 w-8 max-md:h-9 max-md:w-9 rounded-full p-0 {isRecording
          ? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
          : ''}"
        disabled={disabled || isLoading || !hasAudioModality}
        onclick={onMicClick}
        type="button"
      >
        <span class="sr-only">{isRecording ? 'Stop recording' : 'Start recording'}</span>

        {#if isRecording}
          <Square class="{cIconClassDefault} animate-pulse fill-white" />
        {:else}
          <Mic class={cIconClassDefault} />
        {/if}
      </Button>
    </Tooltip.Trigger>

    {#if !hasAudioModality}
      <Tooltip.Content>
        <p>{fTranslate('text_5408a261204f')}</p>
      </Tooltip.Content>
    {/if}
  </Tooltip.fRoot>
</div>
