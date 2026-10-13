<script lang="ts">
  import * as Tooltip from '$lib/components/ui/tooltip';

  interface Props {
    text: string;
    class?: string;
    showTooltip?: boolean;
  }

  let { class: className = '', showTooltip = true, text }: Props = $props();

  let vTextElement: HTMLSpanElement | undefined = $state();
  let vIsTruncated = $state(false);

  function fCheckTruncation() {
    if (vTextElement) {
      vIsTruncated = vTextElement.scrollWidth > vTextElement.clientWidth;
    }
  }

  $effect(() => {
    if (vTextElement) {
      fCheckTruncation();

      const cObserver = new ResizeObserver(fCheckTruncation);

      cObserver.observe(vTextElement);

      return () => cObserver.disconnect();
    }
  });
</script>

{#if vIsTruncated && showTooltip}
  <Tooltip.fRoot>
    <Tooltip.Trigger class="{className} min-w-0">
      <span bind:this={vTextElement} class="block truncate">
        {text}
      </span>
    </Tooltip.Trigger>

    <Tooltip.Content class="z-[9999]">
      <p>{text}</p>
    </Tooltip.Content>
  </Tooltip.fRoot>
{:else}
  <span bind:this={vTextElement} class="{className} block min-w-0 truncate">
    {text}
  </span>
{/if}
