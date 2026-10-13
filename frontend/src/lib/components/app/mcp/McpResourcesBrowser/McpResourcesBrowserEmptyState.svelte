<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  interface Props {
    isLoading: boolean;
  }

  let { isLoading }: Props = $props();
</script>

<div class="py-4 text-center text-sm text-muted-foreground">
  {#if isLoading}{fTranslate('text_f851843d3fed')}{:else}{fTranslate('text_1e2ad76694e1')}{/if}
</div>
