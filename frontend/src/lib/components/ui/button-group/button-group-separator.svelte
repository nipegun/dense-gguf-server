<script lang="ts">
  import { fCn } from '$lib/components/ui/utils';
  import type { HTMLAttributes } from 'svelte/elements';

  let { ...restProps }: HTMLAttributes<HTMLDivElement> = $props();
</script>

<div class={fCn('shrink-0 self-stretch bg-border', 'w-px')} {...restProps}></div>
