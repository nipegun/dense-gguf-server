<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Pencil, RefreshCw, Trash2 } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    isHealthChecking: boolean;
    onEdit: () => void;
    onRefresh: () => void;
    onDelete: () => void;
  }

  let { isHealthChecking, onDelete, onEdit, onRefresh }: Props = $props();
</script>

<div class="flex shrink-0 items-center gap-1">
  <Button
    aria-label={fTranslate('text_464c4ffd019e')}
    class="h-7 w-7"
    onclick={onEdit}
    size="icon"
    variant="ghost"
  >
    <Pencil class="h-3.5 w-3.5" />
  </Button>

  <Button
    aria-label={fTranslate('text_0e9161011702')}
    class="h-7 w-7"
    disabled={isHealthChecking}
    onclick={onRefresh}
    size="icon"
    variant="ghost"
  >
    <RefreshCw class="h-3.5 w-3.5" />
  </Button>

  <Button
    aria-label={fTranslate('text_e2d0a54968ea')}
    class="hover:text-destructive-foreground h-7 w-7 text-destructive hover:bg-destructive/10"
    onclick={onDelete}
    size="icon"
    variant="ghost"
  >
    <Trash2 class="h-3.5 w-3.5" />
  </Button>
</div>
