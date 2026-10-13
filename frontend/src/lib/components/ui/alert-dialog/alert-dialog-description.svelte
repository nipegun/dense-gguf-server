<script lang="ts">
  import { fCn } from '$lib/components/ui/utils.js';
  import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';

  let {
    class: className,
    ref = $bindable(null),
    ...restProps
  }: AlertDialogPrimitive.DescriptionProps = $props();
</script>

<AlertDialogPrimitive.Description
  bind:ref
  class={fCn('text-sm text-muted-foreground', className)}
  data-slot="alert-dialog-description"
  {...restProps}
/>
