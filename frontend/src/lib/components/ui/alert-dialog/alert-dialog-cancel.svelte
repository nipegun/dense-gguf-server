<script lang="ts">
  import { fButtonVariants } from '$lib/components/ui/button/index.js';
  import { fCn } from '$lib/components/ui/utils.js';
  import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';

  let {
    class: className,
    ref = $bindable(null),
    ...restProps
  }: AlertDialogPrimitive.CancelProps = $props();
</script>

<AlertDialogPrimitive.Cancel
  bind:ref
  class={fCn(fButtonVariants({ variant: 'outline' }), className)}
  data-slot="alert-dialog-cancel"
  {...restProps}
/>
