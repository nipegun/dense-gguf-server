<script lang="ts">
  interface Props {
    class?: string;
    multiple?: boolean;
    onFileSelect?: (pFiles: File[]) => void;
  }

  let { class: className = '', multiple = true, onFileSelect }: Props = $props();

  let vFileInputElement: HTMLInputElement | undefined;

  export function click() {
    vFileInputElement?.click();
  }

  function fHandleFileSelect(pEvent: Event) {
    const input = pEvent.target as HTMLInputElement;

    if (input.files) {
      onFileSelect?.(Array.from(input.files));
    }
  }
</script>

<input
  bind:this={vFileInputElement}
  class="hidden {className}"
  {multiple}
  onchange={fHandleFileSelect}
  type="file"
/>
