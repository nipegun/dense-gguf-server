export function fPortalToBody(node: HTMLElement) {
  if (typeof document === 'undefined') {
    return;
  }

  const cTarget = document.body;

  if (!cTarget) {
    return;
  }

  cTarget.appendChild(node);

  return {
    mDestroy() {
      if (node.parentNode === cTarget) {
        cTarget.removeChild(node);
      }
    }
  };
}
