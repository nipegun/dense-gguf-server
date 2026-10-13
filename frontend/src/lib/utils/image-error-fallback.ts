/**
 * Simplified HTML fallback for external images that fail to load.
 * Displays a centered message with a link to open the image in a new tab.
 */
export function fGetImageErrorFallbackHtml(pSrc: string): string {
  return `<div class="image-error-content">
		<span>Image cannot be displayed</span>
		<a href="${pSrc}" target="_blank" rel="noopener noreferrer">(open link)</a>
	</div>`;
}
