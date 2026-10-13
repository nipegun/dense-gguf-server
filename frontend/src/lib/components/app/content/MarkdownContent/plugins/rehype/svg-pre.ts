import { fCreatePreTransform } from './pre-transform';
import { SVG } from '$lib/constants';

/**
 * Converts svg code blocks to <pre class="svg-block"> for client-side rendering.
 * Also claims xml blocks whose content starts with <svg, since models often emit
 * svg inside an xml fence.
 */
export const fRehypeSvgPre = fCreatePreTransform(
  [SVG.LANGUAGE, SVG.XML_LANGUAGE],
  SVG.BLOCK_CLASS,
  (pText) => pText.startsWith(SVG.TAG_PREFIX)
);
