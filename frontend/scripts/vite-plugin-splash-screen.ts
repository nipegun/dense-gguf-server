import {
  cAppleDevices,
  cBuildConfig,
  cRegexPatterns,
  cSplashLink
} from '../src/lib/constants/pwa.constants';
import { cNewline, cTab } from '../src/lib/constants/special-characters.constants';
import { SplashOrientation } from '../src/lib/enums/splash.enums';
import type { SplashDimensions } from '../src/lib/types';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';

let vProcessed = false;

const cOutputDir = process.env.DGS_UI_OUT_DIR ?? cBuildConfig.OUTPUT_DIR;

/**
 * Generate iOS splash screen <link> tags from generated apple-splash-*.png files.
 * Returns an array of HTML link strings to be injected into the page head.
 */
export function fGenerateSplashScreenLinks(pOutDir: string): string[] {
  const aFiles = readdirSync(pOutDir).filter((pF) => pF.match(cRegexPatterns.SPLASH_FILE));

  if (aFiles.length === 0) return [];

  const dDimMap = new Map<string, SplashDimensions>();

  for (const [vDims, vSpec] of Object.entries(cAppleDevices)) {
    const [vW, vH] = vDims.split('x').map(Number);

    // logical-point dimensions
    dDimMap.set(`${vW}x${vH}`, { deviceH: vSpec.height, deviceW: vSpec.width, dpr: vSpec.dpr });
    dDimMap.set(`${vH}x${vW}`, { deviceH: vSpec.height, deviceW: vSpec.width, dpr: vSpec.dpr });
    // pixel dimensions (used by actual generated splash files)
    dDimMap.set(`${vW * vSpec.dpr}x${vH * vSpec.dpr}`, {
      deviceH: vSpec.height,
      deviceW: vSpec.width,
      dpr: vSpec.dpr
    });
    dDimMap.set(`${vH * vSpec.dpr}x${vW * vSpec.dpr}`, {
      deviceH: vSpec.height,
      deviceW: vSpec.width,
      dpr: vSpec.dpr
    });
  }

  const aLightLinks: string[] = [];
  const aDarkLinks: string[] = [];

  for (const file of aFiles) {
    const cMatch = file.match(cRegexPatterns.SPLASH_FILE);

    if (!cMatch) continue;

    const cOrientation = cMatch[1] as SplashOrientation;
    const cIsDark = !!cMatch[2];
    const cPixelW = parseInt(cMatch[3]);
    const cPixelH = parseInt(cMatch[4]);
    const cKey = `${cPixelW}x${cPixelH}`;
    const cSpec = dDimMap.get(cKey);

    if (!cSpec) {
      console.warn(`Unknown splash screen dimensions: ${cKey} (${file})`);

      continue;
    }

    const { deviceH: vDeviceH, deviceW: vDeviceW, dpr: vDpr } = cSpec;
    const cMedia = `screen and (device-width: ${vDeviceW}px) and (device-height: ${vDeviceH}px) and (-webkit-device-pixel-ratio: ${vDpr}) and (orientation: ${cOrientation})`;
    const cHref = `./${file}`;

    if (cIsDark) {
      aDarkLinks.push(
        `${cSplashLink.HTML} media="${cMedia}${cSplashLink.DARK_MEDIA_SUFFIX}" href="${cHref}">`
      );
    } else {
      aLightLinks.push(`${cSplashLink.HTML} media="${cMedia}" href="${cHref}">`);
    }
  }

  return [...aLightLinks, ...aDarkLinks];
}

export function fSplashScreenPlugin(): Plugin {
  return {
    apply: 'build',
    closeBundle() {
      setTimeout(() => {
        try {
          if (vProcessed) return;

          vProcessed = true;

          const cOutDir = resolve(cOutputDir);
          const cIndexPath = resolve(cOutDir, 'index.html');

          if (!existsSync(cIndexPath)) return;

          let vContent = readFileSync(cIndexPath, 'utf-8');

          // Inject iOS splash screen <link> tags into <head>.
          // The @vite-pwa/assets-generator generates apple-splash-*.png files;
          // this scans them and creates the <link> tags SvelteKit needs.
          const aSplashLinks = fGenerateSplashScreenLinks(cOutDir);

          if (aSplashLinks.length > 0) {
            console.log(`Generated ${aSplashLinks.length} apple-splash link tags`);
            const cSplashHtml = aSplashLinks.map((pL) => cTab + cTab + pL).join(cNewline);

            vContent = vContent.replace(
              cRegexPatterns.HEAD_CLOSE,
              cSplashHtml + cNewline + cTab + cTab + '</head>'
            );
          }

          // Remove trailing \r from Windows line endings
          vContent = vContent.replace(/\r/g, '');
          vContent = cBuildConfig.GUIDE_COMMENT + cNewline + vContent;

          writeFileSync(cIndexPath, vContent, 'utf-8');
          console.log('Updated index.html');
        } catch (error) {
          console.error('Failed to process build output:', error);
        }
      }, 100);
    },
    name: 'dgscpp:splash-screen'
  };
}
