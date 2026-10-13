#!/usr/bin/env node

/**
 * Apply circular mask to pwa-*.png icons.
 * Uses the maskable icon as source (white bg, full logo) to avoid
 * the small-colormap pwa icons looking bad when cropped to a circle.
 *
 * Usage: node scripts/make-icons-circular.js [--padding-pct <0-50>] [--scale-pct <50-100>]
 *
 * - padding-pct: percentage of icon size kept as padding around the circle (default: 25)
 * - scale-pct: scale down the source image before cropping (default: 85)
 *
 * maskable-icon and apple-touch-icon are left untouched.
 */

import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const cStaticDir = path.resolve(__dirname, '..', 'static');
const cPaddingPct = process.argv.reduce((pAcc, pArg, pI, pArgs) => {
  if (pArg === '--padding-pct' && pArgs[pI + 1]) return parseFloat(pArgs[pI + 1]);

  return pAcc;
}, 0);
// Scale down the source image before cropping to circle
const cScalePct = process.argv.reduce((pAcc, pArg, pI, pArgs) => {
  if (pArg === '--scale-pct' && pArgs[pI + 1]) return parseFloat(pArgs[pI + 1]);

  return pAcc;
}, 85); // default 85% - icon fills 85% of the circular area
// Source for circular icons: the maskable icon (white bg, full logo)
const cSourceIcon = 'maskable-icon-512x512.png';
const aTargetIcons = ['pwa-64x64.png', 'pwa-192x192.png', 'pwa-512x512.png'];
// maskable-icon and apple-touch-icon stay square
const aUntouchedIcons = ['maskable-icon-512x512.png', 'apple-touch-icon-180x180.png'];

async function fMakeCircle(pTargetFilename) {
  const cTargetPath = path.join(cStaticDir, pTargetFilename);
  const cSourcePath = path.join(cStaticDir, cSourceIcon);

  if (!fs.existsSync(cSourcePath)) {
    console.log(`⏭️  ${cSourceIcon} not found, skipping`);

    return;
  }

  if (!fs.existsSync(cTargetPath)) {
    console.log(`⏭️  ${pTargetFilename} not found, skipping`);

    return;
  }

  const cMetadata = await sharp(cTargetPath).metadata();
  const cSize = Math.max(cMetadata.width, cMetadata.height);
  const radius = Math.floor((cSize * (1 - cPaddingPct / 100)) / 2);
  const center = Math.floor(cSize / 2);
  // Build circular mask as RGBA buffer: white opaque circle on transparent bg
  const cMaskBuf = Buffer.alloc(cSize * cSize * 4, 0);

  for (let vY = 0; vY < cSize; vY++) {
    for (let vX = 0; vX < cSize; vX++) {
      const cDx = vX - center;
      const cDy = vY - center;
      const cDist = Math.sqrt(cDx * cDx + cDy * cDy);

      if (cDist < radius) {
        const cI = (vY * cSize + vX) * 4;

        cMaskBuf[cI] = 255;
        cMaskBuf[cI + 1] = 255;
        cMaskBuf[cI + 2] = 255;
        cMaskBuf[cI + 3] = 255;
      }
    }
  }

  const cTmpMask = path.join(cStaticDir, '.mask-tmp.png');

  await sharp(cMaskBuf, {
    raw: { channels: 4, height: cSize, width: cSize }
  })
    .png()
    .toFile(cTmpMask);

  // Step 1: Scale source relative to circle diameter (not full icon), composite centered onto white canvas of full size
  const cCircleDiameter = Math.floor(cSize * (1 - cPaddingPct / 100));
  const cScaledSize = Math.floor((cCircleDiameter * cScalePct) / 100);
  const cOffset = Math.floor((cSize - cScaledSize) / 2);
  const cScaledBuf = await sharp(cSourcePath)
    .resize(cScaledSize, cScaledSize, {
      background: { alpha: 1, b: 255, g: 255, r: 255 },
      fit: 'cover'
    })
    .ensureAlpha()
    .png()
    .toBuffer();
  // Step 2: Composite scaled image onto white background, then apply circular mask
  const cOutput = await sharp({
    create: {
      background: { alpha: 1, b: 255, g: 255, r: 255 },
      channels: 4,
      height: cSize,
      width: cSize
    }
  })
    .composite([
      { input: cScaledBuf, left: cOffset, top: cOffset },
      { blend: 'dest-in', input: cTmpMask, left: 0, top: 0 }
    ])
    .png()
    .toBuffer();

  fs.writeFileSync(cTargetPath, cOutput);
  fs.unlinkSync(cTmpMask);

  console.log(
    `✓ ${pTargetFilename} → circle from ${cSourceIcon}, ${cPaddingPct}% padding (size=${cSize}, r=${radius}, scale=${cScalePct}%, circleDiameter=${cCircleDiameter})`
  );
}

async function fMain() {
  console.log(
    `Circular mask: ${cPaddingPct}% padding, ${cScalePct}% scale, source=${cSourceIcon}\n`
  );
  for (const cIcon of aTargetIcons) {
    await fMakeCircle(cIcon);
  }

  console.log('\nUnchanged:');
  for (const cIcon of aUntouchedIcons) {
    const cFp = path.join(cStaticDir, cIcon);

    console.log(
      `  ${cIcon} (${fs.existsSync(cFp) ? fs.statSync(cFp).size + ' bytes' : 'missing'})`
    );
  }
}

fMain();
