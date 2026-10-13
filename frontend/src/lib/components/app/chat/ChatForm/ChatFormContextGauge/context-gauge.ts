import { ColorLevel } from '$lib/enums';

const cWarningThreshold = 80;
const cCriticalThreshold = 95;

export function fColorLevelFromPercent(pPercent: number | null): ColorLevel {
  if (pPercent === null) return ColorLevel.cNeutral;

  if (pPercent >= cCriticalThreshold) return ColorLevel.cCritical;

  if (pPercent >= cWarningThreshold) return ColorLevel.cWarning;

  return ColorLevel.OK;
}

export function fColorLevelTextClass(pLevel: ColorLevel): string {
  switch (pLevel) {
    case ColorLevel.cCritical:
      return 'text-red-400';
    case ColorLevel.cWarning:
      return 'text-amber-400';
    case ColorLevel.OK:
      return 'text-muted-foreground';
    default:
      return 'text-muted-foreground';
  }
}

export function fColorLevelBgClass(pLevel: ColorLevel): string {
  switch (pLevel) {
    case ColorLevel.cCritical:
      return 'bg-red-500';
    case ColorLevel.cWarning:
      return 'bg-amber-500';
    case ColorLevel.OK:
      return 'bg-green-500';
    default:
      return 'bg-muted';
  }
}
