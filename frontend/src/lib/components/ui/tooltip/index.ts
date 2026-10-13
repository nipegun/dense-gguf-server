import Content from './tooltip-content.svelte';
import Trigger from './tooltip-trigger.svelte';
import { Tooltip as TooltipPrimitive } from 'bits-ui';

const fRoot = TooltipPrimitive.Root;
const fProvider = TooltipPrimitive.Provider;
const fPortal = TooltipPrimitive.Portal;

export {
  fRoot,
  Trigger,
  Content,
  fProvider,
  fPortal,
  //
  fRoot as Tooltip,
  Content as TooltipContent,
  Trigger as TooltipTrigger,
  fProvider as TooltipProvider,
  fPortal as TooltipPortal
};
