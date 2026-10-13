import { fTranslate } from '$lib/i18n';
import { cSetWorkingDirectoryLabel } from '$lib/constants';
import { ChatFormCommandAction } from '$lib/enums';
import type { ChatCommandsOptions, ChatFormCommand } from '$lib/types';

/**
 * The slash commands surfaced by the `/` command picker, in display order.
 *
 * Availability is supplied as predicates rather than store imports: this
 * module is re-exported through the `$lib/utils` barrel, and importing
 * stores at module load would create a circular dependency (the stores
 * themselves import from `$lib/utils`).
 */
export function fGetChatCommands(pOptions: ChatCommandsOptions): ChatFormCommand[] {
  return [
    {
      action: ChatFormCommandAction.PROMPT,
      description: fTranslate('dgs_73cd3b7c83d2'),
      disabled: !pOptions.hasPrompts(),
      name: 'prompt'
    },
    {
      action: ChatFormCommandAction.CWD,
      description: cSetWorkingDirectoryLabel,
      disabled: !pOptions.hasCwdTools(),
      keywords: ['current working directory'],
      name: 'cwd'
    },
    {
      action: ChatFormCommandAction.MODEL,
      description: fTranslate('text_1d446005add3'),
      disabled: !pOptions.showModelSelector,
      name: 'model'
    }
  ];
}
