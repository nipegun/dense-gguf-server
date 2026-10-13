import { fTranslate } from '$lib/i18n';
import { cFileTypeIcons } from '$lib/constants';
import { AttachmentAction, AttachmentItemEnabledWhen, AttachmentMenuItemId } from '$lib/enums';
import type { AttachmentMenuItem } from '$lib/types';

/**
 * File attachment menu items shown in both the desktop dropdown and mobile sheet.
 * The "Tools" submenu is handled separately by each component.
 */
export const aAttachmentFileItems: AttachmentMenuItem[] = [
  {
    action: AttachmentAction.FILE_UPLOAD,
    class: 'images-button',
    disabledTooltip: 'Image processing requires a vision model',
    enabledWhen: AttachmentItemEnabledWhen.HAS_VISION_MODALITY,
    icon: cFileTypeIcons.image,
    id: AttachmentMenuItemId.IMAGES,
    label: fTranslate('text_be7e2f201293')
  },
  {
    action: AttachmentAction.FILE_UPLOAD,
    class: 'audio-button',
    disabledTooltip: 'Audio files processing requires an audio model',
    enabledWhen: AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY,
    icon: cFileTypeIcons.audio,
    id: AttachmentMenuItemId.AUDIO,
    label: fTranslate('text_85b791f3241c')
  },
  {
    action: AttachmentAction.FILE_UPLOAD,
    class: 'video-button',
    disabledTooltip: 'Video files processing requires a video model',
    enabledWhen: AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY,
    icon: cFileTypeIcons.video,
    id: AttachmentMenuItemId.VIDEO,
    label: fTranslate('text_f1719b8182ad')
  },
  {
    action: AttachmentAction.FILE_UPLOAD,
    enabledWhen: AttachmentItemEnabledWhen.cAlways,
    icon: cFileTypeIcons.text,
    id: AttachmentMenuItemId.TEXT,
    label: fTranslate('text_58c701e56182')
  },
  {
    action: AttachmentAction.FILE_UPLOAD,
    disabledTooltip: 'PDFs will be converted to text. Image-based PDFs may not work properly.',
    enabledWhen: AttachmentItemEnabledWhen.cAlways,
    hasEnabledTooltip: true,
    icon: cFileTypeIcons.pdf,
    id: AttachmentMenuItemId.PDF,
    label: fTranslate('text_fad030751fb1')
  }
];

export const cAttachmentTooltipText = 'Add files, prompts, tools or MCP Servers';

/** Text shared by the add menu's two surfaces: the desktop dropdown and the mobile drawer. */
export const cAttachmentMenuText = {
  ADD_FILES: 'Add files',
  ADD_TO_CHAT: 'Add to chat',
  DESCRIPTION: 'Add files, system prompt or pick the tools the model may call',
  SYSTEM_MESSAGE: 'System Message',
  TOOLS: 'Tools'
} as const;

/** Modality labels behind the file-upload row's enabled-when flags, both surfaces. */
export const dAttachmentModalityText: Record<AttachmentItemEnabledWhen, string> = {
  [AttachmentItemEnabledWhen.cAlways]: '',
  [AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY]: 'Audio',
  [AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY]: 'Video',
  [AttachmentItemEnabledWhen.HAS_VISION_MODALITY]: 'Vision'
};
