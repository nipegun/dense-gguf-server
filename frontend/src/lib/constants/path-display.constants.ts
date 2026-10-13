/**
 * Constants for synthetic working-directory messages.
 *
 * The synthetic cwd-change message is text the UI renders as a folder row
 * and the model sees as a turn reminder. The prefix and cleared marker keep
 * the human-readable wording; the file-link regexes parse the
 * `[file:///abs/path](display)` payload back out on the UI side.
 */

import { UrlProtocol } from '$lib/enums';

export const cCwdChangedPrefix = 'Set working directory to ';
export const cCwdClearedText = 'Working directory cleared';

/** Trailing separator that marks a path as a directory. */
export const cDirectoryPathSuffix = '/';

export const cHomeTilde = '~';
export const cHomeTildePrefix = '~/'; // tilde plus path separator

/** Scheme prefix of the file link embedded in a synthetic cwd message. */
export const cFileUriPrefix = `${UrlProtocol.FILE}//`;

/** Matches the leading `[file:///abs/path](display)` link; not anchored to the end so trailing guidance may follow. */
export const cCwdLinkRegex = /^\[file:\/\/([\s\S]*?)\]\(([\s\S]*?)\)/;
