// grace window after a visibilitychange before we kick a reader whose socket likely died
// while the tab was hidden. covers brief background pauses without thrashing live streams
export const cStreamVisibilityKickMs = 3000;

// separator joining a conversation id and its per-model stream identity
// suffix (conv::model) used by the server side replay buffer
export const cConversationIdSeparator = '::';

/**
 * Server-sent events wire format, shared by the chat stream and the
 * /models/sse status feed (text/event-stream).
 */

// blank line between two events
export const cSseRecordSeparator = '\n\n';

// line break inside an event
export const cSseLineSeparator = '\n';

// data field prefix, the value follows after an optional space
export const cSseDataPrefix = 'data:';

// end-of-stream marker on the chat completion stream
export const cSseDoneMarker = '[DONE]';
