// Server timestamps (terminal last_active, inbox created_at) are UTC. Older
// servers sent them NAIVE - no offset - and `new Date(naive)` parses a
// date-time without an offset as BROWSER-LOCAL time, so every relative age was
// off by the viewer's UTC offset ("8h ago" for a pane active a minute ago in
// UTC+8). Current servers send an explicit offset; this keeps the UI right
// against an older server too. A value that already carries `Z` or `±hh:mm` is
// parsed as given.
const HAS_ZONE = /(?:[zZ]|[+-]\d\d:?\d\d)$/

export function parseServerTime(s: string): Date {
  return new Date(HAS_ZONE.test(s) ? s : s + 'Z')
}
