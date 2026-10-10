// Server times are UTC; an older server sends them with no offset. Parsed with
// a bare `new Date(naive)` they are read as BROWSER-LOCAL, so the dashboard
// showed "8h ago" for a pane active a minute ago when viewed in UTC+8.
//
// The zone is pinned HERE, not left to the runner: in UTC the two readings
// coincide, so a UTC CI runner would pass these whether or not the bug exists.
// Vitest runs each file in its own process, so this does not leak.
// `process` via globalThis: web/ has no @types/node, and tsc covers src/test.
;(globalThis as any).process.env.TZ = 'Asia/Taipei'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { parseServerTime } from '../time'
import { fmtRel } from '../components/DashboardHome'
import { formatRelativeTime } from '../components/InboxPanel'

const NOW = Date.UTC(2026, 9, 1, 13, 38, 12) // 2026-10-01T13:38:12Z
const MINUTE_AGO_NAIVE = '2026-10-01T13:37:12' // UTC, no offset: an older server
const MINUTE_AGO_AWARE = '2026-10-01T13:37:12+00:00' // a current server

describe('server timestamps are UTC whatever the browser zone', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('runs in +08:00 (else every case below proves nothing)', () => {
    expect(new Date(NOW).getTimezoneOffset()).toBe(-480)
  })

  it('reads a naive server time as UTC', () => {
    expect(parseServerTime(MINUTE_AGO_NAIVE).getTime()).toBe(NOW - 60_000)
  })

  it('keeps an explicit offset as given', () => {
    expect(parseServerTime('2026-10-01T13:37:12Z').getTime()).toBe(NOW - 60_000)
    expect(parseServerTime(MINUTE_AGO_AWARE).getTime()).toBe(NOW - 60_000)
    expect(parseServerTime('2026-10-01T21:37:12+08:00').getTime()).toBe(NOW - 60_000)
    expect(parseServerTime('2026-10-01T21:37:12+0800').getTime()).toBe(NOW - 60_000)
  })

  it('dashboard: a naive time a minute old reads "1m ago", not hours', () => {
    expect(fmtRel(MINUTE_AGO_NAIVE)).toBe('1m ago')
  })

  it('dashboard: an offset-bearing time a minute old reads "1m ago"', () => {
    expect(fmtRel(MINUTE_AGO_AWARE)).toBe('1m ago')
  })

  it('inbox: a naive time a minute old reads "1m ago", not hours', () => {
    expect(formatRelativeTime(MINUTE_AGO_NAIVE)).toBe('1m ago')
  })
})
