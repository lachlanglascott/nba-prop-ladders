# Injury-report status calibration (2025-26)

`uv run python -m nba_ladders.status_calibration --season-id 22025`

For each of the 164 game days, the report published at ~5 PM ET (before most tip-offs; falls
back to 4:30 / 5:30 / 4:00 / 6:00) vs whether the player appeared in that day's box score.
13,392 listings. Note the report file naming changed mid-season (`_05PM` → `_05_00PM`); the
fetcher handles both.

| Status | Listed | Sat | P(sits) | Minutes if played |
|---|---|---|---|---|
| Out | 9,830 | 9,825 | 99.9% | 8.6 |
| Doubtful | 345 | 342 | **99.1%** | 7.7 |
| Questionable | 1,492 | 791 | **53%** | 23.7 |
| Probable | 590 | 72 | 12% | 24.7 |
| Available | 1,135 | 274 | 24% | 22.5 |

By reason (Questionable): injury/illness 50% (1,126), rest/management 49% (148), other 72% (218).

**Takeaways**
- **Doubtful ≈ Out.** Treat Doubtful centres as full alerts (the sheet already does).
- **Questionable is a coin flip** (53%), so a Questionable starting centre is worth pricing both ways,
  and the 5 PM status is not the final word. Later reports (and lineups ~30 min before tip) resolve it.
- "Available" players who sat are mostly two-way/G-League designations and coach's decisions.

Used in `src/nba_ladders/sheet.py` (`P_OUT`). Caveat: one season; the ~5 PM snapshot is earlier
than the final pre-tip status for late games.
