# DISK Retrieval Test Plan

Known query -> source-page pairs used to validate DISK ranking quality and
detect "hub page" false positives.

## Known Pairs

| Test Image | Expected Book | Expected Page |
|---|---|---|
| `D:\trivpics\2023-5.jpg` | Encyclopedia Of Monsters, The | page 206 |
| `D:\trivpics\2024-8.png` | Ad boy Vintage advertising with character - Warren Dotz | page 98 |
| `D:\stcloudtrivia\2023-19.jpg` | Television cartoon shows an illustrated encyclopedia | page 365 |

## Test 1: Fast Canonical Check (single chunk, < 1 minute)

Purpose: Confirm baseline DISK correctness without a full corpus run.

```bash
curl -X POST "http://localhost:8000/disk/search?chunk_ids=183&collections=books&top_k=20&k=5&threshold=0.7" -F "file=@D:/trivpics/2023-5.jpg"
```

Expected:
- #1 path ends with `Encyclopedia Of Monsters...-page206.jpg`
- Top votes approximately `130-170` on current index
- Strong separation from #2 (typically > 100 vote margin)

Fail conditions:
- Correct page is not rank #1
- Top vote margin vs #2 is small (< 30) for this canonical test

## Test 2: Full-Corpus Hub-Page Check (batch)

Purpose: Verify that dense page-0 "hub" pages do not dominate unrelated queries.

Run `backend\run_batch_disk_search.bat` on a folder containing at least:
- `D:\trivpics\2023-5.jpg`
- `D:\trivpics\2024-8.png`

For `2024-8.png`, expected outcome:
- Target `Ad boy...-page98.jpg` should be top-ranked or very close to #1
- If target is #2, top-vs-#2 margin should be small (near tie), not a blowout

Fail conditions:
- `Ad boy...-page98.jpg` drops out of top 10
- Same unrelated page-0 repeatedly wins by large margins across many queries

## Baseline Notes

- Prior run (search #2228, batch DISK, 1014 chunks):
  - `2024-8.png` ranked **#2** with 297 votes
  - #1 had 305 votes (8-vote margin)
  - "Encyclopedia of Television Shows page 0" appeared as #1 in many searches
    (hub-page false-positive pattern)

## Reporting Template

For each test query, log:
- Query image
- Search ID
- Rank of expected page
- Expected page votes
- #1 votes
- #2 votes
- Margin (`top1 - top2`)
- Notes (e.g., hub page observed, near tie, clear winner)
