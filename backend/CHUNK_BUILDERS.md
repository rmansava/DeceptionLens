# DISK Chunk Builder Batches

This file documents how the `run_build_*_chunks.bat` scripts work and what each category uses.

## Scripts to run

- `backend/run_build_books_chunks.bat`
- `backend/run_build_boardgames_chunks.bat`
- `backend/run_build_printads_chunks.bat`
- `backend/run_build_albums_chunks.bat`
- `backend/run_build_comics_chunks.bat`
- `backend/run_build_cereal_chunks.bat`

All of these are resumable and safe to restart.

## Common behavior

Every run script does the same control flow:

1. Sets `DISK_MAX_VECTORS_PER_CHUNK=19500000` (~10 GB FAISS payload target).
2. Runs the matching Python builder (`build_*_disk_chunks.py`).
3. If the builder exits with CUDA poison code `86`, waits briefly and starts it again.
4. Stops auto-restarting after `MAX_RESTARTS=20`.

CUDA poison restart wiring:

- Batch files set `DISK_CUDA_POISON_EXIT_CODE=86`.
- Python builders raise `CudaPoisonedError` and exit with that code.
- The batch loop catches code `86` and re-launches Python.

## Category modes and paths

### Books

- Script: `backend/run_build_books_chunks.bat`
- Builder: `backend/build_books_disk_chunks.py`
- Source mode: local only
- Source images: `D:\books\pdf-images`
- Chunks: `T:\faiss\disk_retrieval\chunks`
- IDs/progress: `D:\faiss\disk_retrieval\chunk_ids`

### Board Games

- Script: `backend/run_build_boardgames_chunks.bat`
- Builder: `backend/build_boardgames_disk_chunks.py`
- Mode A: local direct if `C:\boardgames` exists
- Mode B: NAS buffered if local folder is missing
  - NAS source: `T:\archiverelated\board games`
  - Local temp buffer: `C:\boardgames-temp`
- Chunks: `T:\faiss\disk_retrieval\boardgames_chunks`
- IDs/progress: `D:\faiss\disk_retrieval\boardgames_chunk_ids`

### Print Ads

- Script: `backend/run_build_printads_chunks.bat`
- Builder: `backend/build_printads_disk_chunks.py`
- Mode A: local direct if `C:\printads` or `C:\print ads` exists
- Mode B: NAS buffered if local folder is missing
  - NAS source: `T:\archiverelated\print ads`
  - Local buffer: `C:\printads_buffer`
- Chunks: `S:\faiss\disk_retrieval\printads_chunks`
- IDs/progress: `D:\faiss\disk_retrieval\printads_chunk_ids`

### Albums

- Script: `backend/run_build_albums_chunks.bat`
- Builder: `backend/build_albums_disk_chunks.py`
- Mode A: local direct if `C:\albums` exists
- Mode B: NAS buffered if local folder is missing
  - NAS source: `T:\archiverelated\albums`
  - Local buffer: `C:\albums_buffer`
- Chunks: `U:\faiss\disk_retrieval\albums_chunks`
- IDs/progress: `D:\faiss\disk_retrieval\albums_chunk_ids`

### Comics

- Script: `backend/run_build_comics_chunks.bat`
- Builder: `backend/build_comics_disk_chunks.py`
- Mode: NAS buffered
  - NAS source: `T:\archiverelated\comics`
  - Local buffer: `C:\comics_buffer`
- Chunks: `T:\faiss\disk_retrieval\comics_chunks`
- IDs/progress: `D:\faiss\disk_retrieval\comics_chunk_ids`

### Cereal

- Script: `backend/run_build_cereal_chunks.bat`
- Builder: `backend/build_cereal_disk_chunks.py`
- Mode A: local direct if `C:\cereal` exists
- Mode B: NAS buffered if local folder is missing
  - NAS source: `T:\archiverelated\cereal`
  - Local buffer: `C:\cereal_buffer`
- Chunks: `T:\faiss\disk_retrieval\cereal_chunks`
- IDs/progress: `D:\faiss\disk_retrieval\cereal_chunk_ids`

## Resume files

Each category keeps progress and mapping files in its IDs directory:

- `build_progress.json`
- `path_lookup.json`
- `build_log.txt`
- optional: `cuda_bad_images.txt`

## Useful env vars

The builders support these overrides:

- `DISK_MAX_VECTORS_PER_CHUNK` (chunk size target)
- `DISK_MAX_IMAGE_DIM` (resize cap; `none`/`0` disables resize)
- `DISK_CUDA_POISON_EXIT_CODE` (restart code, default `86`)
- `DISK_CUDA_ERROR_STREAK_FOR_RECOVERY`
- `DISK_CUDA_MAX_RECOVERY_ATTEMPTS`
- `DISK_PROGRESS_SAVE_EVERY_IMAGES`
- `DISK_PATH_DB_SYNC_EVERY_CHUNKS` (where applicable)

You can set these in the run bat before the `python` call.

## Current resize mode across categories

All `run_build_*_chunks.bat` scripts are currently configured to:

- set `DISK_MAX_IMAGE_DIM=2048`
- keep aspect ratio during resize (`scale = 2048 / max(h,w)` and proportional width/height)
- index resized images normally (not skipped)

### Revisit plan for higher-detail pass later

When ready for a higher-detail pass on any category:

1. Set `DISK_MAX_IMAGE_DIM=4096` (or `none`) in that category's run bat.
2. Re-run that category's build script.
3. For strict consistency across the full collection, do a full rebuild at the chosen cap.

### Quality note

This is a throughput tradeoff. It is still much safer than the old 1600px bug,
but it may reduce snippet fidelity on some fine-detail pages versus 4096/full-res.
