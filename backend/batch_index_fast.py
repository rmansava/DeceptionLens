"""
Fast batch indexer - loads model ONCE and processes all books efficiently.
Uses larger GPU batches for better throughput.
"""
import os
import sys
import glob
import gc
import time
import warnings

# Suppress all the annoying warnings BEFORE imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch, helpers
import numpy as np
import cv2

# Try importing InsightFace (optional)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False

# Configuration
BOOKS_ROOT = r"D:\books\pdf-images"
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-books"
FACES_INDEX = "faces-books"
PROGRESS_FILE = "batch_progress_fast.txt"
OLD_PROGRESS_FILE = "batch_progress_opensearch.txt"
LOG_FILE = "batch_index_fast.log"

# Batch sizes
GPU_BATCH_SIZE = 16
OPENSEARCH_BATCH_SIZE = 100

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def log(msg, color=None, to_file=True):
    """Print to console and optionally log file."""
    ts = datetime.now().strftime("%H:%M:%S")
    if color:
        print(f"{DIM}[{ts}]{RESET} {color}{msg}{RESET}", flush=True)
    else:
        print(f"{DIM}[{ts}]{RESET} {msg}", flush=True)
    if to_file:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] {msg}\n")


def print_header():
    """Print a nice header."""
    print(f"\n{CYAN}{BOLD}")
    print("  ____  _  _  _   _  ___       ___           _                    ")
    print(" |  _ \\(_)| \\| | / _ \\ __ __ |_ _| _ _   __| | ___ __ __ ___  _ _ ")
    print(" | |_) || || .` || (_) |\\ V /  | | | ' \\ / _` |/ -_)\\ \\ // -_)| '_|")
    print(" |____/ |_||_|\\_| \\___/  \\_/  |___||_||_|\\__,_|\\___|/_\\_\\\\___||_|  ")
    print(f"{RESET}")
    print(f"  {DIM}Fast GPU Batch Indexer - DINOv2 + InsightFace{RESET}\n")


def print_status_box(total, completed, remaining):
    """Print a status box."""
    print(f"  {CYAN}+-----------------------+{RESET}")
    print(f"  {CYAN}|{RESET}  Total books: {BOLD}{total:>6}{RESET}  {CYAN}|{RESET}")
    print(f"  {CYAN}|{RESET}  Completed:   {GREEN}{completed:>6}{RESET}  {CYAN}|{RESET}")
    print(f"  {CYAN}|{RESET}  Remaining:   {YELLOW}{remaining:>6}{RESET}  {CYAN}|{RESET}")
    print(f"  {CYAN}+-----------------------+{RESET}\n")


class FastBatchIndexer:
    """Efficient batch indexer that keeps models loaded."""

    def __init__(self, enable_visual=True, enable_faces=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # OpenSearch client
        self.client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            timeout=60
        )

        # Load DINOv2 once
        self.processor = None
        self.model = None
        self.enable_visual = enable_visual
        if enable_visual:
            print(f"  {DIM}Loading DINOv2...{RESET}", end=" ", flush=True)
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
            self.model.eval()
            print(f"{GREEN}OK{RESET}")

        # Load InsightFace once
        self.face_app = None
        self.enable_faces = enable_faces
        if enable_faces and INSIGHTFACE_AVAILABLE:
            print(f"  {DIM}Loading InsightFace...{RESET}", end=" ", flush=True)
            # Suppress InsightFace output
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
                    self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                finally:
                    sys.stdout = old_stdout
            print(f"{GREEN}OK{RESET}")

        device_color = GREEN if self.device == "cuda" else YELLOW
        print(f"  {DIM}Device:{RESET} {device_color}{self.device.upper()}{RESET}\n")

    def get_visual_embeddings_batch(self, image_paths: list) -> list:
        """Get DINOv2 embeddings for a batch of images."""
        if not self.model:
            return [None] * len(image_paths)

        embeddings = []
        valid_images = []
        valid_indices = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
                valid_indices.append(i)
            except:
                pass

        if not valid_images:
            return [None] * len(image_paths)

        try:
            inputs = self.processor(images=valid_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            for emb in batch_embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)
        except:
            return [None] * len(image_paths)

        result = [None] * len(image_paths)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb
        return result

    def get_face_embeddings(self, image_path: str) -> list:
        """Extract face embeddings from an image."""
        if self.face_app is None:
            return []
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return []
            faces = self.face_app.get(img)
            return [face.embedding for face in faces]
        except:
            return []

    def index_book(self, book_path: str, book_name: str = None):
        """Index a single book efficiently using GPU batching."""
        if book_name is None:
            book_name = os.path.basename(book_path)

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif', '*.bmp']
        files_set = set()
        for ext in image_extensions:
            for f in glob.glob(os.path.join(book_path, '**', ext), recursive=True):
                files_set.add(os.path.normpath(f))
            for f in glob.glob(os.path.join(book_path, '**', ext.upper()), recursive=True):
                files_set.add(os.path.normpath(f))

        files = sorted(list(files_set))
        if not files:
            return {"visual": 0, "faces": 0}

        visual_actions = []
        face_actions = []
        visual_count = 0
        face_count = 0

        for batch_start in range(0, len(files), GPU_BATCH_SIZE):
            batch_files = files[batch_start:batch_start + GPU_BATCH_SIZE]

            if self.enable_visual:
                embeddings = self.get_visual_embeddings_batch(batch_files)
                for file_path, embedding in zip(batch_files, embeddings):
                    if embedding is not None:
                        visual_actions.append({
                            "_index": VISUAL_INDEX,
                            "_id": file_path,
                            "_source": {
                                "embedding": embedding.tolist(),
                                "path": file_path,
                                "filename": os.path.basename(file_path),
                                "book": book_name
                            }
                        })
                        visual_count += 1

            if self.enable_faces and self.face_app:
                for file_path in batch_files:
                    face_embeddings = self.get_face_embeddings(file_path)
                    for i, face_emb in enumerate(face_embeddings):
                        face_id = f"{file_path}_face_{i}"
                        face_actions.append({
                            "_index": FACES_INDEX,
                            "_id": face_id,
                            "_source": {
                                "embedding": face_emb.tolist(),
                                "path": file_path,
                                "source_image": file_path,
                                "face_index": i,
                                "book": book_name
                            }
                        })
                        face_count += 1

            if len(visual_actions) >= OPENSEARCH_BATCH_SIZE:
                try:
                    helpers.bulk(self.client, visual_actions, refresh=False)
                except:
                    pass
                visual_actions = []

            if len(face_actions) >= OPENSEARCH_BATCH_SIZE:
                try:
                    helpers.bulk(self.client, face_actions, refresh=False)
                except:
                    pass
                face_actions = []

        if visual_actions:
            try:
                helpers.bulk(self.client, visual_actions, refresh=False)
            except:
                pass

        if face_actions:
            try:
                helpers.bulk(self.client, face_actions, refresh=False)
            except:
                pass

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {"visual": visual_count, "faces": face_count}


def get_all_books():
    """Get all book directories."""
    books = []
    for entry in os.listdir(BOOKS_ROOT):
        full_path = os.path.join(BOOKS_ROOT, entry)
        if os.path.isdir(full_path):
            books.append(entry)
    return sorted(books)


def count_images(book_path):
    """Count images in a book directory."""
    files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']:
        for f in Path(book_path).glob(ext):
            files.add(str(f).lower())
        for f in Path(book_path).glob(ext.upper()):
            files.add(str(f).lower())
    return len(files)


def load_completed():
    """Load set of completed books from both progress files."""
    completed = set()
    for pf in [OLD_PROGRESS_FILE, PROGRESS_FILE]:
        if os.path.exists(pf):
            with open(pf, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        completed.add(line.strip())
    return completed


def mark_completed(book_name):
    """Mark a book as completed."""
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(book_name + '\n')


def format_time(seconds):
    """Format seconds as human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    print_header()

    # Get books to process
    all_books = get_all_books()
    completed = load_completed()
    to_index = [b for b in all_books if b not in completed]

    print_status_box(len(all_books), len(completed), len(to_index))

    if not to_index:
        print(f"  {GREEN}Nothing to index - all done!{RESET}\n")
        return

    # Load models ONCE
    indexer = FastBatchIndexer(enable_visual=True, enable_faces=True)

    try:
        indexer.client.indices.refresh(index=VISUAL_INDEX)
        indexer.client.indices.refresh(index=FACES_INDEX)
    except:
        pass

    start_time = time.time()

    for i, book in enumerate(to_index):
        book_path = os.path.join(BOOKS_ROOT, book)
        img_count = count_images(book_path)

        # Truncate book name for display
        display_name = book[:45] + "..." if len(book) > 45 else book

        if img_count == 0:
            print(f"  {YELLOW}[SKIP]{RESET} {DIM}{display_name}{RESET}")
            mark_completed(book)
            continue

        # Progress indicator
        pct = ((i + 1) / len(to_index)) * 100
        print(f"  {CYAN}[{i+1}/{len(to_index)}]{RESET} {display_name} {DIM}({img_count} imgs){RESET}", end="", flush=True)

        book_start = time.time()
        result = indexer.index_book(book_path, book)
        book_time = time.time() - book_start

        imgs_per_sec = img_count / book_time if book_time > 0 else 0
        mark_completed(book)

        # ETA calculation
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = len(to_index) - (i + 1)
        eta = avg_time * remaining

        # Results on same line
        face_str = f" +{result['faces']}f" if result['faces'] > 0 else ""
        print(f" {GREEN}OK{RESET} {DIM}{imgs_per_sec:.0f}/s{face_str} | ETA {format_time(eta)}{RESET}")

    # Final refresh
    try:
        indexer.client.indices.refresh(index=VISUAL_INDEX)
        indexer.client.indices.refresh(index=FACES_INDEX)
    except:
        pass

    total_time = time.time() - start_time
    print(f"\n  {GREEN}{BOLD}COMPLETE!{RESET} {DIM}Total time: {format_time(total_time)}{RESET}\n")


if __name__ == "__main__":
    main()
