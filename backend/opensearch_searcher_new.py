import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch
import numpy as np
import cv2

try:
    import kornia.feature as KF
    KORNIA_AVAILABLE = True
except ImportError:
    KF = None
    KORNIA_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-books"
FACES_INDEX = "faces-books"

class OpenSearchSearcher:
    def __init__(self, visual_index=VISUAL_INDEX, faces_index=FACES_INDEX):
        self.visual_index = visual_index
        self.faces_index = faces_index
        self.client = OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}], http_compress=True, timeout=30)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OpenSearch Searcher using device: {self.device}")
        print("Loading DINOv2 model for search...")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.model.eval()
        print("DINOv2 loaded.")
        self.extractor = None
        self.matcher = None
        if KORNIA_AVAILABLE:
            try:
                print("Loading DISK + LightGlue for geometric verification...")
                self.extractor = KF.DISK.from_pretrained("depth").to(self.device).eval()
                self.matcher = KF.LightGlue(features="disk").to(self.device).eval()
                print("DISK + LightGlue loaded.")
            except Exception as e:
                print(f"Failed to load DISK/LightGlue: {e}")
        self.face_app = None
        self.face_app_loaded = False