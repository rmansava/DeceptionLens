"""Find which encyclopedia page matches the test image"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch
import os

# Load DINOv2
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').cuda()
model.eval()

def get_embedding(path):
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb

test_img = r'D:/trivpics/2023-5.jpg'
emb_test = get_embedding(test_img)

# Search all encyclopedia of monsters pages
enc_dir = r'D:\books\pdf-images\encyclopedia of monsters'
best_sim = 0
best_page = None

print(f'Searching all pages in encyclopedia of monsters for best match...')
for fname in sorted(os.listdir(enc_dir)):
    if fname.endswith('.jpg'):
        path = os.path.join(enc_dir, fname)
        emb = get_embedding(path)
        sim = np.dot(emb_test, emb)
        if sim > best_sim:
            best_sim = sim
            best_page = fname
        if sim > 0.9:
            print(f'HIGH MATCH: {fname} -> {sim:.4f}')

print(f'\nBest match: {best_page} with similarity {best_sim:.4f}')
