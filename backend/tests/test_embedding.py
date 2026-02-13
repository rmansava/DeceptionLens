"""Test embedding similarity between test image and page210"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch

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

# Get embeddings
test_img = r'D:/trivpics/2023-5.jpg'
page206 = r'D:\books\pdf-images\encyclopedia of monsters\encyclopedia of monsters-page206.jpg'

print('Computing embeddings...')
emb_test = get_embedding(test_img)
emb_page = get_embedding(page206)

# Cosine similarity (already normalized)
similarity = np.dot(emb_test, emb_page)
print(f'Direct cosine similarity between test image and page206: {similarity:.6f}')

# Now get the stored embedding from OpenSearch
client = OpenSearch(hosts=[{'host': 'localhost', 'port': 9200}])
result = client.search(
    index='dinov2-books',
    body={
        'query': {'term': {'path': page206}},
        '_source': ['embedding', 'path']
    }
)

if result['hits']['total']['value'] > 0:
    stored_emb = np.array(result['hits']['hits'][0]['_source']['embedding'])
    sim_stored = np.dot(emb_test, stored_emb)
    print(f'Similarity with STORED page210 embedding: {sim_stored:.6f}')

    # Check if stored embedding is same as fresh
    sim_fresh_stored = np.dot(emb_page, stored_emb)
    print(f'Fresh vs stored page210 embedding similarity: {sim_fresh_stored:.6f}')
else:
    print('Page210 not found in OpenSearch index!')
