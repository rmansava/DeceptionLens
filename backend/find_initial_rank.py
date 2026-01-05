"""Find initial rank of page210 in OpenSearch results before LightGlue"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch

# Load DINOv2
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').cuda()
model.eval()

# Get query embedding
test_img = r'D:/trivpics/2023-5.jpg'
img = Image.open(test_img).convert('RGB')
inputs = processor(images=img, return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
emb = emb / np.linalg.norm(emb)

# Search OpenSearch for 5000 results
client = OpenSearch(hosts=[{'host': 'localhost', 'port': 9200}])
result = client.search(
    index='dinov2-books',
    body={
        'size': 5000,
        'query': {
            'knn': {
                'embedding': {
                    'vector': emb.tolist(),
                    'k': 5000
                }
            }
        },
        '_source': ['path']
    }
)

# Find page210
hits = result['hits']['hits']
print(f'Total results returned: {len(hits)}')

found = False
for idx, hit in enumerate(hits):
    path = hit['_source']['path']
    if 'encyclopedia' in path.lower() and 'monsters' in path.lower() and 'page210' in path.lower():
        print(f'\nFOUND: page210 at initial rank {idx+1} with score {hit["_score"]:.4f}')
        print(f'Path: {path}')
        found = True
        break

if not found:
    print('\npage210 NOT FOUND in top 5000 results!')
    print('\nSearching for any encyclopedia of monsters pages in results...')
    enc_pages = []
    for idx, hit in enumerate(hits):
        path = hit['_source']['path']
        if 'encyclopedia' in path.lower() and 'monsters' in path.lower():
            enc_pages.append((idx+1, path, hit['_score']))

    if enc_pages:
        print(f'\nFound {len(enc_pages)} encyclopedia of monsters pages:')
        for rank, path, score in enc_pages[:20]:  # Show top 20
            print(f'  Rank {rank}: {path.split(chr(92))[-1]} (score: {score:.4f})')
    else:
        print('No encyclopedia of monsters pages found in top 5000!')
