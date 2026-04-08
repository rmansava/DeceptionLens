"""Quick diagnostic: test DISK keypoint extraction on 10 board game images."""
import os, cv2, numpy as np, torch
import kornia.feature as KF
import kornia as K

# Find 10 images spread across folders
images = []
for root, dirs, files in os.walk(r"C:\boardgames"):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            images.append(os.path.join(root, f))
            if len(images) >= 10:
                break
    if len(images) >= 10:
        break

print(f"Found {len(images)} test images\n")

device = torch.device('cuda')
extractor = KF.DISK.from_pretrained('depth').to(device).eval()
print("DISK model loaded\n")

for img_path in images:
    raw = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        fsize = os.path.getsize(img_path)
        print(f"FAILED to load ({fsize} bytes): {img_path}")
        continue
    h, w = img.shape[:2]
    orig = f"{w}x{h}"
    if max(h, w) > 4096:
        scale = 4096 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    if new_h != h or new_w != w:
        img = cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = K.image_to_tensor(img, False).float() / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        feats = extractor(tensor)[0]
    kp = len(feats.descriptors)
    fsize = os.path.getsize(img_path) // 1024
    print(f"{orig:>12s}  {kp:>6,} keypoints  {fsize:>6,} KB  {os.path.basename(img_path)}")

del extractor
torch.cuda.empty_cache()
