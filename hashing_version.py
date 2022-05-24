from PIL import Image 
import imagehash
import os
import numpy as np
from pathlib import Path

distances = []
images = []

request_image_path = os.path.join("static","img","b1.jpg")
request_image = Image.open(request_image_path).convert('RGB')
request_hash = imagehash.phash(request_image)


for img_path in sorted(Path("./static/img").glob("*.jpg")):
    
    img = Image.open(img_path).convert('RGB')
    hash = imagehash.phash(img)

    distances.append(np.abs(request_hash-hash))
    images.append(img_path)

ids = np.argsort(distances, axis=0)


scores = [(distances[id], images[id]) for id in ids]
print(scores[:3])

