import numpy as np 
from PIL import Image 
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path 
import os 

# read image features 
fe = FeatureExtractor() 
features = [] 
img_paths = []

for feature_path in Path(os.path.join('static', 'feature')).glob('*.npy'):
    features.append(np.load(feature_path)) 
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features) 

request_path = os.path.join('static','img', 'a3.jpg')
img=Image.open(request_path).convert('RGB')

query = fe.extract(img)
query = query.numpy()
query = np.expand_dims(query, axis=0)

print('-------------')
print(type(features))
print(type(query))
print('-------------')
print(features.shape)
print(query.shape)

dists = np.linalg.norm(features-query, axis=2) #L2 distances to features 
dists = np.squeeze(dists, axis=1)

ids = np.argsort(dists, axis=0)

print('-------------')
print(dists)

print(ids)


scores = [(dists[id], img_paths[id]) for id in ids]

print(scores[:3])