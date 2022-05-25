from distutils.command.upload import upload
import numpy as np 
from PIL import Image 
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path 
import os 

app = Flask(__name__)

# read image features 
fe = FeatureExtractor() 
features = [] 
img_paths = []
for feature_path in Path(os.path.join('static', 'feature')).glob('*.npy'):
    features.append(np.load(feature_path)) 
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features) 


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        #save query image
        img = Image.open(file.stream).convert('RGB')
        uploaded_img_path = os.path.join("static", "uploaded", datetime.now().isoformat().replace(":",".")+"_"+file.filename)
        img.save(uploaded_img_path)

        #Run search
        query = fe.extract(img)
        query = query.numpy()
        query = np.expand_dims(query, axis=0)

        dists = np.linalg.norm(features-query, axis=2)
        dists = np.squeeze(dists, axis=1)

        ids = np.argsort(dists, axis=0)[:3]
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                                query_path=uploaded_img_path,
                                scores=scores)
    
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run("0.0.0.0")