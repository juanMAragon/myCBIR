import cv2 
import os
import numpy as np
from pathlib import Path 
from sklearn import cluster, neighbors

SEED = 123
file_list = [x for x in Path(os.path.join('static', 'img')).glob('*.jpg')]

sift = cv2.xfeatures2d.SIFT_create()
l_descriptors = [] #list containing the descriptors of all the images in the dataset


for img_path in file_list:
    print(img_path)
    img_gray = cv2.imread(str(img_path))

    # keypoints: list containing n feature points cordinate (in this case 2158)
    # descriptors: for each keypoint it stores 128 values.  
    img_keypoints, img_descriptors = sift.detectAndCompute(img_gray, None)
    l_descriptors.append(img_descriptors)


# getting cluster center
cluster_pt_ratio = 1.0
dataset_descriptors = np.concatenate(l_descriptors, axis=0)

cluster_pt_num = int(cluster_pt_ratio*dataset_descriptors.shape[0])

np.random.shuffle(dataset_descriptors)

kmeans = cluster.KMeans(n_clusters=3, random_state=SEED)

kmeans.fit(dataset_descriptors[:cluster_pt_num])

print('Building BoW...')
num_imgs = len(file_list)
img_fea_mat = []
for i in range(num_imgs):
    
    num_fea_pt = l_descriptors[i].shape[0]
    hist = np.zeros(4096)
    word_idx = kmeans.predict(l_descriptors[i])
    for pt_idx in range(num_fea_pt):
        hist[word_idx[pt_idx]] += 1
    
    img_fea_mat.append(hist)

img_fea_mat = np.stack(img_fea_mat, axis=0)

print('Building KNN...')

knn = neighbors.NearestNeighbors(n_neighbors=3)
knn.fit(img_fea_mat)
print('Finished!')


# --------------------------------------------------

query_img = os.path.join("static","img","a3.jpg")
img_gray = cv2.imread(str(query_img))

keypoints, descriptors = sift.detectAndCompute(img_gray, None)


num_fea_pt = l_descriptors[i].shape[0]
hist = np.zeros(4096)
word_idx = kmeans.predict(l_descriptors[i])
for pt_idx in range(num_fea_pt):
    hist[word_idx[pt_idx]] += 1

img_fea = np.expand_dims(hist, axis=0)
dists, inds = knn.kneighbors(img_fea, 3, return_distance=True)


similar_imgs = []

print('query: ', query_img)

for cnt, im_idx in enumerate(inds[0]):
    similar_im_path = os.path.join("static", "img", file_list[im_idx])
    similar_imgs.append((similar_im_path, dists[0][cnt]))

print('similar images:')
print(similar_imgs)