#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:12:24 2021

@author: jimmytabet
"""

#%% DIM RED/CLUSTERING
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import scale
big_cells=np.load('/Users/jimmytabet/NEL/Smart Micro/datasets/cell_dataset_isolate_9224.npy')

#%%
newdims = (big_cells.shape[0],50,50)
# newdims = (3000,50,50)

big_cells= big_cells[:newdims[0]]

cells = np.zeros(newdims)
for i in range(len(big_cells)):
     cells[i] = cv2.resize(big_cells[i],newdims[1:])
    
print(cells.shape) 

#%% use border and fft
# cells = np.load( '/home/nel-lab/Desktop/ShannonEntropy_2Dimgs/cell_dataset_isolate_9224.npy')
# print(cells.shape)
# #%%
# #import matplotlib.pyplot as plt
# #for i in range(10):
# #    plt.imshow((cells[i]>0)*cells[i], cmap='gray')
# #    plt.pause(0.2)
# #    
# #%%
# borders = 10
# new_cells = cells[:3000,borders:-borders,borders:-borders]

# #new_cells = new_cells[(new_cells>0).sum(axis=(1,2)) == np.prod(new_cells.shape[1:])]
# print(new_cells.shape)
# print(borders)

# big_cells = new_cells
# #%%
# newdims = (big_cells.shape[0],50,50)
# #newdims = (3000,50,50)

# big_cells= big_cells[:newdims[0]]

# cells = np.zeros(newdims)
# for i in range(len(big_cells)):
#      img = big_cells[i]
#      f = np.fft.fft2(img)
#      fshift = np.fft.fftshift(f)
#      magnitude_spectrum = 20*np.log(np.abs(fshift))
#      cells[i] = cv2.resize(magnitude_spectrum,newdims[1:])
    
# print(cells.shape) 

#%% dimensionality reduction
from sklearn.decomposition import PCA, NMF, KernelPCA, FastICA
from sklearn.manifold import TSNE

comp = 3
p = PCA(n_components=comp)
n = NMF(n_components=comp, max_iter=3000)
t = TSNE(n_components=comp, random_state=0)
k = KernelPCA(n_components=comp)
f = FastICA(n_components=comp, max_iter=3000)
p_20 = PCA(n_components=20)
i_p20 = FastICA(n_components=comp)

data = cells.reshape(cells.shape[0],-1)
data_scaled = scale(data, copy=True)

#%%
pca_data = p.fit_transform(data_scaled)
print('Done PCA')
nmf_data = n.fit_transform(data)
print('Done NMF')
tsne_data = t.fit_transform(data_scaled)
print('Done TSNE')
kpca_data = k.fit_transform(data_scaled)
print('Done KPCA')
fica_data = f.fit_transform(data_scaled)
print('Done FICA')
temp_data = p_20.fit_transform(data_scaled)
pca_fica_data = i_p20.fit_transform(temp_data)
print('Done PCA_FICA')

#%% compare dim reduction methods
dim_red = [pca_data,nmf_data,tsne_data,kpca_data,fica_data,pca_fica_data]
name = ['pca','nmf','tsne','kpca','fica','pca_fica']


rows = 2

# fig = plt.figure()
# for i in range(len(dim_red)): 
   
#     if comp == 3:
#         ax = fig.add_subplot(rows,np.ceil(len(dim_red)/rows).astype(int),i+1, projection='3d')
#     else:
#         ax = fig.add_subplot(rows,np.ceil(len(dim_red)/rows).astype(int),i+1)
        
#     ax.scatter(*dim_red[i].T, alpha=0.1)
#     ax.set_title(name[i])

#%% compare KMeans for dim reduction methods
from sklearn.cluster import KMeans

# 4 phases of mitosis
no_clus = 12
no_clus_tech = 1
cluster = KMeans(n_clusters = no_clus)

kmeans = []
kmeans_labels = []
kmeans_centroids = []
for i in dim_red:
    temp = cluster.fit(i)
    kmeans.append(temp)
    kmeans_labels.append(temp.labels_)
    kmeans_centroids.append(temp.cluster_centers_)
    
fig_all = plt.figure()
for i in range(len(kmeans)): 
    if comp == 2:
        ax = fig_all.add_subplot(rows,np.ceil(len(dim_red)/rows).astype(int),i+1)
    else:
        ax = fig_all.add_subplot(rows,np.ceil(len(dim_red)/rows).astype(int),i+1, projection='3d')
    
    
    for j in np.unique(kmeans_labels[i]):
        ax.scatter(*dim_red[i][kmeans_labels[i]==j].T)

    
    ax.scatter(*kmeans_centroids[i].T, c='k', marker='x')
    ax.set_title(name[i]+' kmeans clustering')
    
#%% closest points to centroids function
def closest_ids(dim_red_list, cluster_centroids):
    closest = []
    for dim, centroid in zip(dim_red_list, cluster_centroids):
        temp_dist = []
        for row in dim:
            diff = row-centroid
            temp_dist.append(np.linalg.norm(diff, axis=1))
        temp_dist = np.array(temp_dist)
        temp_closest = np.argsort(temp_dist, axis=0)
        closest.append(temp_closest)
        
    return closest

#%% find 'clos' closest ids
kmeans_closest = closest_ids(dim_red, kmeans_centroids)

clos = 100
closest = [i[:clos] for i in kmeans_closest]
closest_cells_id = np.array(closest).T

#%% plot a figure for each cluster
from skimage.util import montage

for method in range(len(kmeans_closest)):
    mont = montage(cells[closest_cells_id[:,:,method].ravel()],
            rescale_intensity=True,
            grid_shape=cells[closest_cells_id[:,:,method]].shape[:2])

    plt.figure()
    plt.tight_layout()
    plt.title(name[method])
    plt.imshow(mont, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig('plots/'+name[method]+'_mont_all', dpi=1000, bbox_inches='tight')
    plt.close('all')

#%% EVR
evr = p_20.explained_variance_ratio_
cum_sum = np.cumsum(evr)
plt.title('PCA Explained Variance Ratio - 20 Components')
plt.plot(np.arange(1,len(evr)+1), evr, c='b', label='Explained Variance Ratio')
plt.plot(np.arange(1,len(evr)+1), cum_sum, c='r', label='Cumulative Explained Variance Ratio')
plt.legend()
plt.ylim([0,1])
plt.ylabel('Explained Variance Ratio')
plt.yticks(np.arange(0,1.1,0.1))
plt.xlim([0,len(evr)+1])
plt.xlabel('Number of Components')
plt.xticks(np.arange(1,len(evr)+1))

#%% duplicates
for i in range(len(name)):
    print(name[i], no_clus*clos-np.unique(closest_cells_id[:,:,i]).size)


#%% closest and closest match - OLD
# closest = [i[0] for i in kmeans_closest]
# closest_cells_id = np.array(closest).T

# #%% show closest cells based on cluster assignment
# fig = plt.figure()
# fig.suptitle('Closest Cell Based on Cluster Assignment')
# for i, cc_id in enumerate(closest_cells_id.ravel()):
#     ax = fig.add_subplot(closest_cells_id.shape[0],closest_cells_id.shape[1],i+1)
#     ax.imshow(cells[cc_id], cmap='gray')
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     if i in range(closest_cells_id.shape[1]):
#         ax.set_title(name[i])
        
#     if i%closest_cells_id.shape[1] == 0:
#         ax.set_ylabel('cluster ' + str(i//closest_cells_id.shape[1]+1))
        
# #%% order images according to closest match
# from scipy.optimize import linear_sum_assignment

# # use sum of pixel intensity to determine match
# sums = cells[closest_cells_id].sum(axis=(2,3))
# # take first dim red method as base
# base_method = 'tsne'
# base_col = name.index(base_method)
# base = sums[:,base_col]

# # match other dim red methods to first
# # init similar as first dim red method IDs
# similar = closest_cells_id.T[base_col]
# for i,col in enumerate(np.delete(sums,base_col,axis=1).T):
#     temp = []
#     for elem in col:
#         temp.append(abs(elem-base))
    
#     # cost is abs difference between pixel intensity of first and current dim red method
#     cost = np.array(temp).T
    
#     # solve using Hungarian algorithm
#     row_ind, col_ind = linear_sum_assignment(cost)
    
#     # reorder dim red col to create similar closest_cells_id array    
#     col_cc_id = np.delete(closest_cells_id,base_col,axis=1).T[i]
#     similar = np.column_stack([similar, col_cc_id[col_ind]])

# #%% show closest cells based on similarity to base dim red method
# fig = plt.figure()
# fig.suptitle('Closest Cell Based on Similarity to Base Method ('+base_method+')')
# for i, cc_id in enumerate(similar.ravel()):
#     ax = fig.add_subplot(similar.shape[0],similar.shape[1],i+1)
#     ax.imshow(cells[cc_id], cmap='gray')
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     if i in range(similar.shape[1]):
#         ax.set_title(name[i])

#     if i == (similar.shape[0]-1)*similar.shape[1]+base_col:
#         ax.set_xlabel('^ base method ^')
