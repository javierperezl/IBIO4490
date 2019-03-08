#!/usr/bin/env python3

def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
   
    # colorSpace : 'rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy' or 'hsv+xy'
    # clusteringMethod = 'kmeans', 'gmm', 'hierarchical' or 'watershed'.
    # numberOfClusters positive integer (larger than 2)
    
    import matplotlib.pyplot as plt
    from skimage import io, color
    import numpy as np
    imgSize = rgbImage.shape
    x = np.zeros((rgbImage.shape[0],rgbImage.shape[1]),dtype="uint8")
    y = np.zeros((rgbImage.shape[0],rgbImage.shape[1]),dtype="uint8")
    for i in range(rgbImage.shape[0]):
        for j in range(rgbImage.shape[1]):
            x[i][j] = i
            y[i][j] = j
    
    if colorSpace == 'rgb':
        imgColorSp = rgbImage
        if clusteringMethod == 'hierarchical':
            imgColorSp = np.resize(imgColorSp,(100,100,3)) #para jerarquico se redimensiona a 100x100x3
            imgColorSpRes = np.reshape(imgColorSp,(10000,imgColorSp.shape[2]))
        else:
            imgColorSpRes = np.reshape(imgColorSp,(imgColorSp.shape[0]*imgColorSp.shape[1],imgColorSp.shape[2]))
    elif colorSpace == 'lab':
        imgColorSp = color.rgb2lab(rgbImage)
        if clusteringMethod == 'hierarchical':
            imgColorSp = np.resize(imgColorSp,(100,100,3))
            imgColorSpRes = np.reshape(imgColorSp,(10000,imgColorSp.shape[2]))
        else:
            imgColorSpRes = np.reshape(imgColorSp,(imgColorSp.shape[0]*imgColorSp.shape[1],imgColorSp.shape[2]))
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(imgColorSpRes)
        imgColorSpRes = scaler.transform(imgColorSpRes) #se normalizan los tres canales
    
    elif colorSpace == 'hsv':
        imgColorSp = color.rgb2hsv(rgbImage)
        if clusteringMethod == 'hierarchical':
            imgColorSp = np.resize(imgColorSp,(100,100,3))
            imgColorSpRes = np.reshape(imgColorSp,(10000,imgColorSp.shape[2]))
        else:
            imgColorSpRes = np.reshape(imgColorSp,(imgColorSp.shape[0]*imgColorSp.shape[1],imgColorSp.shape[2]))
    elif colorSpace == 'rgb+xy':     
        imgColorSp = np.dstack((rgbImage,x,y))
        if clusteringMethod == 'hierarchical':
            imgColorSp = np.resize(imgColorSp,(100,100,3))
            imgColorSpRes = np.reshape(imgColorSp,(10000,imgColorSp.shape[2]))
        else:
            imgColorSpRes = np.reshape(imgColorSp,(imgColorSp.shape[0]*imgColorSp.shape[1],imgColorSp.shape[2]))
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(imgColorSpRes)
        imgColorSpRes = scaler.transform(imgColorSpRes) #se normalizan los cinco canales
        
    elif colorSpace == 'lab+xy':
        imgColorSp = np.dstack((color.rgb2lab(rgbImage),x,y))
        if clusteringMethod == 'hierarchical':
            imgColorSp = np.resize(imgColorSp,(100,100,3))
            imgColorSpRes = np.reshape(imgColorSp,(10000,imgColorSp.shape[2]))
        else:
            imgColorSpRes = np.reshape(imgColorSp,(imgColorSp.shape[0]*imgColorSp.shape[1],imgColorSp.shape[2]))
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(imgColorSpRes)
        imgColorSpRes = scaler.transform(imgColorSpRes) #se normalizan los cinco canales
        
    elif colorSpace == 'hsv+xy':
        imgColorSp = np.dstack((color.rgb2hsv(rgbImage),x,y))
        if clusteringMethod == 'hierarchical':
            imgColorSp = np.resize(imgColorSp,(100,100,3))
            imgColorSpRes = np.reshape(imgColorSp,(10000,imgColorSp.shape[2]))
        else:
            imgColorSpRes = np.reshape(imgColorSp,(imgColorSp.shape[0]*imgColorSp.shape[1],imgColorSp.shape[2]))
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(imgColorSpRes)
        imgColorSpRes = scaler.transform(imgColorSpRes) #se normalizan los cinco canales
    
    if clusteringMethod == 'kmeans':
        #kmeans
        from sklearn.cluster import KMeans
    
        imgSize = rgbImage.shape
        kmeans = KMeans(numberOfClusters).fit(imgColorSpRes)
        # Predicting the clusters
        segmentation = kmeans.predict(imgColorSpRes)
        segmentation = np.reshape(segmentation ,(imgSize[0],imgSize[1]))
        #adapted from http://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/
        
    elif clusteringMethod == 'gmm':
        #from sklearn.mixture import GaussianMixture
        from sklearn.mixture import GaussianMixture
        imgSize = rgbImage.shape
        gmm = GaussianMixture(numberOfClusters).fit(imgColorSpRes)
        segmentation = gmm.predict(imgColorSpRes)
        segmentation = np.reshape(segmentation ,(imgSize[0],imgSize[1]))
        
    elif clusteringMethod == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        cluster = AgglomerativeClustering(numberOfClusters, affinity='euclidean', linkage='ward')      
        segmentation = cluster.fit_predict(imgColorSpRes)  
        segmentation = np.reshape(segmentation ,(100,100))
        #adapted from https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    
    elif clusteringMethod == 'watershed':
        #watershed
        import numpy as np
        from scipy import ndimage as ndi
        from skimage import feature
        from skimage import morphology
        
        imgColorSp = np.mean(imgColorSp,axis=2) #se promedian los canales respecto a la 3ra dim
        maxlocal = feature.peak_local_max(imgColorSp, num_peaks=numberOfClusters, num_peaks_per_label=1, indices=False)
        marcadores = ndi.label(maxlocal)[0]
        segmentation = morphology.watershed(-imgColorSp, marcadores, mask=imgColorSp)
        #addapted from http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html   
    
    segmentation = segmentation + 1 #para evitar que hayan grupos con etiqueta #0
    return segmentation
    #return imgColorSpRes

def evalSegmentation (gt, segmentation):
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import imageio
    import numpy as np

    segmentation_gt=gt['groundTruth'][0,5][0][0]['Segmentation']
    maxLabelGT = np.amax(segmentation_gt)

    ratio = np.zeros((1,maxLabelGT),dtype="double")
    for i in range(maxLabelGT):
        real = np.where(segmentation_gt== i, 1, 0)
        overlap = real*segmentation
        
        total = (real==1).sum()
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(overlap)
        overlap = scaler.transform(overlap) #normalizar
        overlap = np.where(overlap==1,1,0)
        if (overlap==1).sum()==0 or total==0:
            ratio[0][i] = 0
        else:
            ratio[0][i] = ((overlap==1).sum())/total
    
    ratioGlobal = np.mean(ratio)
    return ratioGlobal


import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np
"""
img = 'BSDS_small/train/22090.jpg'
rgbImage = io.imread(img)
colorSpace = 'hsv'
clusteringMethod = 'gmm'
numberOfClusters = 5
segmentation = segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters)

plt.imshow(segmentation, cmap=plt.get_cmap('tab20b')) #or another colormap that you like https://matplotlib.org/examples/color/colormaps_reference.html
plt.show()

import scipy.io as sio


# Load .mat
gt=sio.loadmat(img.replace('jpg', 'mat'))

#Load segmentation from sixth human
segm=gt['groundTruth'][0,5][0][0]['Segmentation']
plt.imshow(segm, cmap=plt.get_cmap('hot'))
plt.colorbar()
plt.show()


ratio = evalSegmentation (gt, segmentation)
print(ratio)

"""
