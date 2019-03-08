#!/usr/bin/env python3

def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def groundtruth(img_file):
    import scipy.io as sio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,5][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    
    """def check_dataset(folder):
    import os
    import urllib
    import zipfile
    from StringIO import StringIO
    zipdata = StringIO()
    zipdata.write(get_zip_data())
    myzipfile = zipfile.ZipFile(zipdata)"""
    """if not os.path.exists(folder): #si no existe el dataset lo descarga
        urlD = 'http://157.253.196.67/BSDS_small.zip'
        urllib.request.urlretrieve (urlD, "BSDS_small.zip")

    with zipfile.ZipFile("BSDS_small","r") as zip_ref:
        zip_ref.extractall("BSDS_small")"""
    
if __name__ == '__main__':
    import argparse
    import imageio
    #from Segment import segmentByClustering # Change this line if your function has a different name
    parser = argparse.ArgumentParser()

    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)
	
    opts = parser.parse_args()

    #check_dataset(opts.img_file.split('/')[0])
    from Segment import segmentByClustering # Change this line if your function has a different name

    img = imageio.imread(opts.img_file)
    seg = segmentByClustering(rgbImage=img, colorSpace=opts.color, clusteringMethod=opts.method, numberOfClusters=opts.k)
    imshow(img, seg, title='Prediction')
    groundtruth(opts.img_file)