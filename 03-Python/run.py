#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 08:34:11 2019

@author: Javier Pérez
"""
import numpy as np
import shutil
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageDraw

import os

import urllib
import zipfile

import time

# [2] 
#tomado de https://stackoverflow.com/questions/12332975/installing-python-module-within-code
def install_and_import(package, lib): 
    import importlib
    try:
        importlib.import_module(lib)
        print("Paquete encontrado: "+package)
    except ImportError:
        print("Instalando paquete: "+package)
        import pip
        pip.main(['install', package])
    finally:
        print("Importando paquete: "+package)
        globals()[package] = importlib.import_module(lib)

install_and_import('python-resize-image', 'resizeimage')

install_and_import('python-resize-image', 'resizeimage')

start_time = time.time()

if not os.path.exists('dataset'): #si no existe el dataset lo descarga
    urlD = 'https://www.dropbox.com/s/3ptvx317p52f6d6/AwA2.zip?dl=1'
    urllib.request.urlretrieve (urlD, "AwA2.zip")

    with zipfile.ZipFile("AwA2.zip","r") as zip_ref:
        zip_ref.extractall("dataset")
    
N = 12; #12 imagenes

randIdx = np.zeros((N,), dtype=int)
for x in range(N):
    randIdx[x] = np.random.randint(1,450 + 1) #genera N=12 al azar entre 1-450 



listAwA2 = os.listdir("dataset/AwA2") #todas las imgs deL dataset
imgBig = np.zeros((256*3,256*4,3),dtype="uint8") #img gigante que tendra las N=12 imgs

if os.path.exists('AwA2Crop'): #verifica si existe la nueva carpte
    shutil.rmtree('AwA2Crop') #si existe la borra

os.mkdir('AwA2Crop') #crea una carpeta para las imgs recortadas
dictImg = {}
#print(listAwA2)

labels = [] #etiquetas de las imagenes

for x in range(N):
    with open('dataset/AwA2/'+listAwA2[randIdx[x]], 'r+b') as f:
        with Image.open(f) as image:
            cropImg = resizeimage.resize_cover(image, [256, 256]) #[1]
            cropImg.save('AwA2Crop/'+listAwA2[randIdx[x]], image.format) #[1]
            dictImg[x] = mpimg.imread('AwA2Crop/'+listAwA2[randIdx[x]]) #lee la img
            aux = listAwA2[randIdx[x]].split('_') #obtiene etiqueta de la img
            labels.append(aux[0]) #guarda etiqueta
            

#reemplaza c/imagen en la posicion correspondiente en la nueva img grande
imgBig[256*0:256*1,256*0:256*1,:] = dictImg[0];
imgBig[256*0:256*1,256*1:256*2,:] = dictImg[1];
imgBig[256*0:256*1,256*2:256*3,:] = dictImg[2];
imgBig[256*0:256*1,256*3:256*4,:] = dictImg[3];

imgBig[256*1:256*2,256*0:256*1,:] = dictImg[4];
imgBig[256*1:256*2,256*1:256*2,:] = dictImg[5];
imgBig[256*1:256*2,256*2:256*3,:] = dictImg[6];
imgBig[256*1:256*2,256*3:256*4,:] = dictImg[7];

imgBig[256*2:256*3,256*0:256*1,:] = dictImg[8];
imgBig[256*2:256*3,256*1:256*2,:] = dictImg[9];
imgBig[256*2:256*3,256*2:256*3,:] = dictImg[10];
imgBig[256*2:256*3,256*3:256*4,:] = dictImg[11];


image = Image.fromarray(imgBig,'RGB') #guarda img grande como Image de array

draw = ImageDraw.Draw(image) #para escribir sobre la img grande
#escribe texto centrado sobre cada img
draw.text(xy=(100+256*0,120+256*0),text=labels[0],fill=(255,255,0))
draw.text(xy=(100+256*1,120+256*0),text=labels[1],fill=(255,255,0))
draw.text(xy=(100+256*2,120+256*0),text=labels[2],fill=(255,255,0))
draw.text(xy=(100+256*3,120+256*0),text=labels[3],fill=(255,255,0))
draw.text(xy=(100+256*0,120+256*1),text=labels[4],fill=(255,255,0))
draw.text(xy=(100+256*1,120+256*1),text=labels[5],fill=(255,255,0))
draw.text(xy=(100+256*2,120+256*1),text=labels[6],fill=(255,255,0))
draw.text(xy=(100+256*3,120+256*1),text=labels[7],fill=(255,255,0))
draw.text(xy=(100+256*0,120+256*2),text=labels[8],fill=(255,255,0))
draw.text(xy=(100+256*1,120+256*2),text=labels[9],fill=(255,255,0))
draw.text(xy=(100+256*2,120+256*2),text=labels[10],fill=(255,255,0))
draw.text(xy=(100+256*3,120+256*2),text=labels[11],fill=(255,255,0))

print("--- %s seconds ---" % (time.time() - start_time)) #time

image.show() #muestra img

"""
plt.figure()
plt.imshow(imgBig)
plt.show()
"""



#References
#[1] Python Software Foundation. python-resize-image 1.1.18 https://pypi.org/project/python-resize-image/
#[2] stackoverflow https://stackoverflow.com/questions/12332975/installing-python-module-within-code










