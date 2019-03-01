#!/usr/bin/env python3
"""
Created on Tue Feb 26 14:41:14 2019

@author: Javier Pérez
"""

import sys
sys.path.append('python')

def unpickle(file):
    import pickle
    import numpy as np
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='latin1')
        _dict['labels'] = np.array(_dict['labels'])
        _dict['data'] = _dict['data'].reshape(_dict['data'].shape[0], 3, 32, 32).transpose(0,2,3,1)

    return _dict

def get_data(data, sliced=1):
    from skimage import color
    import numpy as np
    data_x = data['data']
    data_x = color.rgb2gray(data_x)
    data_x = data_x[:int(data_x.shape[0]*sliced)]
    data_y = data['labels']
    data_y = data_y[:int(data_y.shape[0]*sliced)]
    return data_x, data_y

def merge_dict(dict1, dict2):
    import numpy as np
    if len(dict1.keys())==0: return dict2
    new_dict = {key: (value1, value2) for key, value1, value2 in zip(dict1.keys(), dict1.values(), dict2.values())}
    for key, value in new_dict.items():
        if key=='data':
            new_dict[key] = np.vstack((value[0], value[1]))
        elif key=='labels':
            new_dict[key] = np.hstack((value[0], value[1]))            
        elif key=='batch_label':
            new_dict[key] = value[1]
        else:
            new_dict[key] = value[0] + value[1]
    return new_dict

def load_cifar10(meta='cifar-10-batches-py', mode=1):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(i+1))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict



def run(k=16, nxclass=10, numimtest = 10000, small_filter=True):
    import time
    import numpy as np
    #Load sample images from disk
    from skimage import color
    from skimage import io
    from skimage.transform import resize

    start_time = time.time()
    datos = load_cifar10(mode=3)
    
    ##Create a filter bank with deafult params
    from fbCreate import fbCreate
    if small_filter:
        fb = fbCreate(support=3, elong = 1, startSigma=0.6)
    else:
        fb = fbCreate(support=2, elong = 2, startSigma=0.6) #, vis=True) # fbCreate(**kwargs, vis=True) for visualization
    
    
    j=0
    imBase = np.zeros((nxclass*10)).tolist()
    labelsBase = np.zeros((nxclass*10)).tolist()
    cont = np.zeros((10,), dtype=int)
    for i in range(10000):
        cont[datos["labels"][i]] = cont[datos["labels"][i]] + 1 
        if not cont[datos["labels"][i]] > nxclass:
            imBase[j] = color.rgb2gray(resize(datos["data"][i], (32, 32)))
            labelsBase[j] = datos["labels"][i]
            j = j + 1
    
    
    #Apply filterbank to sample image
    from fbRun import fbRun
    
    filterResponses = fbRun(fb,np.hstack(imBase))
    
    #Computer textons from filter
    from computeTextons import computeTextons
    map, textons = computeTextons(filterResponses, k)
    
    datos_test = load_cifar10(mode="test") #test images
    
    imTest = np.zeros((numimtest)).tolist()
    labelsTest = np.zeros((numimtest)).tolist()
    for i in range(numimtest):
        imTest[i] = color.rgb2gray(resize(datos_test["data"][i], (32, 32)))
        labelsTest[i] = datos["labels"][i]
    
    
    tmapBase = np.zeros(len(imBase)).tolist()
    #Calculate texton representation with current texton dictionary
    from assignTextons import assignTextons
    for i in range(len(imBase)):
        tmapBase[i] = assignTextons(fbRun(fb,imBase[i]),textons.transpose())
    
    tmapTest = np.zeros(len(imTest)).tolist()
    for i in range(len(imTest)):
        tmapTest[i] = assignTextons(fbRun(fb,imTest[i]),textons.transpose())
    
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    def histc(X, bins):
        import numpy as np
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return np.array(r)
    
    
    histBase = np.zeros(len(tmapBase)).tolist()
    for i in range(len(tmapBase)):
        histBase[i] = histc(tmapBase[i].flatten(), np.arange(k))/(len(tmapBase[i])**2)
    
    histTest = np.zeros(len(tmapTest)).tolist()
    for i in range(len(tmapTest)):
        histTest[i] = histc(tmapTest[i].flatten(), np.arange(k))/(len(tmapTest[i])**2)
    
    #Intersection kernel metric
    def intersection(x,y):
        import numpy as np
        min = np.minimum(x,y)
        d = 1 - np.sum(min)
        return d
    
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(weights='distance',metric=intersection)
    neigh.fit(histBase, labelsBase)
    predictKNN = neigh.predict(histTest)
    
    from sklearn.metrics import confusion_matrix
    ConfMatTestkNN = confusion_matrix(labelsTest,predictKNN)
    NormConfMatTestkNN = ConfMatTestkNN.astype('float') / ConfMatTestkNN.sum(axis=1)[:, np.newaxis]
    ACATestkNN = sum(NormConfMatTestkNN.diagonal())/10
    
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    clf.fit(histBase, labelsBase) 
    predictRF=clf.predict(histTest)
    
    ConfMatTestRF = confusion_matrix(labelsTest,predictRF)
    
    NormConfMatTestRF = ConfMatTestRF.astype('float') / ConfMatTestRF.sum(axis=1)[:, np.newaxis]
    
    ACATestRF = sum(NormConfMatTestRF.diagonal())/10
    
    
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    #tomado de https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
    
    
    # Plot normalized confusion matrix kNN
    plt.figure()
    plot_confusion_matrix(NormConfMatTestkNN, classes=class_names, normalize=True,
                          title='Normalized confusion matrix with kNN')
    #plt.show()
    plt.savefig('ncmkNN'+str(k)+'.jpg', format='jpg', dpi=300)
    
    # Plot normalized confusion matrix RF
    plt.figure()
    plot_confusion_matrix(NormConfMatTestRF, classes=class_names, normalize=True,
                          title='Normalized confusion matrix with RF')
    #plt.show()
    plt.savefig('ncmRF'+str(k)+'.jpg', format='jpg', dpi=300)
    print("RESULTADOS:  NXCLASS=%s, k=%s, numimtest=%s", (nxclass, k, numimtest))
    print("El ACA de Test con kNN es: ",ACATestkNN)
    print("El ACA de Test con RF es: ",ACATestRF)
    print("Total time: %s seconds " % (time.time() - start_time)) #time
    return ACATestkNN, ACATestRF
if __name__ == "__main__":
    nxclass = 5
    numimtest = 100
    #Set number of clusters
    for k in [16*4]:
        for small_filter in [True, False]:
           ACATestkNN, ACATestRF = run(k, nxclass, numimtest, small_filter)
