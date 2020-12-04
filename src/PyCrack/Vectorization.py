# PyCrack is distributed under the MIT license.
#
# Copyright (C) 2020  -- Katrin Beyer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
``Vectorization`` is the module for ``PyCrack`` to perform ...

This module contains the classes and methods to perform ...

The module currently contains the following classes:

* ``ClassA``: Class for...

* ``ClassB``: Class for...

"""

import numpy as np
import scipy as sp
from skimage.morphology import medial_axis, skeletonize
import scipy.ndimage as spim
import os
import matplotlib.pyplot as plt
import skimage.io as sio

class Features:
     
    def __init__(self,verbose=False):
        
        # Attributes
        self.verbose = verbose
        self.X = None
        self.nodes = None
        self.Xcontour = None
        self.skel = None
        self.images = None
        self.images_normalized = None
        
    
    def fit(self,X=None):
        
        if X is None:
            raise ValueError('PyCrack: no X was found.')
            
        self.X = skeletonize(X)*1.0
    
    def read_images(self,path, extension, reshape=None, as_gray=True):
        
        fnames = [f for f in os.listdir(path) if f.endswith('.'+extension)]
        fnames = np.sort(fnames)

        if reshape is not None:
            if len(reshape) != 2:
                raise TypeError('PyCrack: reshape must be a list or ndarray with two positions.')

        images = []
        images_normalized = []
        for filenames in fnames:

            im=sio.imread(os.path.join(path , filenames), as_gray=as_gray)

            # resize to given size (if given)
            if (reshape is not None):
                im = im.resize(reshape, Image.ANTIALIAS) 

            #X.append(np.asarray(im, dtype=np.uint8)) 
            images.append(np.array(im))
            images_normalized.append(1-np.asarray(im, dtype=np.int64)/255)
            
            self.images = images
            self.images_normalized = images_normalized
        
    @staticmethod
    def getnodes(Xin):

        # hits
        structures = []
        structures.append([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        structures.append([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        structures.append([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        structures.append([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        structures.append([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        structures.append([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
        structures.append([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        structures.append([[0, 0, 0], [1, 1, 0], [0, 0, 0]])

        #structures.append([[1, 1, 1], [0, 1, 1], [1, 0, 0]])
        #structures.append([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
        #structures.append([[1, 0, 0], [0, 1, 1], [1, 1, 1]])
        #structures.append([[0, 0, 1], [1, 1, 0], [1, 1, 1]])

        crossings = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                     [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                     [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                     [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                     [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                     [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                     [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                     [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                     [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                     [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                     [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                     [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                     [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                     [1,0,1,1,0,1,0,0],[0,1,1,1,1,0,0,1],[1,1,0,1,0,1,1,1],
                     [1,1,1,1,0,1,0,0],[0,1,0,0,1,1,1,1]];

        for i in range(len(crossings)):
            A = crossings[i]
            B = np.ones((3,3))

            B[1,0] = A[0]
            B[0,0] = A[1]
            B[0,1] = A[2]
            B[0,2] = A[3]
            B[1,2] = A[4]
            B[2,2] = A[5]
            B[2,1] = A[6]
            B[2,0] = A[7]

            structures.append(B.tolist())

        nodes = []
        for i in range(len(structures)):
            structure1 = np.array(structures[i])
            X0 = spim.binary_hit_or_miss(Xin, structure1=structure1).astype(int)
            r0, c0 = np.nonzero(X0 == 1) 

            for j in range(len(r0)):
                nodes.append([c0[j],r0[j]])

        nodes = np.array(nodes)
        
        return nodes
        
    def contour(self, sigma=1, method_center='medial_axis', showfig=False):

        Xfil = spim.gaussian_filter(self.X, sigma=sigma)
        xmean = np.mean(Xfil)

        Xcontour = np.zeros(np.shape(Xfil))
        for i in range(np.shape(Xfil)[0]):
            for j in range(np.shape(Xfil)[1]):

                if Xfil[i,j]>xmean:
                    Xcontour[i,j]=1
                else:
                    Xcontour[i,j]=0
                    
        self.Xcontour = Xcontour
        # Compute the centerline/skeleton
        if method_center=='medial_axis':
            skel = medial_axis(Xcontour)*1.0
        elif method_center=='skeletonize':
            skel = skeletonize(Xcontour)*1.0
        else:
            raise ValueError('PyCrack: not identified method to compute the skeleton/centerline.')
        
        self.skel = skel
        if self.verbose: print('PyCrack: get nodes and coordinates.')
        nodes_contour = self.getnodes(skel)
        coords_x = self.getcoords(self.X)
        coords_c = self.getcoords(skel)
        
        if showfig:
            if self.verbose: print('PyCrack: plot graphic.')
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.imshow(Xcontour, cmap=plt.cm.gray)
            ax.scatter(coords_x[:,0],coords_x[:,1],s=0.05,c='y')
            ax.scatter(coords_c[:,0],coords_c[:,1],c='b',marker='o', s=(72./fig.dpi/8)**2)
            ax.scatter(nodes_contour[:,0],nodes_contour[:,1],s=20,c='red')
            ax.axis('off')
            for i in range(len(nodes_contour)):
                ax.text(nodes_contour[i,0],nodes_contour[i,1],s=str(i))

            fig.tight_layout()
            plt.show()

    @staticmethod
    def getcoords(Y):
        
        y, x = np.nonzero(Y > 0)
        
        coords = []
        for i in range(len(x)):
            coords.append([x[i],y[i]])
            
        coords = np.array(coords)
        
        return coords

    
        

    
    
    
