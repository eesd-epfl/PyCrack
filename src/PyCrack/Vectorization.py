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
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage as spim
#import os
import matplotlib.pyplot as plt
import skimage.io as sio

class Features:
     
    def __init__(self, sigma=1, method_center='medial_axis', path=None, extension=None, reshape=None, as_gray=True, verbose=False, showfig=False):
        
        # Attributes
        self.sigma = sigma
        self.method_center = method_center
        self.path = path
        self.extension = extension
        self.reshape = reshape
        self.as_gray = as_gray
        self.verbose = verbose
        self.X = None
        self.nodes = None
        self.Xcontour = None
        self.skeleton_coarse = None
        self.nodes_coarse = None
        self.coords_crack = None
        self.coords_skeleton = None
        self.images = None
        self.images_normalized = None
        self.showfig = showfig
        
        if path is not None:
            if extension is None:
                raise TypeError('PyCrack: extension cannot be NoneType when using path.')
            else:
                self._read_images(path, extension, reshape=None, as_gray=True)
        
    
    def fit(self,X=None):
        
        # X is either an image with the cracks identified or an skeletonized image.
        if X is None:
            raise ValueError('PyCrack: X not found.')
            
        self.X = skeletonize(X)*1.0 # times 1.0 to get a double quantity.
        self._coarse()
    
    def _read_images(self,path=None, extension=None, reshape=None, as_gray=True):
        
        if path is not None:
            if extension is None:
                raise TypeError('PyCrack: extension cannot be NoneType when using path.')
        
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

            
    def _coarse(self):

        sigma = self.sigma
        method_center = self.method_center
        Xcoarse = spim.gaussian_filter(self.X, sigma=sigma)
        xmean = np.mean(Xcoarse)

        Xcontour = np.zeros(np.shape(Xcoarse))
        for i in range(np.shape(Xcoarse)[0]):
            for j in range(np.shape(Xcoarse)[1]):

                if Xcoarse[i,j]>xmean:
                    Xcontour[i,j]=1
                else:
                    Xcontour[i,j]=0
                    
        self.Xcontour = Xcontour # Attribute
        
        # Compute the centerline/skeleton
        if method_center=='medial_axis':
            skeleton_coarse = medial_axis(Xcontour)*1.0
        elif method_center=='skeletonize':
            skeleton_coarse = skeletonize(Xcontour)*1.0
        else:
            raise ValueError('PyCrack: no method to compute the skeleton/centerline is selected.')
        
        if self.verbose: print('PyCrack: get nodes and coordinates.')
            
        self.skeleton_coarse = skeleton_coarse # Attribute
        nodes_coarse = self._getnodes(skeleton_coarse)
        coords_crack = self._getcoords(self.X)
        coords_skeleton = self._getcoords(skeleton_coarse)
        
        # Attributes
        self.nodes_coarse = nodes_coarse
        self.coords_crack = coords_crack
        self.coords_skeleton = coords_skeleton
        
        if self.showfig:
            
            if self.verbose: print('PyCrack: plot graphic.')
                
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.imshow(Xcontour, cmap=plt.cm.gray)
            ax.scatter(coords_crack[:,0],coords_crack[:,1],s=0.05,c='y')
            ax.scatter(coords_skeleton[:,0],coords_skeleton[:,1],c='b',marker='o', s=(72./fig.dpi/8)**2)
            ax.scatter(nodes_coarse[:,0],nodes_coarse[:,1],s=20,c='red')
            ax.axis('off')
            for i in range(len(nodes_coarse)):
                ax.text(nodes_coarse[i,0],nodes_coarse[i,1],s=str(i))

            fig.tight_layout()
            plt.show()


    @staticmethod
    def _getcoords(Y):
        
        y, x = np.nonzero(Y > 0)
        
        coords = []
        for i in range(len(x)):
            coords.append([x[i],y[i]])
            
        coords = np.array(coords)
        
        return coords
    
    @staticmethod
    def _getnodes(Xin):

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
    
class Node:  
    
    def __init__(self,node_id=None,coordinates=None,connectivity=None,length_threads=None):
        
        self.node_id = node_id
        self.coordinates = coordinates
        self.connectivity = connectivity
        self.length_threads = length_threads
    
class Nodes:
     
    def __init__(self,window=None,pixel_dim=1):
        
        self.window = window
        self.features_obj = None
        self.node = None
        self.pixel_dim=pixel_dim
        self.nnodes = None
        
    def fit(self,X=None):
        
        # X is an object of Feature.
        if X is None:
            raise ValueError('PyCrack: X not found.')
            
        self.features_obj = X
        
        nnodes = len(self.features_obj.nodes_coarse)
        self.nnodes = nnodes
        nodes_coarse = self.features_obj.nodes_coarse

        nodes_list = []
        for i in range(nnodes):
            node = nodes_coarse[i,:]
            connectivity, distance = self._walker(node)
            nodes_list.append(Node(node_id=i,coordinates=node,connectivity=connectivity,length_threads=distance))
        
        self.node = nodes_list
        
    def _get_branches(self,node):

        window = self.window
        skel = self.features_obj.skeleton_coarse
        nside = 2*window+1

        skpad = skel.copy()
        skpad = np.pad(skpad, window, 'constant', constant_values=0)

        p = node+window


        r0 = p[1]-window
        r1 = p[1]+window+1
        c0 = p[0]-window
        c1 = p[0]+window+1

        f=skpad[r0:r1,c0:c1]
        fmask=np.ones((nside,nside))
        fmask[1:nside-1,1:nside-1]=0

        ffil = f*fmask

        n_branches = int(np.sum(ffil))
        idx = np.where(ffil == 1)

        # cols, rows
        idx=[(node[0]+(idx[1]-window)), (node[1]+(idx[0]-window))]

        #if show_fig:
        #    plt.imshow(f,cmap='gray')
        #    plt.scatter([window],[window],c='r')
        #    plt.show()

        positions = []
        for i in range(n_branches):
            positions.append([idx[0][i],idx[1][i]])     

        return n_branches, positions


    def _walker(self,node):

        n_branches, initial_positions = self._get_branches(node)
        coords_skl = self.features_obj.coords_skeleton
        nodes = self.features_obj.nodes_coarse
        
        nbrs0 = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(coords_skl)
        nnodes = len(nodes)
        nodes_global = []
        for i in range(nnodes):
            distances, indices = nbrs0.kneighbors([nodes[i,:]])
            nodes_global.append(indices[0][0])

        #n_branches = len(positions)
        infty = float("inf")


        nn=3 # fixed [close_left,ref,close_right]
        nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(coords_skl)

        #posi = []
        distance_edges = []
        connectivity = []
        for i in range(n_branches):

            p0 = initial_positions[i]
            distances, indices = nbrs.kneighbors([p0])

            weights = np.ones(np.max(np.shape(coords_skl)))
            dist = []
            for j in range(nn):
                y1=coords_skl[indices[0][j]]
                dist.append(np.linalg.norm(y1-node))

            weights[indices[0][np.argsort(dist)[0]]] = infty
            weights[indices[0][np.argsort(dist)[1]]] = infty

            p_ref = indices[0][np.argsort(dist)[-1]]

            k=0
            boolean=True

            #dsum = 0
            dsum = np.linalg.norm(node - initial_positions[i]) # approximate the distance within the window.
            while boolean:
                if len(np.where(nodes_global==p_ref)[0])==1:
                    connectivity.append(np.where(nodes_global==p_ref)[0][0])
                    pcon = p_ref
                    boolean=False
                else:

                    d, idx = nbrs.kneighbors([coords_skl[p_ref]])
                    idx1 = idx[0][1:]
                    w1 = weights[idx1]

                    weights[p_ref] = infty
                    p_ref = idx1[np.isfinite(w1)][0]

                    d1 = d[0][1:]
                    dsum = dsum + d1[np.isfinite(w1)][0] #distances in pixels.


            distance_edges.append(dsum*self.pixel_dim)
            #posi.append(pcon)

        return connectivity, distance_edges    
        
        
class Thread:
        # Threads of the web!
        def __init__(self,thread_id=None, nodes=None, length=None):
            
            self.thread_id = thread_id
            self.nodes = nodes # object
            self.lenght = length
            
        
class Threads:

        def __init__(self):

            #self.window = window
            self.nodes_obj = None
            self.thread = None
            self.nthreads = None
            #self.pixel_dim=pixel_dim

        def fit(self,X=None):

            # X is an object of Nodes.
            if X is None:
                raise ValueError('PyCrack: X not found.')
                
            self.nodes_obj = X
            threads_nodes, threads_length = self._get_threads()
            
            nthreads = len(threads_nodes)
            self.nthreads = nthreads
           
            threads_list = []
            for i in range(nthreads):
                threads_list.append(Thread(thread_id=i,nodes=threads_nodes[i], length=threads_length[i]))
        
            self.thread = threads_list
                
        def _get_threads(self):
            
            nnodes = self.nodes_obj.nnodes
            
            thread_nodes = []
            length = []
            for i in range(nnodes):
                
                node_obj = self.nodes_obj.node[i]
                connectivity = node_obj.connectivity
                length_threads = node_obj.length_threads
                ncon = len(connectivity)
                
                for j in range(ncon):
                    
                    if connectivity[j]>i:
                        #thread_nodesid.append([i,connectivity[j]])
                        thread_nodes.append([node_obj,self.nodes_obj.node[connectivity[j]]])
                        length.append(length_threads[j])
                
            return thread_nodes, length  
            
            
            
        
        




