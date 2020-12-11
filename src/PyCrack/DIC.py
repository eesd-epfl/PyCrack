"""
``DIC`` is the module for ``pyopt`` to perform ...

This module contains the classes and methods to perform ...

The module currently contains the following classes:

* ``ClassA``: Class for...

* ``ClassB``: Class for...

"""

import numpy as np
import scipy as sp

class ImageCorrelation:
     
    def __init__(self,pixel_dim=None):
        
        self.pixel_dim = pixel_dim
       
    def CropImage(self,im_source=None,center_row=None,center_col=None,width=None):
        
        if width & 1:
            # odd
            side = int((width - 1)/2)
            row_0 = center_row - side
            col_0 = center_col - side
            
            row_1 = center_row + side
            col_1 = center_col + side
            
            id_row = np.arange(row_0,row_1+1)
            id_col = np.arange(col_0,col_1+1)
            
            im_template=im_source[np.ix_(id_row,id_col)]
            
        else:
            # even
            raise ValueError('pyopt: width must be a odd number!')

        return im_template, id_row, id_col
    
    def FindPosition(self,im_template=None, im_ref=None):
       
        im_template -= im_template.mean()
        corr = sp.signal.correlate2d(im_ref, im_template, boundary='symm', mode='same')
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        
        return x, y
    
    def FindPositionLimit(self,im_ref=None,im_source=None,center_row=None,center_col=None,width=None,width_search=None):
       
        if width_search <= width:
            raise ValueError('pyopt: width_search must be larger than width')
        else:
        
            im_new_temp,no_,no_ = self.CropImage(im_source,center_row,center_col,width)
            im_new_ref, id_row, id_col = self.CropImage(im_ref,center_row,center_col,width_search)
            
            im_new_temp -= im_new_temp.mean()
            corr = sp.signal.correlate2d(im_new_ref, im_new_temp, boundary='symm', mode='same')
            y_loc, x_loc = np.unravel_index(np.argmax(corr), corr.shape)
            
            x_global = id_col[x_loc]
            y_global = id_row[y_loc]
        
        return x_global, y_global, im_new_ref

