import cv2
import numpy as np
from skimage.segmentation import slic,find_boundaries
from sklearn.preprocessing import normalize
import skimage
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation,DBSCAN
from scipy.ndimage.measurements import center_of_mass
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D

def segment_whole_image(img):
    rgb_img = img[:, :, ::-1]
    lab_img = skimage.color.rgb2lab(rgb_img)
    sp_img = slic(lab_img, n_segments=600,convert2lab=False)
    label, label_count = np.unique(sp_img, return_counts = True)
    sp3d_img = np.expand_dims(sp_img, axis=2)
    sp3d_img = np.repeat(sp3d_img, 3, axis=2)
    mean_vectors = np.zeros((len(label),5))
    
    # +1 to remove errors with 0
    com = center_of_mass(sp_img+1,labels=sp_img+1,index=label+1)

    for l in label:
        mean_vectors[l,0:3]=np.mean(lab_img[sp_img==l],0)
        mean_vectors[l,3:5]=list(com[l])

    DB = DBSCAN(eps=0.0125,min_samples=10)
    y = DB.fit_predict(normalize(mean_vectors,axis=0))



    seg_img = np.copy(sp_img)
    for l,newl in enumerate(y):
        seg_img[seg_img==l] = newl
        

  
    outliers = (seg_img==-1).astype(int)
    fixed_outliers = scipy.ndimage.label(outliers)[0]
    seg_img[seg_img==-1] = fixed_outliers[fixed_outliers!=0]+np.max(seg_img)
    return seg_img

    

    



def build_mask(seg_img, points):
    x,y = points
    mask = np.where(seg_img==seg_img[x,y],np.ones(seg_img.shape),0)
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    return mask

if __name__ == '__main__':

    # for testing
    img = cv2.imread('24829281_10214644306009667_448251397_n.jpg')
    points = set()
    points.add((215, 150))
    points.add((215, 400))
    mask = build_mask(img, points)

    cv2.imwrite('test.jpg', mask * (255//3))