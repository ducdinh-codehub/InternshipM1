import matplotlib.pyplot as plt
from pydicom import dcmread, read_file
from os import listdir
import numpy as np
from numpy import asarray
path = "./PAT001/"

def load_scan(path):
    slices = [read_file(path + s) for s in listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]) 
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness
    
    return slices

if __name__ == "__main__":
    ctImage = load_scan(path)
    ctImage = asarray(ctImage)

    plt.imshow(ctImage[8].pixel_array,cmap=plt.cm.gray)
    plt.show()