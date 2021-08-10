from pydicom import dcmread
import numpy as np
import matplotlib.pyplot as plt

path = "./PAT001/D0003.dcm"
print(path)

x = dcmread(path)
print(x.ImagePositionPatient)

dcom_file = x
plt.imshow(dcom_file.pixel_array, cmap = plt.cm.gray)
plt.show()

