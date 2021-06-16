import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

file= 'D:\\MLproject_data\\data\\ribfrac-test-images\\RibFrac501-image.nii.gz'
img = nib.load(file)
print(img)
print("point---------------------")
print(img.header['db_name'])
print("point---------------------")
print(type(img))
print("point---------------------")
width, height, queue = img.dataobj.shape
print(width,height,queue)
print(type(width))
#OrthoSlicer3D(img.dataobj).show()
print("point---------------------")
img_arr = img.get_data()
print(type(img_arr))
print(img_arr.shape)
print("point---------------------")
num = 1
for i in range(0, queue, 10):
	img_arr = img.dataobj[:,:,i]
	plt.subplot(5, 6, num)
	plt.imshow(img_arr, cmap='gray')
	num += 1

plt.show()



