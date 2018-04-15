import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

img = plt.imread("uniandes.png")
img = img[:,:,:3]

def gaussian(kernlen1, kernlen2, sig):
	"""creates gaussian kernel with side length l and a sigma of sig"""
	ax = np.arange(-kernlen1 // 2 + 1., kernlen1 // 2 + 1.)
	bx = np.arange(-kernlen2 // 2 + 1., kernlen2 // 2 + 1.)
	x, y = np.meshgrid(ax, bx)
	kernel = np.exp(-(x**2 + y**2) / (2. * sig**2))
	return kernel / np.sum(kernel)

# Padded fourier transform, with the same shape as the image
# We use :func:`scipy.signal.fftpack.fft2` to have a 2D FFT
kernel_ft = fftpack.fft2(gaussian(30, 30, 5), shape=img.shape[:2], axes=(0, 1))
print(kernel_ft)

# convolve
img_ft = fftpack.fft2(img, axes=(0, 1))
# the 'newaxis' is to match to color direction
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

# clip values to range
img2 = np.clip(img2, 0, 1)

# plot output
plt.figure()
plt.imshow(img2)
plt.show()
