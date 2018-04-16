import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

img = plt.imread("uniandes.png")
img = img[:,:,:3]
X, Y = img.shape[:2]
pi2 = np.pi * 2

def bifourier(array):
    (M, N) = array.shape 
    fourier = np.zeros((M,N), dtype = complex)
    for k in range(M):
        for l in range(N):
            suma = 0.0
            for m in range(M):
                for n in range(N):
                    t = np.exp(- 1j * pi2 * ((k * m) / M + (l * n) / N))
                    suma += array[m,n] * t
            fourier[l][k] =  suma
    return np.transpose(fourier)

def gaussian(kernlen1, kernlen2, sig):
	"""creates gaussian kernel with side length l and a sigma of sig"""
	ax = np.arange(-kernlen1 // 2 + 1., kernlen1 // 2 + 1.)
	bx = np.arange(-kernlen2 // 2 + 1., kernlen2 // 2 + 1.)
	x, y = np.meshgrid(ax, bx)
	kernel = np.exp(-(x**2 + y**2) / (2. * sig**2))
	return kernel / np.sum(kernel)

# Transformada de fourier del kernel, que resulta ser una gaussiana
kernel_ft = fftpack.fft2(gaussian(5, 5, 1), axes=(0, 1)) # Usando el metodo nativvo de python
#kernel_ft = bifourier(gaussian(5,5,1)) #Usando mi propia implementación. En caso de querer utilizarla, comentar la linea de arriba y descomentar esta. ATENCION!: No la utilizo porque simplemente nunca se termina de ejecutar cuando lo hago

"""
# convolve
img_ft = fftpack.fft2(img, axes=(0, 1))
# the 'newaxis' is to match to color direction
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

# clip values to range
img2 = np.clip(img2, 0, 1)

# plot output
plt.figure()
plt.imshow(img2)"""

