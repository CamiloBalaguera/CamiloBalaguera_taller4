import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
pi2 = np.pi * 2.0
muestra = np.array(([2,3,4,5], [2,3,4,5], [2,3,4,5], [2,3,4,5]))
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

def invbifourier(array):
    (M, N) = array.shape 
    imagen = np.zeros((M,N), dtype = complex)
    for m in range(M):
        for n in range(N):
            suma = 0.0
            for k in range(M):
                for l in range(N):
                    t = np.exp(1j * pi2 * ((k * m) / M + (l * n) / N))
                    suma += array[l][k] * t
            val = suma / (M*N)
            imagen[m, n] = val
    return np.transpose(imagen)
      


print("Implementacion propia","\n" ,bifourier(muestra),"\n", "Metodo nativo de python", "\n",  fftpack.fft2(muestra))
print("Implementacion propia inversa","\n" ,invbifourier(muestra),"\n", "Metodo nativo de python", "\n",  fftpack.ifft2(muestra))

