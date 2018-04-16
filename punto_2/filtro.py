import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

#Implementación propia de la transformada de fourier bidimensional
def bifourier(array):
	(M, N) = array.shape[:2]
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

#Implementación propia de la inversa de la transformada de fourier bidimensional
def invbifourier(array):
	(M, N) = array.shape[:2]
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
