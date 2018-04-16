import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
print("Ingrese el nombre de la imagen a suavizar")
imagen = input()
img = plt.imread(imagen)
img = img[:,:,:3]
#img = img[:30,:30,:] #Descomente esta linea en caso de querer utilizar los metodos implementados por mi sin demoras al ejecutar.
X, Y = img.shape[:2]
pi2 = np.pi * 2
print("Ingrese el ancho de la gaussiana de suavizado medida en pixeles")
sigma = float(input())

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

#Creación del Kernel gaussiano
def gaussian(sig):
	t = np.linspace(-10, 10, 20)
	bump = np.exp(-0.1*t**2)
	bump /= np.trapz(bump) # normalize the integral to 1
	kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
	result = np.zeros((X,Y))
	result[:kernel.shape[0],:kernel.shape[1]] = kernel
	return result

# Transformada de fourier del kernel, que resulta ser una gaussiana
#kernel_ft = fftpack.fft2(gaussian(Y, X, sigma), shape=img.shape[:2], axes=(0, 1)) # Usando el metodo nativvo de python TT
kernel_ft = fftpack.fft2(gaussian(sigma), axes=(0, 1))
#kernel_ft = bifourier(gaussian(sigma)) #Usando mi propia implementación. En caso de querer utilizarla, comentar la linea en donde se utiliza el metodo nativo de python, marcadas con un TT,  y descomentar las que utilizan los metodos implementados por mi, marcadas con MM. ATENCION!: No la utilizo porque simplemente nunca se termina de ejecutar cuando lo hago. Se sugiere ejecutar el archivo FourierPROPIA.py para comprobar que los metodos nativos de python y el implementado por mi arroja el mismo resultado. Se dejara la alternativa de utilizar una menor cantidad de datos de una imagen especifica para poder probar esto. Para ello, mire el inicio del codigo. MM

# convolución
img_ft = fftpack.fft2(img, axes=(0, 1)) #Usando el metodo nativo. TT
"""img_ft1 = bifourier(img[:,:,0]) #Usando mi implementación. MM
img_ft2 = bifourier(img[:,:,1])
img_ft3 = bifourier(img[:,:,2])
img_ft = np.zeros((X,Y,3), dtype = complex)
for i in range(X):
	for j in range(Y):
		img_ft[i,j,0] = img_ft1[i,j]
		img_ft[i,j,1] = img_ft2[i,j]
		img_ft[i,j,2] = img_ft3[i,j]"""
	
# Se acomoda para que las dimensiones del kernel concuerden con la tercera dimension de la imagen (espectro de colores)
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real #Metodo nativo. TT
"""img21 = invbifourier(img2_ft[:,:,0]).real #Usando mi implementación. MM
img22 = invbifourier(img2_ft[:,:,1]).real
img23 = invbifourier(img2_ft[:,:,2]).real
img2 = np.zeros((X,Y,3), dtype = float)
for i in range(X):
	for j in range(Y):
		img2[i,j,0] = img21[i,j]
		img2[i,j,1] = img22[i,j]
		img2[i,j,2] = img23[i,j]"""
# Se acotan los valores al rango esperado
img2 = np.clip(img2, 0, 1)

# Se hace una gráfica de lo obtenido
plt.figure()
plt.imshow(img2)
plt.axis('off')
plt.grid(False)

plt.savefig("suave.png")

