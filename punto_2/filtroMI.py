import numpy as np
import matplotlib.pyplot as plt
print("Ingrese el nombre de la imagen a filtrar")
imagen = input()
img = plt.imread(imagen)
img = img[:,:,:3]
X, Y = img.shape[:2]
ksize = 3
print("Ingrese alto si desea filtrar frecuencias altas, ingrese bajo si desea filtrar frecuencias bajas")
tipo = input()

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

def lowpassfilter(size):
	t = np.linspace(-10, 10, 20)
	gauss = np.exp(-1*(1/2*0.5**2)*t**2)
	gauss /= np.sum(gauss) # Se normaliza el kernel
	kernel = gauss[:, np.newaxis] * gauss[np.newaxis, :]
	gaussiana = np.zeros((X,Y))
	gaussiana[:kernel.shape[0],:kernel.shape[1]] = kernel
	return gaussiana


def highpassfilter(size):
	kernel = np.zeros((size,size))
	for i in range(len(kernel)):
		for j in range(len(kernel[0])):
			if i+j == size/size or i+j == size:
				kernel[i,j] = -1/4
			elif i+j == ((size-1)/2)*2:
				kernel[i,j] = 2	
	padding = np.zeros((X,Y))
	padding[:kernel.shape[0],:kernel.shape[1]] = kernel				
	return padding

# Transformada de fourier del kernel.
if tipo == "alto":
	kernel_ft = bifourier(highpassfilter(ksize)) #Usando mi propia implementación.
elif tipo == "bajo":
	kernel_ft = bifourier(lowpassfilter(ksize)) #Usando mi propia implementación.


# Fourier de la imagen
img_ft1 = bifourier(img[:,:,0]) #Usando mi implementación. MM
img_ft2 = bifourier(img[:,:,1])
img_ft3 = bifourier(img[:,:,2])
img_ft = np.zeros((X,Y,3), dtype = complex)
for i in range(X):
	for j in range(Y):
		img_ft[i,j,0] = img_ft1[i,j]
		img_ft[i,j,1] = img_ft2[i,j]
		img_ft[i,j,2] = img_ft3[i,j]
	
# Se realiza la convolución. Se acomoda para que las dimensiones del kernel concuerden con la tercera dimension de la imagen (espectro de colores)
img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
img21 = invbifourier(img2_ft[:,:,0]).real #Usando mi implementación. MM
img22 = invbifourier(img2_ft[:,:,1]).real
img23 = invbifourier(img2_ft[:,:,2]).real
img2 = np.zeros((X,Y,3), dtype = float)
for i in range(X):
	for j in range(Y):
		img2[i,j,0] = img21[i,j]
		img2[i,j,1] = img22[i,j]
		img2[i,j,2] = img23[i,j]

# Se acotan los valores al rango esperado
img2 = np.clip(img2, 0, 1)

# Se hace una gráfica de lo obtenido
plt.figure()
plt.imshow(img2)
plt.axis('off')
plt.grid(False)
if tipo == "alto":
	plt.savefig("altas.png")
elif tipo == "bajo":
	plt.savefig("bajas.png")


