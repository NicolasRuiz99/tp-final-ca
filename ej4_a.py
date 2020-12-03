import random
from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

tamanio_matriz = size - 1
matriz = None
vector = None
resultado = None

def generar_matriz (tamanio):
    matriz = []
    for i in range (tamanio):
        matriz.append([])
        for j in range (tamanio):
            num = random.randint(0,100)
            matriz[i].append(num)

    return matriz

def generar_vector (tamanio):
    vector = []
    for i in range(tamanio):
        num = random.randint(0,100)
        vector.append(num)
    
    return vector

def mult_fila_vector (fila,vector):
    res = 0
    for i in range (len(fila)):
        res += fila[i] * vector[i]

    return res

def print_matriz(matriz):
    for i in range (len(matriz)):
        fila = matriz[i]
        for j in fila:
            print (str(j)+"  ",end=" ")
        print (" ")

if rank == 0:
    matriz = generar_matriz(tamanio_matriz)
    print_matriz(matriz)
    vector = generar_vector(tamanio_matriz)
    print (vector)
    for i in range (tamanio_matriz):
        comm.send(matriz[i],i+1)

vector = comm.bcast(vector,0)

if rank != 0:
    fila = comm.recv()
    resultado = mult_fila_vector(fila,vector)

vector_resultado = comm.gather(resultado)

if rank == 0:
    vector_resultado = vector_resultado[1:]
    print ("VECTOR RESULTADO:")
    print (vector_resultado)


"""
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

matrix = None # matriz n*n vacía
bufferReceptor = np.zeros((1, size), dtype=np.int) # vector columna vacío
vectorResultOfProduct = np.zeros((size, 1), dtype=np.int) # vector resultante vacío
vector = None # vector columna vacío


if rank == 0:
    matrix = np.random.randint(0, 10, size=(size, size), dtype=np.int) # matriz n*n
    vector = np.random.randint(0, 10, size=(size, 1), dtype=np.int) # vector columna

# repartimos las filas de la matriz entre los procesos participantes
comm.Scatter([matrix, MPI.INT], [bufferReceptor, MPI.INT])

# repartimos el vector a multiplicar entre los procesos participantes
vectorToMultiply = comm.bcast(vector, root)

# realizamos los productos correspondientes
vectorResult = np.matmul(bufferReceptor, vectorToMultiply)

# enviamos el vector resultante de cada producto realizado por los demas procesos, al proceso 0
productResult = comm.gather(vectorResult[0], root)

if rank == 0:
    for i in range(len(productResult)):
        vectorResultOfProduct.put(i, productResult[i].item())

    print(f'Hola soy el proceso {rank}, el producto {matrix} · {vector} = {vectorResultOfProduct}')
"""