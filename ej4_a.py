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
