#mpiexec -n 4 python ej4_b1.py 6 (tamanio matriz)
import random
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
procesos = comm.Get_size()-1

tamanio_matriz = int (sys.argv[1])
filas_proceso = tamanio_matriz // procesos
matriz = None
vector = None
parte_resultado = None

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
    ultima_fila = 0
    for i in range (procesos):
        filas = []
        for j in range (filas_proceso):
            filas.append(matriz[ultima_fila])
            ultima_fila += 1
        comm.send(filas,i+1)

vector = comm.bcast(vector,0)

if rank != 0:
    parte_resultado = []
    filas = comm.recv()
    for fila in filas:
        parte_resultado.append (mult_fila_vector(fila,vector))

vector_resultado = comm.gather(parte_resultado)

if rank == 0:
    vector_resultado = vector_resultado[1:]
    vector_final = []
    for i in vector_resultado:
        for j in i:
            vector_final.append(j)
    print ("VECTOR RESULTADO:")
    print (vector_final)
