import numpy as np
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
total_procesos = comm.Get_size()

def multiplicar_matriz (A,B):
    C = np.empty((A.shape[0],B.shape[1]))
    filasA = A.shape[0]
    columnasB = B.shape[1]
    columnasA = A.shape[1]
                
    columnas_proceso = columnasA // (total_procesos-1)
    if (rank == (total_procesos-1)) and (columnasA % (total_procesos-1) != 0):
        columnas_proceso += 1
    #print (columnas_proceso)

    for i in range (filasA):
        for j in range (columnasB):

            suma = 0

            if rank == 0:

                proceso = 1
                for col in range(0,columnasA,columnas_proceso):
                    #print ("proceso " + str(proceso),flush=True)
                    #print ("col" + str(col),flush=True)
                    if (proceso == (total_procesos)) and (columnasA % (total_procesos-1) != 0):
                        break
                    else:
                        comm.send(col,dest=proceso)
                    proceso += 1

                for proc in range(1,total_procesos):
                    suma += comm.recv(None,source=proc)

                C[i,j] = suma
            
            else:
                col = comm.recv(None,source=0)
                print (f'proceso {rank} col {col}',flush=True)
                for k in range(col,col+columnas_proceso):
                    #print (str(A[i,k]) + " * " + str(B[k,j]))
                    suma += A[i,k]*B[k,j]
                    #print(str(suma))
                
                comm.send(suma,dest=0)

    return C

matrix = np.random.randint(0, 10, size=(4, 4), dtype=np.int) # matriz n*m
matrix2 = np.random.randint(0, 10, size=(4, 1), dtype=np.int) # matriz n*m

matrix = comm.bcast(matrix)
matrix2 = comm.bcast(matrix2)

C = multiplicar_matriz(matrix,matrix2)

if rank == 0:
    print (matrix)
    print (matrix2)
    print (C)
    

