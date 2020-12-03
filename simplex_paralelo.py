import numpy as np
import logging
import time
from mpi4py import MPI
import sys

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger()

def multiplicar_matriz (A,B):
    
    if (len(A.shape) == 1):
      A = np.atleast_2d (A)
    if (len(B.shape) == 1):
      B = B.reshape(-1,1)

    C = np.empty((A.shape[0],B.shape[1]))
    filasA = A.shape[0]
    columnasB = B.shape[1]
    columnasA = A.shape[1]
                
    columnas_proceso = columnasA // (total_procesos-1)
    #print(str(columnas_proceso))
    if (rank == (total_procesos-1)) and (columnasA % (total_procesos-1) != 0):
      columnas_proceso += 1

    for i in range (filasA):
        for j in range (columnasB):

            suma = 0

            if rank == 0:

                proceso = 1
                for col in range(0,columnasA,columnas_proceso):

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
                #print (f'proceso {rank} col {col}',flush=True)
                for k in range(col,col+columnas_proceso):
                    #print (str(A[i,k]) + " * " + str(B[k,j]))
                    #print(f'proceso {rank}, columnas {columnas_proceso}, k: {k}',flush=True)
                    suma += A[i,k]*B[k,j]
                    #print(f'suma: {suma}',flush=True)
                """
                  A[:,ve][ 20.  60.  30. 250.]
                  B_1[[1. 0. 0. 0.]
                  [0. 1. 0. 0.]
                  [0. 0. 1. 0.]
                  [0. 0. 0. 1.]]

                  20 / 60 / 30 
                  """

                
                comm.send(suma,dest=0)

    C = comm.bcast(C,0)

    if (C.shape[0] == 1 or C.shape[1] == 1):
        C = C.flatten()

    return C

def simplex_init(c, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], 
                         equalities=[], eqThreshold=[], maximization=True, M=1000.):
  '''
    Construye la primera base y los parámetros extendidos A' y c'. El orden en 
    el que quedan las restricciones es
    1) <=
    2) >=
    3) =

    En el mismo orden se agregan las variables 
    1) slack de <=
    2) exceso y artificial de >=
    3) artificial de =

    Parameters:
    c: Vector de coeficientes del funcional de las variables de decisión
    greaterThans: lista de listas con las filas de las restricciones de mayor o igual
    gtThreshold: lista de los lados derechos de las restricciones de mayor o igual
    lessThans: lista de listas con las filas de las restricciones de menor o igual
    ltThreshold: lista de los lados derechos de las restricciones de menor o igual
    equalities: lista de listas con las filas de las restricciones de igual
    eqThreshold: lista de los lados derechos de las restricciones de igual
    maximization: true si el problema es de maximización, false si el problema es de minimización
    M: penalización que se le asignará a las variables artificiales en caso de ser necesarias

    Returns:
    base: lista de indices correspondientes a las variables básicas de la base inicial
    c_p: vector c extendido con las variables slack, de exceso y artificiales
    A: matriz extendida
    b: vector de lados derechos de las restricciones
  '''
  # Inicialización
  cant_gt = len(gtThreshold)
  cant_lt = len(ltThreshold)
  cant_eq = len(eqThreshold)

  cant_por_proceso = cant_lt // (total_procesos-1)
  
  m = cant_gt + cant_lt + cant_eq
  n = len(c)
  n_p = m + n + cant_gt+ cant_eq # contando además las variables artificiales
  c_p = np.zeros(n_p)
  c_p[:n] = c if maximization else -c
  
  base = []
  
  A = np.empty((m, n_p))
  b = np.empty(m)
  if cant_lt > 0:
    A[0:cant_lt,:n] = lessThans
    b[0:cant_lt] = ltThreshold
  if cant_gt > 0:
    A[cant_lt:(cant_lt+cant_gt),:n] = greaterThans
    b[cant_lt:(cant_lt+cant_gt)] = gtThreshold
  if cant_eq > 0:
    A[(cant_lt+cant_gt):(cant_lt+cant_gt+cant_eq),:n] = equations
    b[(cant_lt+cant_gt):(cant_lt+cant_gt+cant_eq)] = eqThreshold

  
  if rank == 0:

    for i in range(cant_lt):
      A[:, n+i] = [(1. if i == j else 0) for j in range(m)]
      base.append(n+i)

    for i in range(cant_lt, cant_lt + 2*cant_gt, 2):
      A[:, n+i] = [(-1. if i == j else 0) for j in range(m)]
      A[:, n+i+1] = [(1. if i == j else 0) for j in range(m)]
      c_p[n+i+1] = -M
      base.append(n+i+1)
  

    for i in range(cant_lt + 2*cant_gt, cant_lt + 2*cant_gt + cant_eq):
      A[:, n+i] = [(1. if i == j else 0) for j in range(m)]
      c_p[n+i] = -M
      base.append(n+i)

    """
    primer for paralelo
    

    proceso = 1
    for i in range(0,cant_lt,cant_por_proceso):
      comm.send(i,dest=proceso)
      proceso += 1
    
    # Concatenar los arrays soluciones
    for i in range(1,total_procesos):
      resultados = comm.recv(None,source=i)
      for tupla in resultados:
        pos = tupla[1]
        A[:,n+pos] = tupla[0]
        base.append(n+pos)

    """

    

  #else:
    """
    i = comm.recv(None,source=0)
    resultados = []
    #if (rank == (total_procesos-1)) and (cant_lt % (total_procesos-1) != 0):
    #  cant_por_proceso += 1
    for x in range(i,i+cant_por_proceso):
      resultados.append(([(1. if x == j else 0) for j in range(m)],x))  

    comm.send(resultados,dest=0)  
    """

  base = comm.bcast(base)
  c_p = comm.bcast(c_p)
  A = comm.bcast(A)
  b = comm.bcast(b)

  return base, c_p, A, b


def solve_linear_program(base, c_p, A, b):
  '''
  Resuelve el programa lineal dado por los parámetros c_p, A y b, partiendo de 
  la base inicial "base". La primer matriz básica resultante debe ser la matriz 
  identidad, lo cual se asume como precondición.

  Parameters:
  base: base inicial (lista de índices)
  c_p: vector de coeficientes del funcional
  A: matriz de coeficientes tecnológicos
  b: vector de lados derechos (se asumen todos sus elementos >= 0)

  Returns:
  x_opt: diccionario de variables básicas y sus respectivos valores
  z_opt: valor óptimo de la función objetivo
  B_1: matriz inversa de la base óptima (sirve para construir la tabla óptima 
  y hallar los precios sombra y costos reducidos, de ser necesario)
  
  '''

  #B = A[:, base]
  m = len(b)
  B_1 = np.eye(m)

  # Iteración del método

  zj_cj = np.round(multiplicar_matriz(c_p[base],multiplicar_matriz(B_1,A)) - c_p, 10)
  
  # esto de antes se podria mejorar calculando sólo para las VNB

  # Mientras pueda mejorar
  while any(x < 0 for x in zj_cj):
    # Determinar Variable Entrante
    ve = np.argmin(zj_cj)
    log.info("Nueva variable entrante: x_{}".format(ve))
    # Vector correspondiente a la variable entrante

    A_ve = multiplicar_matriz(B_1,A[:,ve])

    # Calculamos los cocientes tita


    b_p = multiplicar_matriz(B_1,b)

    titas = [(b_p[i]/A_ve[i] if A_ve[i] > 0 else np.nan) for i in range(m)]
    
    if all(np.isnan(tita) for tita in titas):
      log.info("Problema no acotado")
      raise("Problema no acotado")
    # Determinar Variable Saliente
    vs = np.nanargmin(titas)
    log.info("Nueva variable saliente: x_{}".format(base[vs]))
    base[vs] = ve
    log.info("Nueva base: {}".format(base))
    # Actualizar matriz inversa B_1
    E = np.eye(m)
    E[:,vs] = A_ve
    E_1 = np.eye(m)
    E_1[:,vs] = [(-E[i, vs]/E[vs, vs] if i != vs else 1./E[vs, vs]) for i in range(m)]


    B_1 = multiplicar_matriz(E_1,B_1)


    zj_cj = np.round(multiplicar_matriz(c_p[base], multiplicar_matriz(B_1,A)) - c_p, 10)

  # Cuando ya no puede mejorar
  

  b_p = multiplicar_matriz(B_1,b)

  x_opt = {base[j]: b_p[j] for j in range(m)}
  
  
  
  z_opt = multiplicar_matriz(c_p[base], b_p)

  return x_opt, z_opt, B_1



np.random.seed(12345)
num_variables = 30
num_restricciones = 50
A = [np.random.rand(num_variables) for j in range(num_restricciones)]
c = np.random.rand(num_variables)
b = np.random.rand(num_restricciones)


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
total_procesos = comm.Get_size()

start_time = time.time()

base, c_p, A, b = simplex_init(c, lessThans=A, ltThreshold=b, maximization=True, M=10.)
#base, c_p, A, b = simplex_init([300., 250., 450.], greaterThans=[[0., 250., 0.]], gtThreshold=[500.], lessThans=[[15., 20., 25.], [35., 60., 60.], [20., 30., 25.]], ltThreshold=[1200., 3000., 1500.], maximization=True, M=1000.)

x_opt, z_opt, _ = solve_linear_program(base, c_p, A, b)

tiempo = (time.time() - start_time)

if rank == 0:

  print("La solución es:")
  for j in x_opt:
    print("x_{} = {}".format(j, x_opt[j]))
    
  print("Esto produce un funcional de z = {}".format(z_opt))

  print("--- %s seconds ---" % tiempo)


"""
segundo for que va en el else del rank
"""
"""
cant_por_proceso = comm.recv(None,source=0,tag=3)
m = comm.recv(None,source=0,tag=4)
i = comm.recv(None,source=0,tag=5)
resultados = []
resultados2 = []
for x in range(i,i+cant_por_proceso):
resultados.append(([(-1. if i == j else 0) for j in range(m)],x))  
resultados2.append(([(1. if i == j else 0) for j in range(m)],x))
print(resultados)

comm.send(resultados,dest=0)
comm.send(resultados2,dest=0)
"""



"""
segundo for que va dentro del rank = 0
"""
"""
print("cant_lt:"+ str(cant_lt))
print("cant_lt+2*cant_gt:"+ str(cant_lt + 2*cant_gt))
print("cant_por_proceso:"+ str(cant_por_proceso))

cant_por_proceso = ((cant_lt + 2*cant_gt)-cant_lt) // (total_procesos-1)

proceso = 1
for i in range(cant_lt,cant_lt + 2*cant_gt,cant_por_proceso*2):
  comm.send(cant_por_proceso,dest=proceso,tag=3)
  comm.send(m,dest=proceso,tag=4)
  comm.send(i,dest=proceso,tag=5)
  #print (proceso)
  proceso += 1


for i in range(1,total_procesos):
  resultados = comm.recv(None,source=i)
  resultados2 = comm.recv(None,source=i)
  for tupla in resultados:
    pos = tupla[1]
    A[:,n+pos] = tupla[0]

  for tupla in resultados2:
    pos = tupla[1]
    A[:,n+pos+1]
    c_p[n+pos+1] = -M
    base.append(n+pos+1)
"""
  