import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import logging
import time

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger()

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
  zj_cj = np.round(np.matmul(c_p[base], np.matmul(B_1, A)) - c_p, 10)
  # esto de antes se podria mejorar calculando sólo para las VNB

  # Mientras pueda mejorar
  while any(x < 0 for x in zj_cj):
    # Determinar Variable Entrante
    ve = np.argmin(zj_cj)
    log.info("Nueva variable entrante: x_{}".format(ve))
    # Vector correspondiente a la variable entrante
    A_ve = np.matmul(B_1, A[:, ve])
    # Calculamos los cocientes tita
    b_p = np.matmul(B_1, b)
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
    
    B_1 = np.matmul(E_1, B_1)
    zj_cj = np.round(np.matmul(c_p[base], np.matmul(B_1, A)) - c_p, 10)
  # Cuando ya no puede mejorar
  b_p = np.matmul(B_1, b)
  x_opt = {base[j]: b_p[j] for j in range(m)}
  z_opt = np.matmul(c_p[base], b_p)
  return x_opt, z_opt, B_1


np.random.seed(12345)
num_variables = 30
num_restricciones = 50
A = [np.random.rand(num_variables) for j in range(num_restricciones)]
c = np.random.rand(num_variables)
b = np.random.rand(num_restricciones)

start_time = time.time()

base, c_p, A, b = simplex_init(c, lessThans=A, ltThreshold=b, maximization=True, M=10.)
#base, c_p, A, b = simplex_init([300., 250., 450.], greaterThans=[[0., 250., 0.]], gtThreshold=[500.], lessThans=[[15., 20., 25.], [35., 60., 60.], [20., 30., 25.]], ltThreshold=[1200., 3000., 1500.], maximization=True, M=1000.)
x_opt, z_opt, _ = solve_linear_program(base, c_p, A, b)

tiempo = (time.time() - start_time)

print("La solución es:")
for j in x_opt:
  print("x_{} = {}".format(j, x_opt[j]))
print("Esto produce un funcional de z = {}".format(z_opt))

print("--- %s seconds ---" % tiempo)