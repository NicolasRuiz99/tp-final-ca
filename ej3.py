from mpi4py import MPI

n = 2

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
total = comm.Get_size()

if rank == 0:
    print ("Ingrese el mensaje a enviar: ")
    msj = input()

for i in range(n):
    if rank == 0:
        req = comm.irecv(None,source=(total-1))
        comm.send(msj,dest=1)   
        print ("El proceso " + str(rank) + " envia el M" + str(rank) + " con el dato '" + str(msj) + "' al proceso 1",flush=True)
        msj = req.wait()
    else:
        destino = rank+1
        if (destino >= (total)):
            destino = 0
        msj = comm.recv(None,source=rank-1)
        comm.send(msj,dest=destino)
        print ("El proceso " + str(rank) + " envia el M" + str(rank) + " con el dato '" + str(msj) + "' al proceso " + str(destino),flush=True)