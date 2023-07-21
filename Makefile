CC=gcc-10
CFLAGS=-fopenmp -O3 
mpCC=mpicc
MCFLAGS=-O3 -fopenmp


all : OpenMP OpenMP_Tasks1 OpenMP_Tasks2 OpenMP_Tasks3 MPI1 MPI2 OpenMP_MPI
.PHONY : all

OpenMP: OpenMP.c
	$(CC) $(CFLAGS) -o OpenMP_hooke OpenMP.c

OpenMP_Tasks1 : OpenMP_Tasks1.c
	$(CC) $(CFLAGS)  -o OpenMP_Tasks1_hooke OpenMP_Tasks1.c

OpenMP_Tasks2 : OpenMP_Tasks2.c
	$(CC) $(CFLAGS) -o OpenMP_Tasks2_hooke OpenMP_Tasks2.c 

OpenMP_Tasks3 : OpenMP_Tasks3.c
	$(CC) $(CFLAGS) -o OpenMP_Tasks3_hooke OpenMP_Tasks3.c 

MPI1: MPI1.c
	$(mpCC) $(MCFLAGS) -o MPI1_hooke MPI1.c

MPI2: MPI2.c
	$(mpCC) $(MCFLAGS) -o MPI2_hooke MPI2.c

OpenMP_MPI: OpenMP_MPI.c
	$(mpCC) $(MCFLAGS) -o OpenMP_MPI_hooke OpenMP_MPI.c

clean: 
	rm -f  OpenMP_hooke OpenMP_Tasks1_hooke OpenMP_Tasks2_hooke OpenMP_Tasks3_hooke MPI1_hooke MPI2_hooke OpenMP_MPI_hooke