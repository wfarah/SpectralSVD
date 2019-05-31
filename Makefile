CC = gcc
#CFLAGS = -O2 -march=native -ftree-vectorize -lm -Wall
INTEL_LIBS = -lmkl_core -lmkl_intel_lp64 -lmkl_gnu_thread -lpthread -lgomp
CUDA_LIBS = -lcudart
INTEL_INCLUDE = -I/apps/skylake/software/mpi/intel/2018.1.163-gcc-6.4.0/openmpi/3.0.0/imkl/2018.1.163/mkl/include
#LAPACK_DIR = /apps/skylake/software/mpi/intel/2018.1.163-gcc-6.4.0/openmpi/3.0.0/lapack/3.8.0/lib
#LAPACK_LIBS = -L$(LAPACK_DIR) -lcblas -llapacke -llapack -lrefblas -lgfortran -ltmglib 
LAPACK = -L/apps/skylake/software/mpi/intel/2018.1.163-gcc-6.4.0/openmpi/3.0.0/lapack/3.8.0/lib/ -lcblas -llapacke -llapack -lrefblas -lgfortran -ltmglib

#PSRDADA = -L/fred/oz002/psrhome/linux_64/lib -lpsrdada
PSRDADA = -L$(PSRHOME_PSRDADA_PATH)/lib -lpsrdada
#DADA_INCLUDE = -I $(PSRHOME)/software/psrdada/76e4c67/include/
DADA_INCLUDE = -I $(PSRHOME_PSRDADA_PATH)/include

OPENMP = -fopenmp
DEBUG = -ggdb3

all: clean dbsvddb

dbsvddb:
	$(CC) $(CFLAGS) $(DADA_INCLUDE) $(INTEL_INCLUDE) $(OPENMP) dbsvddb.c -o dbsvddb -lm $(INTEL_LIBS) $(PSRDADA) $(CUDA_LIBS) 

clean:
	touch dbsvddb; rm dbsvddb
