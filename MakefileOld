# Makefile para RefItera (CUDA + CPU BLAS)
NVCC = nvcc
LDFLAGS = -lcusolver -lcublas -llapack -lblas -lm
COMMON_FLAGS = -std=c++17 -Xcompiler -Wall
DEBUG_FLAGS = -g -G -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG

TARGET_DEBUG = RefItera_debug
TARGET_RELEASE = RefItera_release

# Agregamos RefIteraBL a los objetos
OBJS_DEBUG = main.dbg.o cuRefItera.dbg.o RefIteraBL.dbg.o
OBJS_RELEASE = main.rel.o cuRefItera.rel.o RefIteraBL.rel.o

all: release debug

release: $(TARGET_RELEASE)
debug: $(TARGET_DEBUG)

$(TARGET_DEBUG): $(OBJS_DEBUG)
	$(NVCC) $(OBJS_DEBUG) -o $(TARGET_DEBUG) $(LDFLAGS)

$(TARGET_RELEASE): $(OBJS_RELEASE)
	$(NVCC) $(OBJS_RELEASE) -o $(TARGET_RELEASE) $(LDFLAGS)

# --- Compilación ---

# Main
main.dbg.o: main.c RefItera.h
	$(NVCC) -c main.c -o main.dbg.o $(COMMON_FLAGS) $(DEBUG_FLAGS)
main.rel.o: main.c RefItera.h
	$(NVCC) -c main.c -o main.rel.o $(COMMON_FLAGS) $(RELEASE_FLAGS)

# GPU (CUDA)
cuRefItera.dbg.o: cuRefItera.cu RefItera.h
	$(NVCC) -c cuRefItera.cu -o cuRefItera.dbg.o $(COMMON_FLAGS) $(DEBUG_FLAGS)
cuRefItera.rel.o: cuRefItera.cu RefItera.h
	$(NVCC) -c cuRefItera.cu -o cuRefItera.rel.o $(COMMON_FLAGS) $(RELEASE_FLAGS)

# CPU (C Standard) - Nota: nvcc compila .c usando gcc internamente
RefIteraBL.dbg.o: RefIteraBL.c RefItera.h
	$(NVCC) -c RefIteraBL.c -o RefIteraBL.dbg.o $(COMMON_FLAGS) $(DEBUG_FLAGS)
RefIteraBL.rel.o: RefIteraBL.c RefItera.h
	$(NVCC) -c RefIteraBL.c -o RefIteraBL.rel.o $(COMMON_FLAGS) $(RELEASE_FLAGS)

clean:
	rm -f *.o $(TARGET_DEBUG) $(TARGET_RELEASE)