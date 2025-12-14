/**
 * main.c
 * 
 * Programa principal que prepara los datos y llama a la función
 * de refinamiento en GPU.
 * 
 * Compilar con: nvcc -c main.c -o main.o
 * Ligar con: nvcc main.o cuRefItera.o -o gpu_refine -lcusolver -lcublas
 * 
 * Miguel Ángel Norzagaray Cosío
 * UABCS/dasc - 20251213
 */

#include <stdio.h>
#include <stdlib.h>
#include "RefItera.h"

int main() 
{
    int N = 10;
    
    // Matriz original (Fila por fila, row-major, como la leemos los humanos)
    double A_raw[] = {
         1,  4,  0, -9,  1, 10, -2, -1,  2, -6,
         9,  8,  3,  2,  2, 10, -7,  1, 10, -4,
         3, -6, -5, -5,  6,  3, -3,  9,  8,  1,
        -7,  5, -5,  8,  9,  0, -5, -1,  5,  3,
        -1, -9, -2,  3, -7,  8,  4, -6, -8, 20,
         8,  8, -5,  4,  7,  1,  2, -9, -5,  9,
         3, -7,  6,  3, -7, -9,  1, -1,  1,  7,
        -5, -3,  0,  0,  8,  0,  3,  9,  0,  5,
        -5, 10, -5, -5,  7,  7, -4,  4,  3,  7,
        -3,  9,  2, -1, -1, -6, -7, -8, -3,  0
    };

    // Reservar memoria en Host
    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));

    // Transformar a Column-Major (necesario para cuBLAS/LAPACK) y calcular b
    printf("Profe, preparando datos en Host...\n");
    for(int i = 0; i < N; i++) {
        b[i] = 0.0;
        for(int j = 0; j < N; j++) {
            // A_raw es Row-Major: [i*N + j]
            // cuda requiere que A sea Column-Major: [j*N + i]
            A[j*N + i] = A_raw[i*N + j];
            b[i] += A[j*N + i]; // La solución exacta debería ser todo 1s
        }
    }

    // Llamada a la función CUDA
    printf("Lanzando kernel de refinamiento a la GPU...\n\n");
    //gpu_refinement(N, A, b, x, 20);

    cpu_refinement(N, A, b, x, 20);

    // Mostrar resultados
    puts("\nResultado Final en CPU:");
    for(int i = 0; i < 10; i++) 
        printf("%.16f\n", x[i]);

    free(A); free(b); free(x);
    return 0;
}
