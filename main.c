#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Necesario para memcpy
#include "RefItera.h"

int main() 
{
    int N = 10;
    
    // Matriz "Tramposa"
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

    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));

    // Preparar A (Column-Major) y b
    for(int i = 0; i < N; i++) {
        b[i] = 0.0;
        for(int j = 0; j < N; j++) {
            A[j*N + i] = A_raw[i*N + j];
            b[i] += A[j*N + i]; // b es la suma de la fila
        }
    }

    // ---------------------------------------------------------
    // RONDA 1: CPU (BLAS/LAPACK)
    // ---------------------------------------------------------
    // Inicializamos x con b (primer guess)
    memcpy(x, b, N * sizeof(double)); 
    
    cpu_refinement(N, A, b, x, 20);

    printf("Resultado CPU (primeros 3): %.15f %.15f %.15f\n", x[0], x[1], x[2]);

    // ---------------------------------------------------------
    // RONDA 2: GPU (CUDA/cuBLAS)
    // ---------------------------------------------------------
    printf("\n--------------------------------------------------\n");
    printf("Reiniciando x para la prueba en GPU...\n");
    
    // IMPORTANTE: Reiniciar x "sucio" para que la GPU trabaje desde cero
    memcpy(x, b, N * sizeof(double)); 

    gpu_refinement(N, A, b, x, 20);

    printf("Resultado GPU (primeros 3): %.15f %.15f %.15f\n", x[0], x[1], x[2]);

    // Limpieza
    free(A); free(b); free(x);
    return 0;
}
