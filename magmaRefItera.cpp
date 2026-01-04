#include <stdio.h>
#include <cuda_runtime.h>
#include "magma_v2.h"
#include "RefItera.h"

void magma_refinement(int N, double *h_A, double *h_b, double *h_x, int max_iter) 
{
    printf("\n--- Inicio Refinamiento MAGMA (Híbrido CPU/GPU) ---\n");

    // 1. Inicializar MAGMA
    if (magma_init() != MAGMA_SUCCESS) {
        fprintf(stderr, "Error inicializando MAGMA\n");
        return;
    }
    
    // Crear una cola de ejecución (Queue)
    magma_queue_t queue;
    int dev = 0;
    magma_queue_create(dev, &queue);

    magma_int_t n = N;
    magma_int_t n_rhs = 1;
    magma_int_t info;

    // Punteros Device (GPU)
    double *d_A, *d_LU, *d_b, *d_x, *d_r, *d_z;
    
    // Punteros Host (CPU) - MAGMA requiere IPIV en Host para dgetrf_gpu
    magma_int_t *h_Ipiv;
    magma_imalloc_cpu(&h_Ipiv, N); // Memoria "pinned" en CPU para velocidad

    // Allocación en GPU usando wrappers de MAGMA (alineados)
    magma_dmalloc(&d_A, N * N);
    magma_dmalloc(&d_LU, N * N);
    magma_dmalloc(&d_b, N);
    magma_dmalloc(&d_x, N);
    magma_dmalloc(&d_r, N);
    magma_dmalloc(&d_z, N);

    // Copiar datos Host -> Device
    // magma_setmatrix(rows, cols, elem_size, src_host, lda, dst_dev, ldda, queue)
    magma_setmatrix(N, N, sizeof(double), h_A, N, d_A, N, queue);
    magma_setmatrix(N, N, sizeof(double), h_A, N, d_LU, N, queue); // Copia para factorizar
    magma_setvector(N, sizeof(double), h_b, 1, d_b, 1, queue);
    
    // Inicializar x con b
    magma_dcopy(N, d_b, 1, d_x, 1, queue);

    // --- Paso 1: Factorización LU (Híbrida) ---
    // magma_dgetrf_gpu factoriza en GPU, pero guarda pivotes en CPU (h_Ipiv)
    magma_dgetrf_gpu(n, n, d_LU, n, h_Ipiv, &info);
    
    if (info != 0) 
        printf("Error en MAGMA LU: %d\n", (int)info);

    // --- Paso 2: Solución Inicial ---
    // Resolvemos usando d_LU (GPU) y h_Ipiv (CPU)
    magma_dgetrs_gpu(MagmaNoTrans, n, n_rhs, d_LU, n, h_Ipiv, d_x, n, &info);

    // Constantes
    double alpha_neg = -1.0;
    double alpha_pos = 1.0;
    
    printf("%-5s | %-15s\n", "Iter", "||r|| (MAGMA)");
    printf("--------------------------\n");

    double prev_norm = 1.0e+30;

    // --- Paso 3: Bucle de Refinamiento ---
    for(int k = 0; k < max_iter; k++) {
        // a. Calcular Residuo: r = b
        magma_dcopy(N, d_b, 1, d_r, 1, queue);
        
        // r = -1.0 * A * x + r
        magma_dgemv(MagmaNoTrans, N, N, alpha_neg, d_A, N, d_x, 1, alpha_pos, d_r, 1, queue);

        // b. Norma del residuo
        double curr_norm = magma_dnrm2(N, d_r, 1, queue);

        printf("%-5d | %-1.8e\n", k+1, curr_norm);

        if(curr_norm >= prev_norm && k > 0) {
            printf("STOP: El error dejó de disminuir.\n");
            break;
        }
        prev_norm = curr_norm;

        // c. Calcular Corrección: resolver A*z = r
        magma_dcopy(N, d_r, 1, d_z, 1, queue);
        magma_dgetrs_gpu(MagmaNoTrans, n, n_rhs, d_LU, n, h_Ipiv, d_z, n, &info);

        // d. Actualizar solución: x = x + z
        magma_daxpy(N, alpha_pos, d_z, 1, d_x, 1, queue);
    }

    // Copiar resultado final Host <- Device
    magma_getvector(N, sizeof(double), d_x, 1, h_x, 1, queue);

    // Liberar memoria
    magma_free(d_A); magma_free(d_LU); magma_free(d_b);
    magma_free(d_x); magma_free(d_r); magma_free(d_z);
    magma_free_cpu(h_Ipiv);

    magma_queue_destroy(queue);
    magma_finalize();
}
