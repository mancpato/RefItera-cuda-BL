/**
 * @file cuRefItera.cu
 * @brief Implementación acelerada por GPU usando CUDA (cuBLAS + cuSOLVER).
 *
 * Realiza la factorización LU y el refinamiento iterativo aprovechando
 * el paralelismo masivo y las unidades FMA de la GPU.
 *
 * @author Miguel Ángel Norzagaray Cosío
 * @date 2025-12-13
 *
 * @note Requiere librerías: -lcusolver -lcublas
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "RefItera.h"

// Macro para chequear errores de CUDA (útil para depurar)
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

void gpu_refinement(int N, double *h_A, double *h_b, double *h_x, int max_iter) 
{
    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    
    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    // Punteros en Device
    double *d_A, *d_LU, *d_b, *d_x, *d_r, *d_z;
    int *d_Ipiv, *d_Info;
    int Lwork = 0;
    double *d_Work;

    // Reservar memoria en GPU
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_LU, N * N * sizeof(double))); // Copia para LU
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(double))); // Residuo
    CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(double))); // Corrección
    CUDA_CHECK(cudaMalloc(&d_Ipiv, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Info, sizeof(int)));

    // Copiar datos Host -> Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N*N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_LU, h_A, N*N*sizeof(double), cudaMemcpyHostToDevice)); // LU empieza como copia de A
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N*sizeof(double), cudaMemcpyHostToDevice));
    
    // Inicializar x con b (o ceros, o lo que sea)
    CUDA_CHECK(cudaMemcpy(d_x, d_b, N*sizeof(double), cudaMemcpyDeviceToDevice));

    // --- Paso 1: Factorización LU (dgetrf) ---
    // Calcular tamaño del buffer de trabajo
    cusolverDnDgetrf_bufferSize(cusolverH, N, N, d_LU, N, &Lwork);
    CUDA_CHECK(cudaMalloc(&d_Work, Lwork * sizeof(double)));

    cusolverDnDgetrf(cusolverH, N, N, d_LU, N, d_Work, d_Ipiv, d_Info);

    // --- Paso 2: Solución Inicial (x0) ---
    // Resolvemos A*x = b usando la factorización LU
    // dgetrs sobrescribe d_x con la solución
    cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_LU, N, d_Ipiv, d_x, N, d_Info);

    // Constantes para BLAS
    double alpha_neg = -1.0;
    double alpha_pos = 1.0;
    double beta_pos = 1.0;
    //double beta_zero = 0.0; // Importante para reinicializar vectores

    printf("%-5s | %-15s\n", "Iter", "||r|| (GPU)");
    printf("--------------------------\n");

    double prev_norm = 1.0e+30;

    // --- Paso 3: Bucle de Refinamiento ---
    for(int k = 0; k < max_iter; k++) {
        // a. Calcular Residuo: r = b - A*x
        // Primero hacemos r = b
        CUDA_CHECK(cudaMemcpy(d_r, d_b, N*sizeof(double), cudaMemcpyDeviceToDevice));
        
        // Luego r = -1.0 * A * x + 1.0 * r
        // dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        cublasDgemv(cublasH, CUBLAS_OP_N, N, N, &alpha_neg, d_A, N, d_x, 1, &beta_pos, d_r, 1);

        // b. Norma del residuo
        double curr_norm = 0.0;
        cublasDnrm2(cublasH, N, d_r, 1, &curr_norm);

        printf("%-5d | %-1.8e\n", k+1, curr_norm);

        if(curr_norm >= prev_norm && k > 0) {
            printf("STOP: El error dejó de disminuir.\n");
            break;
        }
        //if(curr_norm < 1e-14) { // Umbral de satisfacción
        //     printf("STOP: Convergencia alcanzada.\n");
        //     break;
        //}
        prev_norm = curr_norm;

        // c. Calcular Corrección: resolver A*z = r
        // Copiamos r en z, porque dgetrs sobrescribe la entrada B con la solución X
        CUDA_CHECK(cudaMemcpy(d_z, d_r, N*sizeof(double), cudaMemcpyDeviceToDevice));
        
        // Resolvemos A*z = r (el resultado queda en d_z)
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_LU, N, d_Ipiv, d_z, N, d_Info);

        // d. Actualizar solución: x = x + z
        // daxpy: y = alpha*x + y -> d_x = 1.0 * d_z + d_x
        cublasDaxpy(cublasH, N, &alpha_pos, d_z, 1, d_x, 1);
    }

    // Copiar resultado final de vuelta al Host
    CUDA_CHECK(cudaMemcpy(h_x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost));

    // Liberar memoria
    cudaFree(d_A); cudaFree(d_LU); cudaFree(d_b); cudaFree(d_x);
    cudaFree(d_r); cudaFree(d_z); cudaFree(d_Ipiv); cudaFree(d_Info); cudaFree(d_Work);
    
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);
}
