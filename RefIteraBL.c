/**
 * @file RefIteraBL.c
 * @brief Implementación de refinamiento iterativo usando LAPACK y BLAS (CPU).
 *
 * Contiene la lógica para resolver sistemas lineales Ax=b reutilizando
 * la factorización LU y refinando la solución iterativamente para mitigar
 * errores de redondeo en aritmética de punto flotante.
 *
 * @author Miguel Ángel Norzagaray Cosío
 * @date 2025-12-13
 * * @see RefItera.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "RefItera.h"

// Prototipos de LAPACK/BLAS (standard underscore convention)
extern void dgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO);
extern void dgetrs_(char* TRANS, int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
extern void dgemv_(char* TRANS, int* M, int* N, double* alpha, double* A, int* LDA, double* X, int* INCX, double* beta, double* Y, int* INCY);
extern double dnrm2_(int* N, double* X, int* INCX);
extern void daxpy_(int* N, double* DA, double* DX, int* INCX, double* DY, int* INCY);

void cpu_refinement(int N, double *h_A, double *h_b, double *h_x, int max_iter)
{
    printf("\n--- Inicio Refinamiento CPU (BLAS/LAPACK) ---\n");

    int INFO;
    int one = 1;
    double one_d = 1.0;
    double neg_one_d = -1.0;

    // Reservar memoria de trabajo
    double *LU = (double*)malloc(N * N * sizeof(double));
    double *r = (double*)malloc(N * sizeof(double));
    double *z = (double*)malloc(N * sizeof(double));
    int *IPIV = (int*)malloc(N * sizeof(int));

    // Copiar A -> LU para factorizar (A se preserva para calcular residuos)
    memcpy(LU, h_A, N * N * sizeof(double));
    
    // Inicializar x con b (primer guess)
    memcpy(h_x, h_b, N * sizeof(double));

    // 1. Factorización LU (dgetrf)
    dgetrf_(&N, &N, LU, &N, IPIV, &INFO);
    if (INFO != 0) {
        printf("Error en LU CPU: %d\n", INFO);
        free(LU); free(r); free(z); free(IPIV);
        return;
    }

    // 2. Solución Inicial (dgetrs)
    dgetrs_("N", &N, &one, LU, &N, IPIV, h_x, &N, &INFO);

    // --- Bucle de Refinamiento ---
    printf("%-5s | %-15s\n", "Iter", "||r|| (CPU)");
    printf("--------------------------\n");

    double prev_norm = 1.0e+30;

    for (int k = 0; k < max_iter; k++) {
        memcpy(r, h_b, N * sizeof(double)); // r = b
        
        // r = -1.0 * A * x + 1.0 * r
        dgemv_("N", &N, &N, &neg_one_d, h_A, &N, h_x, &one, &one_d, r, &one);

        double current_norm = dnrm2_(&N, r, &one);
        printf("%-5d | %-1.8e\n", k+1, current_norm);

        // Criterio de parada
        if (current_norm >= prev_norm && k > 0) {
            printf("El error dejó de disminuir.\n");
            break;
        }
        if (current_norm == 0.0) {
            printf("Residuo cero exacto.\n");
            break;
        }
        prev_norm = current_norm;

        // c. Resolver corrección: A*z = r
        memcpy(z, r, N * sizeof(double));
        dgetrs_("N", &N, &one, LU, &N, IPIV, z, &N, &INFO);

        // d. Actualizar: x = x + z
        daxpy_(&N, &one_d, z, &one, h_x, &one);
    }

    free(LU); free(r); free(z); free(IPIV);
}
