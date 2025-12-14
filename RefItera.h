#ifndef REF_ITERA_H
#define REF_ITERA_H

#ifdef __cplusplus
extern "C" {
#endif

// Función que orquesta el refinamiento iterativo en GPU
// N: Dimensión
// A_host: Puntero a la matriz en RAM (Column-Major)
// b_host: Puntero al vector b en RAM
// x_host: Puntero donde se guardará el resultado final
// max_iter: Límite de iteraciones
void gpu_refinement(int N, double *A_host, double *b_host, double *x_host, int max_iter);

// Versión CPU (BLAS/LAPACK)
void cpu_refinement(int N, double *h_A, double *h_b, double *h_x, int max_iter);

#ifdef __cplusplus
}
#endif

#endif