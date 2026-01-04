# Refinamiento Iterativo: Comparación CPU vs GPU vs Híbrido

Implementación y comparación de refinamiento iterativo para sistemas lineales usando tres enfoques diferentes: CPU (OpenBLAS/LAPACK), GPU (CUDA/cuBLAS) y Híbrido (MAGMA).

## Descripción

Este proyecto resuelve sistemas lineales **Ax = b** usando factorización LU con refinamiento iterativo para mejorar la precisión numérica. El proceso corrige errores de redondeo mediante:

1.  Resolver **Ax₀ = b** (solución inicial).
2.  Calcular residuo **r = b - Ax**.
3.  Resolver **Az = r** (corrección).
4.  Actualizar **x = x + z**.
5.  Repetir hasta convergencia.

## Estructura del Proyecto

```text
.
├── main.c              # Programa principal y orquestador
├── RefItera.h          # Definiciones de funciones y estructuras
├── RefIteraBL.c        # Implementación CPU (OpenBLAS/LAPACK)
├── cuRefItera.cu       # Implementación GPU (CUDA/cuBLAS)
├── magmaRefItera.cpp   # Implementación híbrida (MAGMA C++)
├── Makefile            # Script de compilación robusto
└── README.md           # Este archivo
```

## Requisitos

### Software
* **Compiladores:** GCC y NVCC (CUDA Toolkit ≥ 11.0).
* **Librerías CPU:** OpenBLAS y LAPACK.
* **Librería Híbrida:** MAGMA (≥ 2.6).

### Hardware
* **GPU:** NVIDIA compatible con CUDA.
* **Compute Capability:** Se recomienda ≥ 6.0 (Pascal o superior).

## Compilación

El proyecto utiliza un `Makefile` que gestiona versiones de **Depuración** (Debug) y **Lanzamiento** (Release) por separado para evitar conflictos.

```bash
# Compilar todo (genera ambas versiones)
make

# Compilar solo versión optimizada (Recomendado para benchmarks)
make release

# Compilar solo versión debug (Para desarrollo con gdb/cuda-gdb)
make debug

# Limpiar binarios y objetos
make clean
```

**Configuración del Makefile:**
Antes de compilar, verifica la variable `MAGMA_DIR` en el Makefile si tu instalación no está en `/usr/local/magma`.

## Uso

El sistema genera dos ejecutables distintos según la compilación:

### 1. Ejecución Estándar (Optimized)
Usa esta versión para medir tiempos y rendimiento real.
```bash
./RefItera_release
```

### 2. Ejecución de Depuración
Usa esta versión si necesitas trazar errores (incluye símbolos de debug).
```bash
./RefItera_debug
```

## Resultados Esperados

El programa resuelve una matriz mal condicionada y muestra la reducción del residuo $\|r\|$ en cada paso:

```text
--- Inicio Refinamiento CPU (BLAS/LAPACK) ---
Iter  | ||r|| (CPU)
--------------------------
1     | 1.46482138e-14
2     | 5.32907052e-15
STOP: El error dejó de disminuir.

--- Inicio Refinamiento GPU (CUDA/cuBLAS) ---
Iter  | ||r|| (GPU)
--------------------------
1     | 1.48952049e-14
...
STOP: El error dejó de disminuir.

--- Inicio Refinamiento MAGMA (Híbrido) ---
Iter  | ||r|| (MAGMA)
--------------------------
1     | 1.60180214e-14
...
STOP: El error dejó de disminuir.
```

## Notas Pedagógicas

1.  **Aritmética de Punto Flotante:** Aunque matemáticamente equivalentes, las tres implementaciones difieren en los residuos intermedios debido a la no asociatividad de la suma en punto flotante y el orden de ejecución paralelo.
2.  **Límite de Precisión:** La convergencia se detiene cerca del épsilon de máquina (~10⁻¹⁵ para `double`).
3.  **Arquitecturas:**
    * **CPU:** Baja latencia, ideal para problemas pequeños o seriales.
    * **GPU:** Alta latencia inicial, pero enorme throughput para matrices grandes.
    * **MAGMA:** Busca el equilibrio delegando la factorización pesada a la GPU y el control a la CPU.

## Autor

Miguel Ángel Norzagaray Cosío
UABCS / DASC
Diciembre 2024 - Enero 2025