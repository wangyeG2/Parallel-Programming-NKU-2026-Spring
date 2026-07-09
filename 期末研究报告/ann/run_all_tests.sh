#!/bin/bash
# ============================================================
# run_all_tests.sh — ANN 并行优化自动化测试脚本
# 环境: 华为鲲鹏服务器 (aarch64, ARM NEON)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==================== 配置 ====================
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"
RESULT_CSV_ALG="$RESULT_DIR/results_alg_$(date +%Y%m%d_%H%M%S).csv"
RESULT_CSV_PAR="$RESULT_DIR/results_par_$(date +%Y%m%d_%H%M%S).csv"
LOG_FILE="$RESULT_DIR/test.log"

THREAD_LIST=(1 2 4 8)
P_LIST=(10 50 100 200 500 1000)
NPROBE_LIST=(1 2 4 8 16 32 64)
MPI_PROC_LIST=(1 2 4)
NLIST=256

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_cpu_test() {
    local alg_name=$1
    local compile_flags=$2
    local parallel_method=$3
    local exe_name="main_${alg_name}"
    log "Compiling $exe_name ..."
    if ! g++ main.cc -o "$exe_name" -O2 -fopenmp -lpthread -std=c++11 $compile_flags 2>&1 | tee -a "$LOG_FILE"; then
        log " COMPILE FAILED: $exe_name (see errors above)"
        return 1
    fi
    log "Running $exe_name ..."
    local output
    output=$("./$exe_name" 2>&1)
    local status=$?
    if [ $status -ne 0 ]; then
        log " RUN FAILED: $exe_name (exit code $status)"
        log " ERROR OUTPUT:"
        echo "$output" | tee -a "$LOG_FILE"
        return 1
    fi
    local recall=$(echo "$output" | grep "average recall" | awk '{print $3}')
    local latency=$(echo "$output" | grep "average latency" | awk '{print $4}')
    if [ -n "$recall" ] && [ -n "$latency" ]; then
        echo "$alg_name,$recall,$latency" >> "$RESULT_CSV_ALG"
        echo "$parallel_method,$recall,$latency" >> "$RESULT_CSV_PAR"
        log " OK: recall=$recall, latency=$latency us"
    else
        log " PARSE FAILED: output was -> $output"
    fi
}

run_mpi_test() {
    local alg_name=$1
    local np=$2
    local compile_flags=$3
    local parallel_method=$4
    local exe_name="main_${alg_name}"
    log "Compiling $exe_name ..."
    if ! mpic++ main.cc -o "$exe_name" -O2 -fopenmp -std=c++11 $compile_flags 2>&1 | tee -a "$LOG_FILE"; then
        log " COMPILE FAILED: $exe_name (see errors above)"
        return 1
    fi
    log "Running $exe_name with np=$np ..."
    local output
    output=$(mpirun -np "$np" "$SCRIPT_DIR/$exe_name" 2>&1)
    local status=$?
    if [ $status -ne 0 ]; then
        log " RUN FAILED: $exe_name (exit code $status)"
        log " ERROR OUTPUT:"
        echo "$output" | tee -a "$LOG_FILE"
        return 1
    fi
    local recall=$(echo "$output" | grep "average recall" | tail -n 1 | awk '{print $3}')
    local latency=$(echo "$output" | grep "average latency" | tail -n 1 | awk '{print $4}')
    if [ -n "$recall" ] && [ -n "$latency" ]; then
        echo "$alg_name,$recall,$latency" >> "$RESULT_CSV_ALG"
        echo "$parallel_method,$recall,$latency" >> "$RESULT_CSV_PAR"
        log " OK: recall=$recall, latency=$latency us"
    else
        log " PARSE FAILED: output was -> $output"
    fi
}

# ==================== 准备工作 ====================
if [ -f "ivf-simd-mpi-多线程.h" ] && [ ! -f "ivf-simd-mpi-mt.h" ]; then
    cp "ivf-simd-mpi-多线程.h" "ivf-simd-mpi-mt.h"
    log "Copied ivf-simd-mpi-多线程.h -> ivf-simd-mpi-mt.h"
fi

echo "Algorithm,Recall,Avg_Latency(us)" > "$RESULT_CSV_ALG"
echo "Parallel_Method,Recall,Avg_Latency(us)" > "$RESULT_CSV_PAR"

log "==================== START TESTING ===================="

# ==================== 1. Flat 基线 (serial) ====================
log "--- Testing Flat Baseline (serial) ---"
run_cpu_test "flat" "-DUSE_FLAT" "serial"

# ==================== 2. Flat SIMD ====================
log "--- Testing Flat SIMD (simd) ---"
run_cpu_test "flat_simd" "-DUSE_FLAT_SIMD_OMP -DALG_THREADS=1" "simd"

# ==================== 3. Flat OpenMP ====================
log "--- Testing Flat OpenMP (omp) ---"
for t in "${THREAD_LIST[@]}"; do
    run_cpu_test "flat_omp_t${t}" "-DUSE_FLAT_SIMD_OMP -DALG_THREADS=$t" "omp"
done

# ==================== 4. Flat Pthread (归为omp) ====================
log "--- Testing Flat Pthread (omp) ---"
for t in "${THREAD_LIST[@]}"; do
    run_cpu_test "flat_pthread_t${t}" "-DUSE_FLAT_SIMD_PTHREAD -DALG_THREADS=$t" "omp"
done

# ==================== 5. SQ SIMD ====================
log "--- Testing SQ SIMD (simd) ---"
for p in "${P_LIST[@]}"; do
    run_cpu_test "sq_p${p}" "-DUSE_SQ -DALG_P=$p" "simd"
done

# ==================== 6. PQ SIMD ====================
log "--- Testing PQ SIMD (simd) ---"
for p in "${P_LIST[@]}"; do
    run_cpu_test "pq_p${p}" "-DUSE_PQ -DALG_P=$p" "simd"
done

# ==================== 7. PQ OpenMP ====================
log "--- Testing PQ OpenMP (omp) ---"
for p in "${P_LIST[@]}"; do
    for t in "${THREAD_LIST[@]}"; do
        run_cpu_test "pq_p${p}_t${t}" "-DUSE_PQ -DALG_P=$p -DALG_THREADS=$t" "omp"
    done
done

# ==================== 8. IVF-PQ SIMD ====================
log "--- Testing IVF-PQ SIMD (simd) ---"
for nprobe in "${NPROBE_LIST[@]}"; do
    for p in "${P_LIST[@]}"; do
        run_cpu_test "ivfpq_n${nprobe}_p${p}" "-DUSE_IVF_PQ -DALG_NPROBE=$nprobe -DALG_P=$p -DALG_NLIST=$NLIST" "simd"
    done
done

# ==================== 9. IVF MPI ====================
if command -v mpic++ &>/dev/null; then
    log "--- Testing IVF MPI (mpi) ---"
    for nprobe in "${NPROBE_LIST[@]}"; do
        for np in "${MPI_PROC_LIST[@]}"; do
            run_mpi_test "ivfmpi_n${nprobe}" "$np" "-DUSE_IVF_MPI -DALG_NPROBE=$nprobe -DALG_NLIST=$NLIST" "mpi"
        done
    done

    # ==================== 10. IVF MPI + MT (mpi) ====================
    log "--- Testing IVF MPI + OpenMP (mpi) ---"
    for nprobe in "${NPROBE_LIST[@]}"; do
        for np in "${MPI_PROC_LIST[@]}"; do
            run_mpi_test "ivfmpi_mt_n${nprobe}" "$np" "-DUSE_IVF_MPI_MT -DALG_NPROBE=$nprobe -DALG_NLIST=$NLIST" "mpi"
        done
    done

    # ==================== 11. HNSW (omp) ====================
    log "--- Testing HNSW (omp) ---"
    run_cpu_test "hnsw" "-DUSE_HNSW" "omp"

    # ==================== 12. IVF-HNSW (omp) ====================
    log "--- Testing IVF-HNSW (omp) ---"
    run_cpu_test "ivf_hnsw" "-DUSE_IVF_HNSW -DALG_NLIST=$NLIST" "omp"

    # ==================== 13. IVF-HNSW-MPI (mpi) ====================
    log "--- Testing IVF-HNSW-MPI (mpi) ---"
    for nprobe in "${NPROBE_LIST[@]}"; do
        for np in "${MPI_PROC_LIST[@]}"; do
            run_mpi_test "ivf_hnsw_mpi_n${nprobe}" "$np" "-DUSE_IVF_HNSW_MPI -DALG_NPROBE=$nprobe -DALG_NLIST=$NLIST" "mpi"
        done
    done

    # ==================== 14. Partition-HNSW-MPI (mpi) ====================
    log "--- Testing Partition-HNSW-MPI (mpi) ---"
    for np in "${MPI_PROC_LIST[@]}"; do
        run_mpi_test "partition_hnsw_mpi" "$np" "-DUSE_PARTITION_HNSW_MPI" "mpi"
    done

    # ==================== 15. HNSW-on-HNSW-MPI (mpi) ====================
    log "--- Testing HNSW-on-HNSW-MPI (mpi) ---"
    for nprobe in "${NPROBE_LIST[@]}"; do
        for np in "${MPI_PROC_LIST[@]}"; do
            run_mpi_test "hnsw_on_hnsw_mpi_n${nprobe}" "$np" "-DUSE_HNSW_ON_HNSW_MPI -DALG_NPROBE=$nprobe -DALG_NLIST=$NLIST" "mpi"
        done
    done

else
    log "mpic++ not found, skipping MPI tests."
fi

log "All tests completed!"
log "Algorithm results saved to: $RESULT_CSV_ALG"
log "Parallel method results saved to: $RESULT_CSV_PAR"
