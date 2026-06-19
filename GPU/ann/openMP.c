#include <stdio.h>
#include <omp.h>
 
#define SIZE 1000
 
int main() {
    int array[SIZE];
    long sum = 0;
 
    // 初始化数组
    for (int i = 0; i < SIZE; i++) {
        array[i] = i + 1;
    }
 
    // 并行累加
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < SIZE; i++) {
        sum += array[i];
    }
 
    printf("数组累加结果: %ld\n", sum);
    return 0;
}