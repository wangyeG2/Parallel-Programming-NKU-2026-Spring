#include<iostream>
#include<stdlib.h>
#include<time.h>
using namespace std;

// 宏定义矩阵乘法访问模式
#define CACHE_ACCESS(i, j) a[i][j] * b[j]

// 模板函数实现矩阵向量乘法
template<typename AccessPattern>
int* matrix_vector_mult(int **a, int *b, int n, AccessPattern access)
{
    int* result = new int[n]();
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            result[i] += access(i, j);
    return result;
}

// 定义cache访问模式的函数对象
struct CacheAccess {
    int **a;
    int *b;
    int operator()(int i, int j) const {
        return a[i][j] * b[j];
    }
};

// 宏封装cache函数调用
#define CACHE_MACRO(a, b, n) matrix_vector_mult(a, b, n, CacheAccess{a, b})

// 模板类实现cache优化矩阵向量乘法
template<int Mode>
class CacheMatrixVectorMul {
public:
    static int* compute(int **a, int *b, int n);
};

template<>
int* CacheMatrixVectorMul<0>::compute(int **a, int *b, int n) {
    int* result = new int[n]();
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            result[i] += a[i][j] * b[j];
    return result;
}

// 宏封装模板类调用
#define CACHE_TEMPLATE(a, b, n) CacheMatrixVectorMul<0>::compute(a, b, n)

long long get_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

int mainmain()
{
    int n;
    cin>>n;
    int** a=new int*[n];
    int* b=new int[n];
    for(int i=0;i<n;i++)
    {
        a[i]=new int[n]();
        b[i]=i+1;
    }
        
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            a[i][j]=i+j;

    long long start, end;
    
    // 使用宏版本的cache函数
    start = get_ns();
    for (int i = 0; i < 1000; i++) {
        int* result = CACHE_MACRO(a, b, n);
        delete[] result;
    }
    end = get_ns();
    cout << "cache optimized (macro):" << (end - start) / 1000000.0 << "ms" << endl;
    
    // 使用模板版本的cache函数
    start = get_ns();
    for (int i = 0; i < 1000; i++) {
        int* result = CACHE_TEMPLATE(a, b, n);
        delete[] result;
    }
    end = get_ns();
    cout << "cache optimized (template):" << (end - start) / 1000000.0 << "ms" << endl;
    
    for(int i = 0; i < n; i++)
        delete[] a[i];
    delete[] a;
    delete[] b;
    
    return 0;
}

int main()
{
    int p=1;
    while(p==1)
    {
        mainmain();
        cout<<endl<<endl<<"Again?";
        cin>>p;
    }
}