#include<iostream>
#include<stdlib.h>
#include<time.h>

using namespace std;


int* cache(int **a,int *b,int n)
{
    int* result=new int[n]();
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            result[i]+=a[i][j]*b[j];
    return result;
}

int* cache_with_stride(int **a, int *b, int n, int stride)
{
    int* result = new int[n]();
    for(int i = 0; i < n; i++) {
        for(int s = 0; s < stride; s++) {
            for(int j = s; j < n; j += stride) {
                result[i] += a[i][j] * b[j];
            }
        }
    }
    return result;
}

long long get_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

int mainmain()
{
    int n;
    cin >> n;
    
    int** a = new int*[n];
    int* b = new int[n];
    for(int i = 0; i < n; i++) {
        a[i] = new int[n]();
        b[i] = i + 1;
    }
        
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            a[i][j] = i + j;

    long long start, end;

    start = get_ns();
    for (int i = 0; i < 1000; i++) {
        int* res = cache(a, b, n);
        delete[] res;
    }
    end = get_ns();
    cout << "Cache Optimized (Stride 1): " << (end - start) / 1000000.0 << " ms" << endl;
    int strides[] = {2, 4, 8, 16, 32, 64};
    for(int s : strides) {
        start = get_ns();
        for (int i = 0; i < 1000; i++) {
            int* res = cache_with_stride(a, b, n, s);
            delete[] res;
        }
        end = get_ns();
        cout << "Stride " << s << ": " << (end - start) / 1000000.0 << " ms" << endl;
    }
    for(int i = 0; i < n; i++)
        delete[] a[i];
    delete[] a;
    delete[] b;
    return 0;
}

int main()
{
    int p = 1;
    while(p == 1)
    {
        mainmain();
        cout << endl << "Again?";
        cin >> p;
    }
    return 0;
}