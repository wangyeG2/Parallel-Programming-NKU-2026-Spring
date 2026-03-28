#include<iostream>
#include<stdlib.h>
#include<cmath>
#include<time.h>
using namespace std;

unsigned long long normal(unsigned long long *a,unsigned long long p)
{
    unsigned long long result=0;
    for(unsigned long long i=0;i<p;i++)
    	result+=a[i];
    return result;
}
unsigned long long superscalar1(unsigned long long *a,unsigned long long p)
{
    unsigned long long result1=0,result2=0;
    for(unsigned long long i=0;i<p-1;i+=2)
    {
    	result1+=a[i];
    	result2+=a[i+1];
	}
    return result1+result2;
}

unsigned long long superscalar2(unsigned long long *a, unsigned long long p)
{
    unsigned long long result1 = 0, result2 = 0, result3 = 0, result4 = 0;
    unsigned long long i;
    for (i = 0; i + 3 < p; i += 4) {
        result1 += a[i];
        result2 += a[i + 1];
        result3 += a[i + 2];
        result4 += a[i + 3];
    }
    for (; i < p; i++) {
        result1 += a[i];   
    }
    return result1 + result2 + result3 + result4;
}
long long get_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000 + ts.tv_nsec;
}
int mainmain()
{
    int n;
    cin>>n;
    unsigned long long p=pow(2,n);
    unsigned long long* a=new unsigned long long[p];
        
    for(unsigned long long i=0;i<p;i++)
    	a[i]=i;
    long long start, end;
	start = get_ns();
	for (int i = 0; i < 1000; i++) {
		normal(a, p);
	}
	end = get_ns();
	cout << "normal:" << (end - start) / 1000000.0 << "ms" << endl;
	start = get_ns();
	for (int i = 0; i < 1000; i++) {
		superscalar1(a, p);
	}
	end = get_ns();
	cout << "superscalar:" << (end - start) / 1000000.0 << "ms" << endl;
    start = get_ns();
    for (int i = 0; i < 1000; i++) {
		superscalar2(a, p);
	}
	end = get_ns();
	cout << "superscalar unroll:" << (end - start) / 1000000.0 << "ms" << endl;
    delete[] a;
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