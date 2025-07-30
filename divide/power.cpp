#include <iostream>
#include <chrono>
using namespace std;

long long slowPow(long long a, long long n) {
    long long res = 1;
    for(long long i = 0; i < n; ++i) {
        res *= a;
    }
    return res;
}

long long fastPow(long long a, long long n) {
    long long res = 1;
    long long base = a;
    while(n > 0) {
        if(n & 1) res *= base;
        base = base * base;
        n >>= 1;
    }
    return res;
}

int main() {
    const int cnt = 1000000;
    long long a = 123456;
    long long n = 1000000; // 1e5

    volatile long long res = 0; // 防优化

    // 位运算快速幂
    auto start1 = chrono::high_resolution_clock::now();
    for(int i = 0; i < 10000; ++i) {
        res = fastPow(a, n);
    }
    auto end1 = chrono::high_resolution_clock::now();
    double t1 = chrono::duration<double, milli>(end1 - start1).count();

    // 朴素循环幂
    auto start2 = chrono::high_resolution_clock::now();
    for(int i = 0; i < 10000; ++i) { // 朴素幂太慢，次数降低
        res = slowPow(a, n);
    }
    auto end2 = chrono::high_resolution_clock::now();
    double t2 = chrono::duration<double, milli>(end2 - start2).count();

    
    cout << "bitwise fastPow, time: " << t1 << " ms" << endl;
    cout << "naive slowPow,   time: " << t2 << " ms" << endl;
    return 0;
}

/*
cnt = 10000
F:\TechniqueOptimization\divide>a.exe        
bitwise fastPow, time: 1.006 ms
naive slowPow,   time: 25928.2 ms
*/