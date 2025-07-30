#include <iostream>
#include <chrono>
#include <cstdlib>
using namespace std;

// 手写位移除法
int myDivide(int dividend, int divisor) {
    const int maxx = 2147483647;
    const int minn = -2147483648;
    if (dividend == minn && divisor == -1) {
        return maxx;
    }
    bool negative = (dividend < 0) ^ (divisor < 0);
    long long dd = abs((long long)dividend);
    long long dr = abs((long long)divisor);
    long long ans = 0;
    for(int i=31; i>=0; i--){
        if(dr <= (dd >> i)) {
            dd -= (dr << i);
            ans += (1 << i);
        }
    }
    return (negative ? -ans : ans);
}

int origindivide(int dividend, int divisor) {
    return dividend / divisor;
}

int main() {
    const int cnt = 1000000;
    int dividend = 123456789;
    int divisor  = 12345;
    // volatile int res = 0;  // 防止编译器优化
    volatile int res = 0; 

    // 计时内建除法
    auto start1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < cnt; ++i) {
        res = origindivide(dividend, divisor);
    }
    auto end1 = chrono::high_resolution_clock::now();
    double t1 = chrono::duration<double, milli>(end1 - start1).count();
    
    // 计时手写除法
    auto start2 = chrono::high_resolution_clock::now();
    for (int i = 0; i < cnt; ++i) {
        res = myDivide(dividend, divisor);
    }
    auto end2 = chrono::high_resolution_clock::now();
    double t2 = chrono::duration<double, milli>(end2 - start2).count();

    cout << "builtin '/', time: " << t1 << " ms" << endl;
    cout << "bitwise divide, time: " << t2 << " ms" << endl;
    return 0;
}