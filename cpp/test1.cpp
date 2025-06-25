#include <iostream>
#include <climits>
#include <cmath>
#include <cstdarg>


int main()
{
    using namespace std;
    int n_int = INT_MAX;
    short n_short = SHRT_MAX;
    long n_long = LONG_MAX;
    long long n_llong = LLONG_MAX;

    int re{7};
    char sentence = 'n';
    bool is_ready{true};
    const int Months = 12;

    cout << "re is " << re << endl;

    cout << "int is " << sizeof(int) << " max int is " << n_int << endl; 
    cout << "short is " << sizeof(short) << " max short is " << n_short << endl;
    cout << "llong is " << sizeof(long) << " max llong is " << n_long << endl;
    cout << "llong is " << sizeof(long long) << " max llong is " << n_llong << endl;
    
    cout.put(sentence);
    cout.setf(ios_base::fixed, ios_base::floatfield);
    cout << "pi is about " << 22.0 << endl; // 浮点常量默认double

    char c{33};
    cout << c << endl;


    return 0;
}

int * find_int(int array[], int key, int array_len){
    for(int i=0 ; i < array_len ; i++){
        if(array[i] == key){ 
            return &array[i];
        }
    }
    return NULL;
}

void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

void zero_array(int array[], int array_len){
    int i=array_len;
    while(i > 0){
        array[i--] = 0;
    }
}

long factorial(int n){
    return n <= 1 ? 1 : n * factorial(n-1);  // 分段使用 ?:语法
}

int f(int n){
    return n <= 2 ? 1 : f(n-1) + f(n-2);
}

// #include <cstdarg>
float sum(int n_value, ...){
    va_list var_arg;
    float sum{0};
    va_start(var_arg, n_value);
    for(int i=0; i < n_value; i++){
        sum += va_arg(var_arg, int);
    }
    va_end(var_arg);
    return sum;
}

