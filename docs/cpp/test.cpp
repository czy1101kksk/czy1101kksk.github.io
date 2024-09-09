#include <iostream>
#include <climits>


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
