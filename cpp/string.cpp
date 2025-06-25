#include <iostream>
#include <string>
#include <cstring>

int main(void)
{
    std::string str = "Hello World";
    std::cout << str << " " << str.size() <<std::endl;
    std::cout << R"+*( "(Hello)-(World)" )+*" << std::endl;
    return 0;
}