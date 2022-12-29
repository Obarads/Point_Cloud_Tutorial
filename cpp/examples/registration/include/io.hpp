#ifndef IO_HPP
#define IO_HPP

#include <iostream>
#include <fstream>

template <typename T>
void write_matrix(T matrix, const char *filename)
{
    std::ofstream outputfile(filename);
    outputfile << matrix;
    outputfile.close();
}

#endif
