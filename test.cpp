// Compile with "g++ test.cpp numpy-files.cpp" executable in a.out

#include <iostream>
#include <vector>

#include "numpy-files.h"

# define SIZEOF(_a)    sizeof(_a)/sizeof(*_a)

int main(int argc, char* argv[])
{
    // 1-D array uint32
    uint32_t one_d[32];

    for (size_t i = 0; i < SIZEOF(one_d); ++i) {
        one_d[i] = uint32_t(i);
    }

    std::vector<size_t> dimensions = { size_t(SIZEOF(one_d)) };
    write_numpy_file(one_d, dtype::uint32, dimensions, "one_d.npy");

    // 2-D array float64
    size_t size = 8;
    double two_d[1<<size];
    size_t rows_cols = size / 2;

    for (size_t i = 0; i < (1 << rows_cols); ++i) {
        for (size_t j = 0; j < (1 << rows_cols); ++j) {
            size_t index = (i << rows_cols) + j;
            two_d[index] = double(index);
        }
    }

    dimensions = { size_t(1 << rows_cols), size_t(1 << rows_cols) };
    write_numpy_file(two_d, dtype::float64, dimensions, "two_d.npy");

    void* copyOne;
    dtype data_type;
    dimensions.clear();
    read_numpy_file("one_d.npy", copyOne, data_type, dimensions);

    void* copyTwo;
    dimensions.clear();
    read_numpy_file("two_d.npy", copyTwo, data_type, dimensions);

    return 0;
}
