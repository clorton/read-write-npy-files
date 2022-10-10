#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

enum dtype {
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64
};

struct descr {
    std::string name;
    size_t size;
    bool integer;
    bool is_signed;
};

extern const std::map<dtype, descr> dtypes;

void write_numpy_file(void* data, dtype datatype, std::vector<size_t>& dimensions, const std::string& filename);
void read_numpy_file(const std::string& filename, void*& pdata, dtype& data_type, std::vector<size_t>& dimensions);

struct aligned_buffer {
    void* base;
    size_t stride;  // in bytes
    inline void* row(size_t index) { return (void*) ((char*)base + index*stride); }
};

void read_numpy_file_aligned(const std::string& filename, aligned_buffer& pdata, dtype& data_type, std::vector<size_t>& dimensions, size_t alignment = 32);
