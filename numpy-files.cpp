#include "numpy-files.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>

const std::map<dtype, descr> dtypes = {
    { int8,    { "int8",    1, true,  true  } },
    { uint8,   { "uint8",   1, true,  false } },
    { int16,   { "int16",   2, true,  true  } },
    { uint16,  { "uint16",  2, true,  false } },
    { int32,   { "int32",   4, true,  true  } },
    { uint32,  { "uint32",  4, true,  false } },
    { int64,   { "int64",   8, true,  true  } },
    { uint64,  { "uint64",  8, true,  false } },
    { float32, { "float32", 4, false, true  } },
    { float64, { "float64", 8, false, true  } }
};

static void _write_magic_number(std::ofstream& file);
static void _write_version(std::ofstream& file);
static void _build_header(dtype datatype, std::vector<size_t>& dimensions, std::ostringstream& hdr, size_t& num_bytes);
static void _write_header(std::string& header, std::ofstream& file);
static void _write_data(void *data, size_t num_bytes, std::ofstream& file);

void write_numpy_file(void* data, dtype datatype, std::vector<size_t>& dimensions, const std::string& filename)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);

    _write_magic_number(file);
    _write_version(file);

    std::ostringstream header;
    size_t num_bytes = 0;
    _build_header(datatype, dimensions, header, num_bytes);

    std::string header_str = header.str();
    _write_header(header_str, file);

    _write_data(data, num_bytes, file);

    file.close();
    if (file.good()) {
        std::cout << "Wrote " << num_bytes << " bytes (" << num_bytes / dtypes.at(datatype).size << " elements) to '" << filename << "'." << std::endl;
    } else {
        std::cerr << "Error writing array data to '" << filename << "'." << std::endl;
    }
}

void _write_magic_number(std::ofstream& file)
{
    char magic[] = { char(0x93), 'N', 'U', 'M', 'P', 'Y'};
    file.write(magic, 6);

    return;
}

void _write_version(std::ofstream& file)
{
    uint8_t major = 1;
    uint8_t minor = 0;

    file.write((char*)&major, 1);
    file.write((char*)&minor, 1);

    return;
}

void _build_header(dtype datatype, std::vector<size_t>& dimensions, std::ostringstream& hdr, size_t& num_bytes)
{
    descr description = dtypes.at(datatype);

    hdr << "{'descr':'"
        << (description.integer ? (description.is_signed ? 'i' : 'u') : 'f')
        << description.size
        << "','fortran_order':False,'shape':("
        << dimensions[0]
        << ',';

    num_bytes = description.size;
    num_bytes *= dimensions[0];
    for (size_t i = 1; i < dimensions.size(); ++i)
    {
        hdr << dimensions[i] << ',';
        num_bytes *= dimensions[i];
    }
    hdr << ")}\n";
    for (size_t count = 6 + 2 + 2 + hdr.str().size(); count % 64 != 0; ++count)
    {
        hdr << ' ';
    }

    return;
}

void _write_header(std::string& header, std::ofstream& file)
{
    uint16_t header_len = header.size();

    file.write((char*)&header_len, 2);
    file.write((char*)header.data(), header.size());

    return;
}

void _write_data(void *data, size_t num_bytes, std::ofstream& file)
{
    file.write((char *)data, num_bytes);

    return;
}

static void _check_magic_number(std::ifstream& file, const std::string& filename);
static size_t _check_version(std::ifstream& file, const std::string& filename);
static size_t _get_header_size(std::ifstream& file, size_t version);
static void _check_fortran_order(const std::string& header);
static dtype _decode_descr(const std::string& header);
static void _decode_shape(const std::string& header, std::vector<size_t>& dimensions );

void read_numpy_file(const std::string& filename, void*& pdata, dtype& datatype, std::vector<size_t>& dimensions)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    _check_magic_number(file, filename);
    size_t version = _check_version(file, filename);
    size_t header_size = _get_header_size(file, version);
    // {'descr':'u4','fortran_order':False,'shape':(<num_nodes>,<num_intervals>)}
    std::string header(header_size, ' ');
    file.read((char*)header.data(), header_size);

    _check_fortran_order(header);
    datatype = _decode_descr(header);
    _decode_shape(header, dimensions);

    size_t bytes = dtypes.at(datatype).size;
    for (auto dimension : dimensions) {
        bytes *= dimension;
    }

    pdata = malloc(bytes);
    file.read((char*)pdata, bytes);

    file.close();
}

void _check_magic_number(std::ifstream& file, const std::string& filename)
{
    char buffer[6];
    file.read(buffer, 6);
    char magic[] = { char(0x93), 'N', 'U', 'M', 'P', 'Y'};
    if ( memcmp(buffer, magic, 6) != 0 ) {
        std::cerr << "'magic' number for " << filename << " does not match NumPy magic number '\\x93NUMPY'" << std::endl;
        exit(1);
    }
}

size_t _check_version(std::ifstream& file, const std::string& filename)
{
    uint8_t major;
    file.read((char*)&major, 1);
    if ( major != 1 and major != 2 ) {
        std::cerr << "Unsupported NumPy file version - " << major << " in file '" << filename << "'." << std::endl;
        exit(1);
    }

    uint8_t minor;
    file.read((char*)&minor, 1);

    return size_t(major);
}

size_t _get_header_size(std::ifstream& file, size_t version)
{
    size_t header_size;
    if ( version == 1 ) {
        uint16_t temp;
        file.read((char*)&temp, sizeof(temp));
        header_size = size_t(temp);
    }
    else {
        uint32_t temp;
        file.read((char*)&temp, sizeof(temp));
        header_size = size_t(temp);
    }

    return header_size;
}

// ensure 'fortran_order':False
void _check_fortran_order(const std::string& header)
{
    // ¡hacky code! - assumes that if "False" is in the header, it is part of 'fortran_order'
    auto False = header.find("False");
    if ( False == std::string::npos ) {
        std::cerr << "Did not find 'fortran_order':False in file header ('" << header << "')" << std::endl;
        exit(1);
    }
}

// decode descr, '[<=]?[iuf][1248]' supported
dtype _decode_descr(const std::string& header)
{
    auto descr = header.find("descr");
    auto colon = header.find(":", descr);
    auto open = header.find("'", colon);
    auto close = header.find("'", open + 1);
    ++open;
    std::string format = header.substr(open, close-open);
    std::cout << "decode_descr() parsing descr '" << format << "'..." << std::endl;
    // https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
    switch (format[0]) {
        case '<':   // little-endian (Intel x86)
        case '=':   // machine native format
        case '|':   // "not applicable"
            format = format.substr(1);
            break;
        case '>':   // big-endian (Motorla, PowerPC)
            std::cerr << "NumPy file is big-endian which is currently unsupported ('" << header << "')" << std::endl;
            exit(1);
        default:    // no byte-order mark, apparently
            break;
    }
    dtype datatype;
    switch (format[0]) {
        case 'i':   // (signed) integer
            switch (format[1]) {
                case '1':
                    datatype = int8;
                    break;
                case '2':
                    datatype = int16;
                    break;
                case '4':
                    datatype = int32;
                    break;
                case '8':
                    datatype = int64;
                    break;
                default:
                    std::cerr << "Unknown or unsupported data size in NumPy file ('" << header << "')" << std::endl;
                    exit(1);
            }
            break;
        case 'u':   // unsigned integer
            switch (format[1]) {
                case '1':
                    datatype = uint8;
                    break;
                case '2':
                    datatype = uint16;
                    break;
                case '4':
                    datatype = uint32;
                    break;
                case '8':
                    datatype = uint64;
                    break;
                default:
                    std::cerr << "Unknown or unsupported data size in NumPy file ('" << header << "')" << std::endl;
                    exit(1);
            }
            break;
        case 'f':   // floating point
            switch (format[1]) {
                case '4':
                    datatype = float32;
                    break;
                case '8':
                    datatype = float64;
                    break;
                default:
                    std::cerr << "Unknown or unsupported data size in NumPy file ('" << header << "')" << std::endl;
                    exit(1);
            }
            break;
        default:
            std::cerr << "Unknown data type for NumPy file ('" << header << "')" << std::endl;
            exit(1);
    }

    std::cout << "decode_descr() returning '" << dtypes.at(datatype).name << "'." << std::endl;
    return datatype;
}

// decode shape -> std::vector<size_t>
void _decode_shape(const std::string& header, std::vector<size_t>& dimensions)
{
    auto shape = header.find("shape");
    auto open = header.find("(", shape);
    ++open;
    auto close = header.find(")", open);

    std::list<size_t> separators;
    for ( auto comma = header.find(",", open); comma < close; comma = header.find(",", comma+1) ) {
        separators.push_back(comma);
    }
    separators.push_back(close);

    std::cout << "header = '" << header;
    std::cout << "decode_shape() starting at " << open << ", separators at: ";
    for ( auto entry : separators ) {
        std::cout << entry << " ";
    }
    std::cout << std::endl;

    dimensions.clear();
    for ( size_t start = open, finish = separators.front(); start < close; start = finish+1, separators.pop_front(), finish = separators.front()) {
        std::cout << "decode_shape() parsing '" << header.substr(start, finish - start) << "'" << std::endl;
        dimensions.push_back(std::stoull(header.substr(start, finish - start)));
    }

    std::cout << "decode_shape() found the following dimensions: ";
    for ( auto entry : dimensions ) {
        std::cout << entry << " ";
    }
    std::cout << std::endl;
}

void read_numpy_file_aligned(const std::string& filename, aligned_buffer& data, dtype& datatype, std::vector<size_t>& dimensions, size_t alignment)
{
    data.base = 0;
    data.stride = 0;

    std::ifstream file(filename, std::ios::in | std::ios::binary);
    _check_magic_number(file, filename);
    size_t version = _check_version(file, filename);
    size_t header_size = _get_header_size(file, version);

    std::string header(header_size, ' ');
    file.read((char*)header.data(), header_size);

    _check_fortran_order(header);
    datatype = _decode_descr(header);
    _decode_shape(header, dimensions);

    size_t element_size = dtypes.at(datatype).size;

    size_t bytes_per_vector = dimensions[dimensions.size()-1] * element_size;   // raw bytes per vector
    size_t stride = (bytes_per_vector + alignment - 1) & ~(alignment - 1);      // aligned bytes per vector

    size_t total_bytes = stride;
    size_t total_rows = 1;
    for (size_t idim = 0; idim < dimensions.size() - 1; ++idim ) {
        size_t dimension = dimensions[idim];
        total_rows *= dimension;
    }
    total_bytes *= total_rows;

    // data.base = aligned_alloc(alignment, total_bytes);
    if (posix_memalign(&data.base, alignment, total_bytes) == 0) {
        data.stride = stride;

        for ( size_t irow = 0; irow < total_rows; ++irow ) {
            file.read((char*)(data.row(irow)), bytes_per_vector);
        }
    }
    else {
        std::cout << "Could not allocate " << total_bytes << " aligned to " << alignment << " byte boundary." << std::endl;
    }

    file.close();
}
