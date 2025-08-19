#pragma once
#include <cstdint>
#include <string>

namespace znet {

enum class DType {
  Float32,
  Float64,
  Int32,
  Int64,
  Bool,
};

inline constexpr DType kDefaultDType = DType::Float32;

inline constexpr std::size_t size_of(DType dt) {
  switch (dt) {
    case DType::Float32: return 4;
    case DType::Float64: return 8;
    case DType::Int32:   return 4;
    case DType::Int64:   return 8;
    case DType::Bool:    return 1;
  }
  return 0;
}

inline constexpr bool is_floating(DType dt) {
  return dt == DType::Float32 || dt == DType::Float64;
}

inline std::string to_string(DType dt) {
  switch (dt) {
    case DType::Float32: return "float32";
    case DType::Float64: return "float64";
    case DType::Int32:   return "int32";
    case DType::Int64:   return "int64";
    case DType::Bool:    return "bool";
  }
  return "unknown";
}

// --- C++ type to DType mapping -----------------------------------------
// NEW (DType Step 2)
template <typename T> struct CTypeToDType;                 // primary template (no impl)

// NEW (DType Step 2) specializations
template <> struct CTypeToDType<float>   { static constexpr DType value = DType::Float32; };
template <> struct CTypeToDType<double>  { static constexpr DType value = DType::Float64; };
template <> struct CTypeToDType<int32_t> { static constexpr DType value = DType::Int32;   };
template <> struct CTypeToDType<int64_t> { static constexpr DType value = DType::Int64;   };
template <> struct CTypeToDType<bool>    { static constexpr DType value = DType::Bool;    };

// NEW (DType Step 2)
template <typename T>
constexpr DType dtype_of() { return CTypeToDType<std::remove_cv_t<std::remove_reference_t<T>>>::value; }


} // namespace znet
