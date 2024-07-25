#ifndef ARRAYREDUCTION_HPP
#define ARRAYREDUCTION_HPP
namespace sample {

template <class ScalarType, int N>
struct array_type {
  ScalarType myArray[N];

  KOKKOS_INLINE_FUNCTION array_type()
  {
    init();
  }

  KOKKOS_INLINE_FUNCTION void init()
  {
    for (int i = 0; i < N; i++) {
      myArray[i] = 0.0;
    }
  } // initialize myArray to 0

  KOKKOS_INLINE_FUNCTION void operator+=(const array_type &src)
  {
    for (int i = 0; i < N; i++) {
      myArray[i] += src.myArray[i];
    }
  }
};

template <class T, class Space, int N>
struct SumMyArray {
public:
  // Required
  typedef SumMyArray                                                 reducer;
  typedef array_type<T, N>                                           value_type;
  typedef Kokkos::View<value_type *, Space, Kokkos::MemoryUnmanaged> result_view_type;

private:
  value_type &value;

public:
  KOKKOS_INLINE_FUNCTION SumMyArray(value_type &value_)
      : value(value_) {}

  // Required
  KOKKOS_INLINE_FUNCTION void join(value_type &dest, const value_type &src) const
  {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION void init(value_type &val) const
  {
    val.init();
  }

  KOKKOS_INLINE_FUNCTION value_type &reference() const
  {
    return value;
  }

  KOKKOS_INLINE_FUNCTION result_view_type view() const
  {
    return result_view_type(&value, 1);
  }

  KOKKOS_INLINE_FUNCTION bool references_scalar() const
  {
    return true;
  }
};
} // namespace sample

#endif