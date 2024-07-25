/***
 *  For the complex step derivative method:
 *  f'(x) ~  Im [ f(x+ih) ] / h
 *  Define a double complex class that inherits from the
 *  library complex type and overloads appropriate operators.
 *  Mon Jan  8 22:42:20 PST 2001
 *      CODE DOWNLOADED FROM PROF. MARTIN'S WEBSITE
 *          http://mdolab.engin.umich.edu/content/complex-step-guide-cc
 ***/

#ifndef HDRcomplexify
#define HDRcomplexify
#include <Kokkos_Core.hpp>
#include "exafmmTypes.hpp"

#ifndef HDRderivify
KOKKOS_INLINE_FUNCTION exafmm::real_t real(const exafmm::real_t &r)
{
  return r;
}

KOKKOS_INLINE_FUNCTION exafmm::real_t imag(const exafmm::real_t)
{
  return 0.;
}
#endif // HDRderivify

class cplx : public exafmm::complex_t {
public:
  KOKKOS_INLINE_FUNCTION cplx()
      : exafmm::complex_t() {};
  KOKKOS_INLINE_FUNCTION cplx(const exafmm::real_t &d)
      : exafmm::complex_t(d) {};
  KOKKOS_INLINE_FUNCTION cplx(const exafmm::real_t &r, const exafmm::real_t &i)
      : exafmm::complex_t(r, i) {};
  KOKKOS_INLINE_FUNCTION cplx(const Kokkos::complex<double> &z)
      : exafmm::complex_t(z) {};
  KOKKOS_INLINE_FUNCTION cplx(const Kokkos::complex<float> &z)
      : exafmm::complex_t(z) {};
  KOKKOS_INLINE_FUNCTION operator exafmm::real_t()
  {
    return this->real();
  }
  KOKKOS_INLINE_FUNCTION operator int()
  {
    return int(this->real());
  }
  // relational operators
  // Conversion constructor should be able to take care of the
  // operator== and != calls with double, but MIPS compiler
  // complains of ambiguous inheritance.  This should be more
  // efficient anyway.  (A hint of what comes below.)
  friend KOKKOS_INLINE_FUNCTION bool operator==(const cplx &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator==(const cplx &, const exafmm::real_t &);
  friend KOKKOS_INLINE_FUNCTION bool operator==(const exafmm::real_t &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator!=(const cplx &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator!=(const cplx &, const exafmm::real_t &);
  friend KOKKOS_INLINE_FUNCTION bool operator!=(const exafmm::real_t &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator>(const cplx &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator>(const cplx &, const exafmm::real_t &);
  friend KOKKOS_INLINE_FUNCTION bool operator>(const exafmm::real_t &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator<(const cplx &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator<(const cplx &, const exafmm::real_t &);
  friend KOKKOS_INLINE_FUNCTION bool operator<(const exafmm::real_t &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator>=(const cplx &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator>=(const cplx &, const exafmm::real_t &);
  friend KOKKOS_INLINE_FUNCTION bool operator>=(const exafmm::real_t &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator<=(const cplx &, const cplx &);
  friend KOKKOS_INLINE_FUNCTION bool operator<=(const cplx &, const exafmm::real_t &);
  friend KOKKOS_INLINE_FUNCTION bool operator<=(const exafmm::real_t &, const cplx &);
  // here's the annoying thing:
  // Every function in class complex<double> that returns a
  // complex<double> causes ambiguities with function overloading
  // resolution because of the mix of types cplx and
  // complex<double> and double and int in math expressions.
  // So, although they are inherited, must redefine them
  // to return type cplx:
  // basic arithmetic
  KOKKOS_INLINE_FUNCTION cplx        operator+() const;
  KOKKOS_INLINE_FUNCTION cplx        operator+(const cplx &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator+(const exafmm::real_t &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator+(const int &) const;
  KOKKOS_INLINE_FUNCTION friend cplx operator+(const exafmm::real_t &, const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx operator+(const int &, const cplx &);
  KOKKOS_INLINE_FUNCTION cplx        operator-() const;
  KOKKOS_INLINE_FUNCTION cplx        operator-(const cplx &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator-(const exafmm::real_t &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator-(const int &) const;
  KOKKOS_INLINE_FUNCTION friend cplx operator-(const exafmm::real_t &, const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx operator-(const int &, const cplx &);
  KOKKOS_INLINE_FUNCTION cplx        operator*(const cplx &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator*(const exafmm::real_t &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator*(const int &) const;
  KOKKOS_INLINE_FUNCTION friend cplx operator*(const exafmm::real_t &, const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx operator*(const int &, const cplx &);
  KOKKOS_INLINE_FUNCTION cplx        operator/(const cplx &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator/(const exafmm::real_t &) const;
  KOKKOS_INLINE_FUNCTION cplx        operator/(const int &) const;
  KOKKOS_INLINE_FUNCTION friend cplx operator/(const exafmm::real_t &, const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx operator/(const int &, const cplx &);
  // from <math.h>
  KOKKOS_INLINE_FUNCTION friend cplx sin(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx sinh(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx cos(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx cosh(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx tan(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx tanh(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx log10(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx log(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx sqrt(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx exp(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx pow(const cplx &, const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx pow(const cplx &, const exafmm::real_t &);
  KOKKOS_INLINE_FUNCTION friend cplx pow(const cplx &, const int &);
  KOKKOS_INLINE_FUNCTION friend cplx pow(const exafmm::real_t &, const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx pow(const int &, const cplx &);
  // complex versions of these are not in standard library
  // or they need to be redefined:
  // (frexp, modf, and fmod have not been dealt with)
  KOKKOS_INLINE_FUNCTION friend cplx fabs(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx asin(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx acos(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx atan(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx atan2(const cplx &, const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx ceil(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx floor(const cplx &);
  KOKKOS_INLINE_FUNCTION friend cplx ldexp(const cplx &, const int &);
};

KOKKOS_INLINE_FUNCTION bool operator==(const cplx &lhs, const cplx &rhs)
{
  return real(lhs) == real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator==(const cplx &lhs, const exafmm::real_t &rhs)
{
  return real(lhs) == rhs;
}

KOKKOS_INLINE_FUNCTION bool operator==(const exafmm::real_t &lhs, const cplx &rhs)
{
  return lhs == real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator!=(const cplx &lhs, const cplx &rhs)
{
  return real(lhs) != real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator!=(const cplx &lhs, const exafmm::real_t &rhs)
{
  return real(lhs) != rhs;
}

KOKKOS_INLINE_FUNCTION bool operator!=(const exafmm::real_t &lhs, const cplx &rhs)
{
  return lhs != real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator>(const cplx &lhs, const cplx &rhs)
{
  return real(lhs) > real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator>(const cplx &lhs, const exafmm::real_t &rhs)
{
  return real(lhs) > rhs;
}

KOKKOS_INLINE_FUNCTION bool operator>(const exafmm::real_t &lhs, const cplx &rhs)
{
  return lhs > real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator<(const cplx &lhs, const cplx &rhs)
{
  return real(lhs) < real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator<(const cplx &lhs, const exafmm::real_t &rhs)
{
  return real(lhs) < rhs;
}

KOKKOS_INLINE_FUNCTION bool operator<(const exafmm::real_t &lhs, const cplx &rhs)
{
  return lhs < real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator>=(const cplx &lhs, const cplx &rhs)
{
  return real(lhs) >= real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator>=(const cplx &lhs, const exafmm::real_t &rhs)
{
  return real(lhs) >= rhs;
}

KOKKOS_INLINE_FUNCTION bool operator>=(const exafmm::real_t &lhs, const cplx &rhs)
{
  return lhs >= real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator<=(const cplx &lhs, const cplx &rhs)
{
  return real(lhs) <= real(rhs);
}

KOKKOS_INLINE_FUNCTION bool operator<=(const cplx &lhs, const exafmm::real_t &rhs)
{
  return real(lhs) <= rhs;
}

KOKKOS_INLINE_FUNCTION bool operator<=(const exafmm::real_t &lhs, const cplx &rhs)
{
  return lhs <= real(rhs);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator+() const
{
  return +exafmm::complex_t(*this);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator+(const cplx &z) const
{
  return exafmm::complex_t(*this) + exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator+(const exafmm::real_t &r) const
{
  return exafmm::complex_t(*this) + r;
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator+(const int &i) const
{
  return exafmm::complex_t(*this) + exafmm::real_t(i);
}

KOKKOS_INLINE_FUNCTION cplx operator+(const exafmm::real_t &r, const cplx &z)
{
  return r + exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx operator+(const int &i, const cplx &z)
{
  return exafmm::real_t(i) + exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator-() const
{
  return -exafmm::complex_t(*this);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator-(const cplx &z) const
{
  return exafmm::complex_t(*this) - exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator-(const exafmm::real_t &r) const
{
  return exafmm::complex_t(*this) - r;
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator-(const int &i) const
{
  return exafmm::complex_t(*this) - exafmm::real_t(i);
}

KOKKOS_INLINE_FUNCTION cplx operator-(const exafmm::real_t &r, const cplx &z)
{
  return r - exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx operator-(const int &i, const cplx &z)
{
  return exafmm::real_t(i) - exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator*(const cplx &z) const
{
  return exafmm::complex_t(*this) * exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator*(const exafmm::real_t &r) const
{
  return exafmm::complex_t(*this) * r;
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator*(const int &i) const
{
  return exafmm::complex_t(*this) * exafmm::real_t(i);
}

KOKKOS_INLINE_FUNCTION cplx operator*(const exafmm::real_t &r, const cplx &z)
{
  return r * exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx operator*(const int &i, const cplx &z)
{
  return exafmm::real_t(i) * exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator/(const cplx &z) const
{
  return exafmm::complex_t(*this) / exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator/(const exafmm::real_t &r) const
{
  return exafmm::complex_t(*this) / r;
}

KOKKOS_INLINE_FUNCTION cplx cplx::operator/(const int &i) const
{
  return exafmm::complex_t(*this) / exafmm::real_t(i);
}

KOKKOS_INLINE_FUNCTION cplx operator/(const exafmm::real_t &r, const cplx &z)
{
  return r / exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx operator/(const int &i, const cplx &z)
{
  return exafmm::real_t(i) / exafmm::complex_t(z);
}

KOKKOS_INLINE_FUNCTION cplx sin(const cplx &z)
{
  return sin(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx sinh(const cplx &z)
{
  return sinh(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx cos(const cplx &z)
{
  return cos(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx cosh(const cplx &z)
{
  return cosh(exafmm::complex_t(z));
}

#ifdef __GNUC__ // bug in gcc ?? get segv w/egcs-2.91.66 and 2.95.2
KOKKOS_INLINE_FUNCTION cplx tan(const cplx &z)
{
  return sin(exafmm::complex_t(z)) / cos(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx tanh(const cplx &z)
{
  return sinh(exafmm::complex_t(z)) / cosh(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx log10(const cplx &z)
{
  return log(exafmm::complex_t(z)) / log(exafmm::real_t(10.));
}
#else
KOKKOS_INLINE_FUNCTION cplx tan(const cplx &z)
{
  return tan(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx tanh(const cplx &z)
{
  return tanh(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx log10(const cplx &z)
{
  return log10(exafmm::complex_t(z));
}
#endif

KOKKOS_INLINE_FUNCTION cplx log(const cplx &z)
{
  return log(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx sqrt(const cplx &z)
{
  return sqrt(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx exp(const cplx &z)
{
  return exp(exafmm::complex_t(z));
}

KOKKOS_INLINE_FUNCTION cplx pow(const cplx &a, const cplx &b)
{
  return pow(exafmm::complex_t(a), exafmm::complex_t(b));
}

KOKKOS_INLINE_FUNCTION cplx pow(const cplx &a, const exafmm::real_t &b)
{
  return pow(exafmm::complex_t(a), b);
}

KOKKOS_INLINE_FUNCTION cplx pow(const cplx &a, const int &b)
{
  return pow(exafmm::complex_t(a), exafmm::real_t(b));
}

KOKKOS_INLINE_FUNCTION cplx pow(const exafmm::real_t &a, const cplx &b)
{
  return pow(a, exafmm::complex_t(b));
}

KOKKOS_INLINE_FUNCTION cplx pow(const int &a, const cplx &b)
{
  return pow(exafmm::real_t(a), exafmm::complex_t(b));
}

KOKKOS_INLINE_FUNCTION cplx fabs(const cplx &z)
{
  return (real(z) < 0.0) ? -z : z;
}

#define surr_TEENY (1.e-24) /* machine zero compared to nominal magnitude of \
             the real part */

KOKKOS_INLINE_FUNCTION cplx asin(const cplx &z)
{
  // derivative trouble if imag(z) = +/- 1.0
  return cplx(asin(real(z)), imag(z) / sqrt(1.0 - real(z) * real(z) + surr_TEENY));
}

KOKKOS_INLINE_FUNCTION cplx acos(const cplx &z)
{
  // derivative trouble if imag(z) = +/- 1.0
  return cplx(acos(real(z)), -imag(z) / sqrt(1.0 - real(z) * real(z) + surr_TEENY));
}

#undef surr_TEENY

KOKKOS_INLINE_FUNCTION cplx atan(const cplx &z)
{
  return cplx(atan(real(z)), imag(z) / (1.0 + real(z) * real(z)));
}

KOKKOS_INLINE_FUNCTION cplx atan2(const cplx &z1, const cplx &z2)
{
  return cplx(atan2(real(z1), real(z2)),
              (real(z2) * imag(z1) - real(z1) * imag(z2)) / (real(z1) * real(z1) + real(z2) * real(z2)));
}

KOKKOS_INLINE_FUNCTION cplx ceil(const cplx &z)
{
  return cplx(ceil(real(z)), 0.);
}

KOKKOS_INLINE_FUNCTION cplx floor(const cplx &z)
{
  return cplx(floor(real(z)), 0.);
}

KOKKOS_INLINE_FUNCTION cplx ldexp(const cplx &z, const int &i)
{
  return cplx(ldexp(real(z), i), ldexp(imag(z), i));
}

// C^2 multicomplex variable. See notes on 20170915 notebook.
struct multicomplex {
  cplx A; //!< multicomplex "Real"
  cplx B; //!< multicomplex "Imaginary" (j)
};

KOKKOS_FUNCTION multicomplex init_multicomplex(cplx A, cplx B)
{
  multicomplex out;
  out.A = A;
  out.B = B;
  return out;
}

KOKKOS_INLINE_FUNCTION multicomplex init_from_C1(exafmm::complex_t C)
{
  multicomplex out;
  out.A = cplx(Kokkos::real(C), 0);
  out.B = cplx(imag(C), 0);
  return out;
}
KOKKOS_FUNCTION multicomplex product(multicomplex V, multicomplex W)
{
  return init_multicomplex(V.A * W.A - V.B * W.B, V.A * W.B + V.B * W.A);
}
KOKKOS_FUNCTION multicomplex product(multicomplex V, cplx X)
{
  return init_multicomplex(V.A * X, V.B * X);
}
KOKKOS_FUNCTION multicomplex product(multicomplex V, exafmm::real_t x)
{
  return init_multicomplex(V.A * x, V.B * x);
}
// e^{jX} in C^2 where X in C^1
KOKKOS_FUNCTION multicomplex multi_exp(cplx X)
{
  return init_multicomplex(cos(X), sin(X));
}
KOKKOS_FUNCTION multicomplex conjugate(multicomplex V)
{
  return init_multicomplex(V.A, -V.B);
}
KOKKOS_FUNCTION exafmm::complex_t Re1(multicomplex V)
{
  return exafmm::complex_t(exafmm::real_t(V.A), exafmm::real_t(V.B));
}
KOKKOS_FUNCTION exafmm::complex_t Im1(multicomplex V)
{
  return exafmm::complex_t(imag(V.A), imag(V.B));
}

#endif
