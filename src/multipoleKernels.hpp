#ifndef MULTIPOLEKERNELS_HPP
#define MULTIPOLEKERNELS_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include "arrayReduction.hpp"
#include "complexify.hpp"
#include "exafmmTypes.hpp"
namespace exafmm {
//!< L2 norm of vector X
template <class T>
KOKKOS_INLINE_FUNCTION T norm(T *X)
{
  return X[0] * X[0] + X[1] * X[1] + X[2] * X[2]; // L2 norm
}
//! Odd or even
KOKKOS_INLINE_FUNCTION int oddOrEven(int n)
{
  return (((n) & 1) == 1) ? -1 : 1;
}

//! i^2n
KOKKOS_INLINE_FUNCTION int ipow2n(int n)
{
  return (n >= 0) ? 1 : oddOrEven(n); // i^2n
}

KOKKOS_INLINE_FUNCTION void cart2sph(real_t *dX, real_t &r, real_t &theta, real_t &phi)
{
  r     = sqrt(norm(dX));                                         // r = sqrt(x^2 + y^2 + z^2)
  theta = real_t(r) == 0.0 ? real_t(0) : Kokkos::acos(dX[2] / r); // theta = acos(z / r)
  phi   = atan2(dX[1], dX[0]);                                    // phi = atan(y / x)
}

KOKKOS_INLINE_FUNCTION void cart2sph(cplx *dX, cplx &r, cplx &theta, cplx &phi)
{
  r     = sqrt(norm(dX));                               // r = sqrt(x^2 + y^2 + z^2)
  theta = real_t(r) == 0.0 ? cplx(0) : acos(dX[2] / r); // theta = acos(z / r)
  phi   = atan2(dX[1], dX[0]);                          // phi = atan(y / x)
}

template <class T1, class T2>
KOKKOS_INLINE_FUNCTION void sph2cart(T1 r, T1 theta, T1 phi, T2 *spherical, T2 *cartesian)
{
  cartesian[0] = Kokkos::sin(theta) * Kokkos::cos(phi) * spherical[0] // x component (not x itself)
                 + Kokkos::cos(theta) * Kokkos::cos(phi) / r * spherical[1] - Kokkos::sin(phi) / r / Kokkos::sin(theta) * spherical[2];
  cartesian[1] = Kokkos::sin(theta) * Kokkos::sin(phi) * spherical[0] // y component (not y itself)
                 + Kokkos::cos(theta) * Kokkos::sin(phi) / r * spherical[1] + Kokkos::cos(phi) / r / Kokkos::sin(theta) * spherical[2];
  cartesian[2] = Kokkos::cos(theta) * spherical[0] // z component (not z itself)
                 - Kokkos::sin(theta) / r * spherical[1];
}

//! Evaluate solid harmonics \f$ r^n Y_{n}^{m} \f$
KOKKOS_INLINE_FUNCTION void evalMultipole(real_t rho, real_t alpha, real_t beta, complex_t *Ynm, complex_t *YnmTheta)
{
  const complex_t I(0., 1.);
  real_t          x    = Kokkos::cos(alpha);                                    // x = cos(alpha)
  real_t          y    = Kokkos::sin(alpha);                                    // y = sin(alpha)
  real_t          invY = y == 0 ? 0 : 1 / y;                                    // 1 / y
  real_t          fact = 1;                                                     // Initialize 2 * m + 1
  real_t          pn   = 1;                                                     // Initialize Legendre polynomial Pn
  real_t          rhom = 1;                                                     // Initialize rho^m
  complex_t       ei   = Kokkos::exp(I * beta);                                 // exp(i * beta)
  complex_t       eim  = 1.0;                                                   // Initialize exp(i * m * beta)
  for (int m = 0; m < P; m++) {                                                 // Loop over m in Ynm
    real_t       p   = pn;                                                      //  Associated Legendre polynomial Pnm
    const size_t npn = m * m + 2 * m;                                           //  Index of Ynm for m > 0
    const size_t nmn = m * m;                                                   //  Index of Ynm for m < 0
    Ynm[npn]         = rhom * p * eim;                                          //  rho^m * Ynm for m > 0
    Ynm[nmn]         = Kokkos::conj(Ynm[npn]);                                  //  Use conjugate relation for m < 0
    real_t p1        = p;                                                       //  Pnm-1
    p                = x * (2 * m + 1) * p1;                                    //  Pnm using recurrence relation
    YnmTheta[npn]    = rhom * (p - (m + 1) * x * p1) * invY * eim;              //  theta derivative of r^n * Ynm
    rhom *= rho;                                                                //  rho^m
    real_t rhon = rhom;                                                         //  rho^n
    for (int n = m + 1; n < P; n++) {                                           //  Loop over n in Ynm
      const size_t npm = n * n + n + m;                                         //   Index of Ynm for m > 0
      const size_t nmm = n * n + n - m;                                         //   Index of Ynm for m < 0
      rhon /= -(n + m);                                                         //   Update factorial
      Ynm[npm]      = rhon * p * eim;                                           //   rho^n * Ynm
      Ynm[nmm]      = Kokkos::conj(Ynm[npm]);                                   //   Use conjugate relation for m < 0
      real_t p2     = p1;                                                       //   Pnm-2
      p1            = p;                                                        //   Pnm-1
      p             = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);      //   Pnm using recurrence relation
      YnmTheta[npm] = rhon * ((n - m + 1) * p - (n + 1) * x * p1) * invY * eim; // theta derivative
      rhon *= rho;                                                              //   Update rho^n
    } //  End loop over n in Ynm
    rhom /= -(2 * m + 2) * (2 * m + 1); //  Update factorial
    pn = -pn * fact * y;                //  Pn
    fact += 2;                          //  2 * m + 1
    eim *= ei;                          //  Update exp(i * m * beta)
  } // End loop over m in Ynm
}

KOKKOS_INLINE_FUNCTION void evalMultipole(cplx rho, cplx alpha, cplx beta, multicomplex *Ynm, multicomplex *YnmTheta)
{
  const complex_t    I(0., 1.);
  const double       EPS  = 1e-16;
  const cplx         x    = cos(alpha);
  const cplx         y    = sin(alpha);
  const cplx         invY = (real_t(sin(alpha)) <= EPS) ? cplx(0, 0) : 1 / y;
  cplx               fact = 1;
  cplx               pn   = 1;
  cplx               rhom = 1;
  const multicomplex ei   = multi_exp(beta);
  multicomplex       eim  = init_multicomplex(1.0, 0.0);
  for (int m = 0; m < P; m++) {
    cplx p        = pn;
    int  npn      = m * m + 2 * m;
    int  nmn      = m * m;
    Ynm[npn]      = product(eim, rhom * p);
    Ynm[nmn]      = conjugate(Ynm[npn]);
    cplx p1       = p;
    p             = x * (2 * m + 1) * p1;
    YnmTheta[npn] = product(eim, rhom * (p - (m + 1) * x * p1) * invY);
    rhom *= rho;
    cplx rhon = rhom;
    for (int n = m + 1; n < P; n++) {
      int npm = n * n + n + m;
      int nmm = n * n + n - m;
      rhon /= -(n + m);
      Ynm[npm]      = product(eim, rhon * p);
      Ynm[nmm]      = conjugate(Ynm[npm]);
      cplx p2       = p1;
      p1            = p;
      p             = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      YnmTheta[npm] = product(eim, rhon * ((n - m + 1) * p - (n + 1) * x * p1) * invY);
      rhon *= rho;
    }
    rhom /= -(2 * m + 2) * (2 * m + 1);
    pn = -pn * fact * y;
    fact += 2;
    eim = product(eim, ei);
  }
}

//! Evaluate singular harmonics \f$ r^{-n-1} Y_n^m \f$
KOKKOS_INLINE_FUNCTION void evalLocal(real_t rho, real_t alpha, real_t beta, complex_t *Ynm)
{
  const complex_t I(0., 1.);
  const real_t    x    = Kokkos::cos(alpha);                           // x = cos(alpha)
  const real_t    y    = Kokkos::sin(alpha);                           // y = sin(alpha)
  real_t          fact = 1;                                            // Initialize 2 * m + 1
  real_t          pn   = 1;                                            // Initialize Legendre polynomial Pn
  real_t          invR = -1.0 / rho;                                   // - 1 / rho
  real_t          rhom = -invR;                                        // Initialize rho^(-m-1)
  const complex_t ei   = Kokkos::exp(I * beta);                        // exp(i * beta)
  complex_t       eim  = 1.0;                                          // Initialize exp(i * m * beta)
  for (int m = 0; m < P; m++) {                                        // Loop over m in Ynm
    real_t p   = pn;                                                   //  Associated Legendre polynomial Pnm
    int    npn = m * m + 2 * m;                                        //  Index of Ynm for m > 0
    int    nmn = m * m;                                                //  Index of Ynm for m < 0
    Ynm[npn]   = rhom * p * eim;                                       //  rho^(-m-1) * Ynm for m > 0
    Ynm[nmn]   = Kokkos::conj(Ynm[npn]);                               //  Use conjugate relation for m < 0
    real_t p1  = p;                                                    //  Pnm-1
    p          = x * (2 * m + 1) * p1;                                 //  Pnm using recurrence relation
    rhom *= invR;                                                      //  rho^(-m-1)
    real_t rhon = rhom;                                                //  rho^(-n-1)
    for (int n = m + 1; n < P; n++) {                                  //  Loop over n in Ynm
      int npm   = n * n + n + m;                                       //   Index of Ynm for m > 0
      int nmm   = n * n + n - m;                                       //   Index of Ynm for m < 0
      Ynm[npm]  = rhon * p * eim;                                      //   rho^n * Ynm for m > 0
      Ynm[nmm]  = Kokkos::conj(Ynm[npm]);                              //   Use conjugate relation for m < 0
      real_t p2 = p1;                                                  //   Pnm-2
      p1        = p;                                                   //   Pnm-1
      p         = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1); //   Pnm using recurrence relation
      rhon *= invR * (n - m + 1);                                      //   rho^(-n-1)
    } //  End loop over n in Ynm
    pn = -pn * fact * y; //  Pn
    fact += 2;           //  2 * m + 1
    eim *= ei;           //  Update exp(i * m * beta)
  } // End loop over m in Ynm
}

void P2P(Cell *Ci, Cell *Cj)
{
  real_t    dX[3];
  Body     *Bi = Ci->BODY;
  Body     *Bj = Cj->BODY;
  const int ni = Ci->NBODY;
  const int nj = Cj->NBODY;
  for (int i = 0; i < ni; i++) {
    real_t ax = 0;
    real_t ay = 0;
    real_t az = 0;
    real_t sx = 0;
    real_t sy = 0;
    real_t sz = 0;
    for (int j = 0; j < nj; j++) {
      for (int d = 0; d < 3; d++)
        dX[d] = Bi[i].X[d] - Bj[j].X[d];
      real_t R2 = norm(dX);
      if (R2 != 0) {
        real_t S2     = 2 * Bj[j].radius * Bj[j].radius; //    2 * sigma^2
        real_t RS     = R2 / S2;                         //    R^2 / (2 * simga^2)
        real_t cutoff = 0.25 / M_PI / R2 / std::sqrt(R2) * (erf(std::sqrt(RS)) - std::sqrt(4 / M_PI * RS) * exp(-RS));
        ax += (dX[1] * Bj[j].alpha[2] - dX[2] * Bj[j].alpha[1]) * cutoff; // x component of curl G * cutoff
        ay += (dX[2] * Bj[j].alpha[0] - dX[0] * Bj[j].alpha[2]) * cutoff; // y component of curl G * cutoff
        az += (dX[0] * Bj[j].alpha[1] - dX[1] * Bj[j].alpha[0]) * cutoff; // z component of curl G * cutoff

        sx += (Bi[i].alpha[1] * Bj[j].alpha[2] - Bi[i].alpha[2] * Bj[j].alpha[1]) * cutoff; // x component of first term
        sy += (Bi[i].alpha[2] * Bj[j].alpha[0] - Bi[i].alpha[0] * Bj[j].alpha[2]) * cutoff; // y component of first term
        sz += (Bi[i].alpha[0] * Bj[j].alpha[1] - Bi[i].alpha[1] * Bj[j].alpha[0]) * cutoff; // z component of first term

        cutoff = 0.25 / M_PI / R2 / R2 / std::sqrt(R2) * (3 * erf(std::sqrt(RS)) - (2 * RS + 3) * std::sqrt(4 / M_PI * RS) * exp(-RS)) *
                 (Bi[i].alpha[0] * dX[0] + Bi[i].alpha[1] * dX[1] + Bi[i].alpha[2] * dX[2]); // cutoff function for second term

        sx += (Bj[j].alpha[1] * dX[2] - Bj[j].alpha[2] * dX[1]) * cutoff; // x component of second term

        sy += (Bj[j].alpha[2] * dX[0] - Bj[j].alpha[0] * dX[2]) * cutoff; // y component of second term

        sz += (Bj[j].alpha[0] * dX[1] - Bj[j].alpha[1] * dX[0]) * cutoff; // z component of second term
      }
    }
    Bi[i].velocity[0] -= ax;
    Bi[i].velocity[1] -= ay;
    Bi[i].velocity[2] -= az;
    Bi[i].dadt[0] -= sx;
    Bi[i].dadt[1] -= sy;
    Bi[i].dadt[2] -= sz;
  }
}

template <typename ParticleView, typename CellView>
KOKKOS_INLINE_FUNCTION void P2P(const size_t ii, const size_t jj, ParticleView bodyView, CellView cellView, const Kokkos::TeamPolicy<>::member_type &teamMember)
{

  typedef sample::array_type<real_t, 12>                                ValueType;
  typedef sample::SumMyArray<real_t, Kokkos::DefaultExecutionSpace, 12> ArraySumResult;

  const size_t iStart = cellView(ii).BodyOffset;
  const size_t jStart = cellView(jj).BodyOffset;
  const size_t iEnd   = cellView(ii).NBODY + cellView(ii).BodyOffset;
  const size_t jEnd   = cellView(jj).NBODY + cellView(jj).BodyOffset;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, iStart, iEnd), [&](const size_t iIndex) {
    ValueType sum;
    real_t    dX[3];
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, jStart, jEnd), [&](size_t jIndex, ValueType &localSum) {
          for (int d = 0; d < 3; d++)
            dX[d] = bodyView(iIndex).X[d] - bodyView(jIndex).X[d];
          const real_t R2 = norm(dX);
          if (R2 != 0) {
            const real_t S2        =  bodyView(jIndex).radius * bodyView(jIndex).radius; //     sigma^2
            const real_t RS        = R2 / S2/2.0;                                                 //    R^2 / (2 * simga^2)
            const real_t sqrtRS    = Kokkos::sqrt(RS);
            const real_t erfSqrtRS = Kokkos::erf(sqrtRS);
            const real_t expmRS    = Kokkos::exp(-RS);
            const real_t invSqrtR2 = 1.0 / Kokkos::sqrt(R2);
            const real_t t2        = 2.0 * Kokkos::numbers::inv_sqrtpi * sqrtRS * expmRS;
            const real_t one_over_4piR3 = 0.25 * Kokkos::numbers::inv_pi / R2 * invSqrtR2;
            const auto cross1 = one_over_4piR3 * (dX[1] * bodyView(jIndex).alpha[2] - dX[2] * bodyView(jIndex).alpha[1]);
            const auto cross2 = one_over_4piR3 * (dX[2] * bodyView(jIndex).alpha[0] - dX[0] * bodyView(jIndex).alpha[2]);
            const auto cross3 = one_over_4piR3 * (dX[0] * bodyView(jIndex).alpha[1] - dX[1] * bodyView(jIndex).alpha[0]);
            const auto g_gsm  = (erfSqrtRS - t2);

            localSum.myArray[0] += cross1* g_gsm; // x component of curl G * cutoff
            localSum.myArray[1] += cross2 * g_gsm; // y component of curl G * cutoff
            localSum.myArray[2] += cross3* g_gsm; // z component of curl G * cutoff

             real_t aux = t2 / S2 -3.0* g_gsm / R2 ;

            localSum.myArray[3] += aux * cross1 * dX[0];
            localSum.myArray[4] += aux * cross1 * dX[1];
            localSum.myArray[5] += aux * cross1 * dX[2];

            localSum.myArray[6] += aux * cross2 * dX[0];
            localSum.myArray[7] += aux * cross2 * dX[1];
            localSum.myArray[8] += aux * cross2 * dX[2];

            localSum.myArray[9] += aux * cross3 * dX[0];
            localSum.myArray[10] += aux * cross3 * dX[1];
            localSum.myArray[11] += aux * cross3 * dX[2];

            aux = g_gsm *one_over_4piR3;

            localSum.myArray[6] -= aux * bodyView(jIndex).alpha[2];
            localSum.myArray[9] += aux * bodyView(jIndex).alpha[1];

            localSum.myArray[4] += aux * bodyView(jIndex).alpha[2];
            localSum.myArray[10] -= aux * bodyView(jIndex).alpha[0];

            localSum.myArray[5] -= aux * bodyView(jIndex).alpha[1];
            localSum.myArray[8] += aux * bodyView(jIndex).alpha[0];
            
          } }, ArraySumResult(sum));
    Kokkos::single(Kokkos::PerThread(teamMember), [&]() {
      Kokkos::atomic_add(&bodyView(iIndex).velocity[0], -sum.myArray[0]);
      Kokkos::atomic_add(&bodyView(iIndex).velocity[1], -sum.myArray[1]);
      Kokkos::atomic_add(&bodyView(iIndex).velocity[2], -sum.myArray[2]);
      Kokkos::atomic_add(&bodyView(iIndex).J[0][0], -sum.myArray[3]);
      Kokkos::atomic_add(&bodyView(iIndex).J[0][1], -sum.myArray[4]);
      Kokkos::atomic_add(&bodyView(iIndex).J[0][2], -sum.myArray[5]);
      Kokkos::atomic_add(&bodyView(iIndex).J[1][0], -sum.myArray[6]);
      Kokkos::atomic_add(&bodyView(iIndex).J[1][1], -sum.myArray[7]);
      Kokkos::atomic_add(&bodyView(iIndex).J[1][2], -sum.myArray[8]);
      Kokkos::atomic_add(&bodyView(iIndex).J[2][0], -sum.myArray[9]);
      Kokkos::atomic_add(&bodyView(iIndex).J[2][1], -sum.myArray[10]);
      Kokkos::atomic_add(&bodyView(iIndex).J[2][2], -sum.myArray[11]);
    });
  });
}

template <typename ParticleView, typename CellView>
KOKKOS_INLINE_FUNCTION void P2P(const size_t ii, const size_t jj, ParticleView bodyView, ParticleView sensorView, CellView cellView, CellView sensorCells, const Kokkos::TeamPolicy<>::member_type &teamMember)
{

  typedef sample::array_type<real_t, 3>                                ValueType;
  typedef sample::SumMyArray<real_t, Kokkos::DefaultExecutionSpace, 3> ArraySumResult;

  const size_t iStart = sensorCells(ii).BodyOffset;
  const size_t jStart = cellView(jj).BodyOffset;
  const size_t iEnd   = sensorCells(ii).NBODY + sensorCells(ii).BodyOffset;
  const size_t jEnd   = cellView(jj).NBODY + cellView(jj).BodyOffset;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, iStart, iEnd), [&](const size_t iIndex) {
    ValueType sum;
    real_t    dX[3];
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, jStart, jEnd), [&](size_t jIndex, ValueType &localSum) {
          for (int d = 0; d < 3; d++)
            dX[d] = sensorView(iIndex).X[d] - bodyView(jIndex).X[d];
          const real_t R2 = norm(dX);
          if (R2 != 0) {
            const real_t S2        = 2.0 * bodyView(jIndex).radius * bodyView(jIndex).radius; //    2 * sigma^2
            const real_t RS        = R2 / S2;                                                 //    R^2 / (2 * simga^2)
            const real_t sqrtRS    = Kokkos::sqrt(RS);
            const real_t erfSqrtRS = Kokkos::erf(sqrtRS);
            const real_t expmRS    = Kokkos::exp(-RS);
            const real_t invSqrtR2 = 1.0 / Kokkos::sqrt(R2);
            const real_t t2        = 2.0 * Kokkos::numbers::inv_sqrtpi * sqrtRS * expmRS;
            real_t       cutoff    = 0.25 * Kokkos::numbers::inv_pi / R2 * invSqrtR2 * (erfSqrtRS - t2);
            localSum.myArray[0] += (dX[1] * bodyView(jIndex).alpha[2] - dX[2] * bodyView(jIndex).alpha[1]) * cutoff; // x component of curl G * cutoff
            localSum.myArray[1] += (dX[2] * bodyView(jIndex).alpha[0] - dX[0] * bodyView(jIndex).alpha[2]) * cutoff; // y component of curl G * cutoff
            localSum.myArray[2] += (dX[0] * bodyView(jIndex).alpha[1] - dX[1] * bodyView(jIndex).alpha[0]) * cutoff; // z component of curl G * cutoff
          } }, ArraySumResult(sum));
    Kokkos::single(Kokkos::PerThread(teamMember), [&]() {
      Kokkos::atomic_add(&sensorView(iIndex).velocity[0], -sum.myArray[0]);
      Kokkos::atomic_add(&sensorView(iIndex).velocity[1], -sum.myArray[1]);
      Kokkos::atomic_add(&sensorView(iIndex).velocity[2], -sum.myArray[2]);
    });
  });
}

template <typename ParticleView, typename CellView>
KOKKOS_INLINE_FUNCTION void P2P_vorticity(const size_t ii, const size_t jj, ParticleView bodyView, ParticleView sensorView, CellView cellView, CellView sensorCells, const Kokkos::TeamPolicy<>::member_type &teamMember)
{

  typedef sample::array_type<real_t, 3>                                ValueType;
  typedef sample::SumMyArray<real_t, Kokkos::DefaultExecutionSpace, 3> ArraySumResult;

  const size_t iStart = sensorCells(ii).BodyOffset;
  const size_t jStart = cellView(jj).BodyOffset;
  const size_t iEnd   = sensorCells(ii).NBODY + sensorCells(ii).BodyOffset;
  const size_t jEnd   = cellView(jj).NBODY + cellView(jj).BodyOffset;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, iStart, iEnd), [&](const size_t iIndex) {
    ValueType sum;
    real_t    dX[3];
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, jStart, jEnd), [&](size_t jIndex, ValueType &localSum) {
          for (int d = 0; d < 3; d++)
            dX[d] = sensorView(iIndex).X[d] - bodyView(jIndex).X[d];
          const real_t R2 = norm(dX);
            const real_t S2        = 2.0 * bodyView(jIndex).radius * bodyView(jIndex).radius; //    2 * sigma^2
            const real_t zeta_sigma =  Kokkos::exp(-R2 / S2) *Kokkos::numbers::inv_pi * Kokkos::numbers::inv_sqrtpi / Kokkos::numbers::sqrt2 / S2 / bodyView(jIndex).radius;
            localSum.myArray[0] += bodyView(jIndex).alpha[0] * zeta_sigma;
            localSum.myArray[1] += bodyView(jIndex).alpha[1] * zeta_sigma;
            localSum.myArray[2] += bodyView(jIndex).alpha[2] * zeta_sigma; }, ArraySumResult(sum));
    Kokkos::single(Kokkos::PerThread(teamMember), [&]() {
      Kokkos::atomic_add(&sensorView(iIndex).velocity_old[0], -sum.myArray[0]);
      Kokkos::atomic_add(&sensorView(iIndex).velocity_old[1], -sum.myArray[1]);
      Kokkos::atomic_add(&sensorView(iIndex).velocity_old[2], -sum.myArray[2]);
    });
  });
}

template <typename ParticleView, typename CellView>
KOKKOS_INLINE_FUNCTION void P2P_vorticity(const size_t ii, const size_t jj, ParticleView bodyView, CellView cellView, const Kokkos::TeamPolicy<>::member_type &teamMember)
{

  typedef sample::array_type<real_t, 3>                                ValueType;
  typedef sample::SumMyArray<real_t, Kokkos::DefaultExecutionSpace, 3> ArraySumResult;

  const size_t iStart = cellView(ii).BodyOffset;
  const size_t jStart = cellView(jj).BodyOffset;
  const size_t iEnd   = cellView(ii).NBODY + cellView(ii).BodyOffset;
  const size_t jEnd   = cellView(jj).NBODY + cellView(jj).BodyOffset;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, iStart, iEnd), [&](const size_t iIndex) {
    ValueType sum;
    real_t    dX[3];
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, jStart, jEnd), [&](size_t jIndex, ValueType &localSum) {
          for (int d = 0; d < 3; d++)
            dX[d] = bodyView(iIndex).X[d] - bodyView(jIndex).X[d];
          const real_t R2 = norm(dX);
            const real_t S2        = 2.0 * bodyView(jIndex).radius * bodyView(jIndex).radius; //    2 * sigma^2
            const real_t zeta_sigma =  Kokkos::exp(-R2 / S2) *Kokkos::numbers::inv_pi * Kokkos::numbers::inv_sqrtpi / Kokkos::numbers::sqrt2 / S2 / bodyView(jIndex).radius;
            localSum.myArray[0] += bodyView(jIndex).alpha[0] * zeta_sigma;
            localSum.myArray[1] += bodyView(jIndex).alpha[1] * zeta_sigma;
            localSum.myArray[2] += bodyView(jIndex).alpha[2] * zeta_sigma; }, ArraySumResult(sum));
    Kokkos::single(Kokkos::PerThread(teamMember), [&]() {
      Kokkos::atomic_add(&bodyView(iIndex).velocity[0], -sum.myArray[0]);
      Kokkos::atomic_add(&bodyView(iIndex).velocity[1], -sum.myArray[1]);
      Kokkos::atomic_add(&bodyView(iIndex).velocity[2], -sum.myArray[2]);
    });
  });
}

/*
Symmetric PSE Kernel from

A Viscous Vortex Particle Model for Rotor Wake and Interference Analysis
Jinggen Zhao and Chengjian He
DOI : 10.4050/JAHS.55.012007
*/
template <typename ParticleView, typename CellView>
KOKKOS_INLINE_FUNCTION void PSE(const size_t ii, const size_t jj, ParticleView bodyView, CellView cellView, const real_t nu, const Kokkos::TeamPolicy<>::member_type &teamMember)
{

  typedef sample::array_type<real_t, 3>                                ValueType;
  typedef sample::SumMyArray<real_t, Kokkos::DefaultExecutionSpace, 3> ArraySumResult;

  const size_t iStart = cellView(ii).BodyOffset;
  const size_t jStart = cellView(jj).BodyOffset;
  const size_t iEnd   = cellView(ii).NBODY + cellView(ii).BodyOffset;
  const size_t jEnd   = cellView(jj).NBODY + cellView(jj).BodyOffset;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, iStart, iEnd), [&](const size_t iIndex) {
    ValueType sum;
    real_t    dX[3];
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, jStart, jEnd), [&](size_t jIndex, ValueType &localSum) {
          for (int d = 0; d < 3; d++)
            dX[d] = bodyView(iIndex).X[d] - bodyView(jIndex).X[d];
          const real_t R2 = norm(dX);
          if (R2 != 0) {
            const real_t one_over_sigma_ij2 = 1. / (bodyView(iIndex).radius*bodyView(iIndex).radius + bodyView(jIndex).radius* bodyView(jIndex).radius);
            const real_t sigma_power_5 = one_over_sigma_ij2 *one_over_sigma_ij2 * Kokkos::sqrt(one_over_sigma_ij2);
            const real_t vol_i = 4./3. *  Kokkos::numbers::pi * bodyView(iIndex).radius*bodyView(iIndex).radius*bodyView(iIndex).radius;
            const real_t vol_j =  4./3. *  Kokkos::numbers::pi * bodyView(jIndex).radius*bodyView(jIndex).radius*bodyView(jIndex).radius;
            const real_t exp_term = Kokkos::exp(-0.5 *R2 * one_over_sigma_ij2);
            localSum.myArray[0] += (vol_i * bodyView(jIndex).alpha[0] - vol_j * bodyView(iIndex).alpha[0]) * exp_term*sigma_power_5; 
            localSum.myArray[1] += (vol_i * bodyView(jIndex).alpha[1] - vol_j * bodyView(iIndex).alpha[1]) * exp_term*sigma_power_5; 
            localSum.myArray[2] += (vol_i * bodyView(jIndex).alpha[2] - vol_j * bodyView(iIndex).alpha[2]) * exp_term*sigma_power_5; 
          } }, ArraySumResult(sum));
    Kokkos::single(Kokkos::PerThread(teamMember), [&]() {
      const real_t prefix = nu / Kokkos::numbers::sqrt2 * Kokkos::numbers::inv_pi * Kokkos::numbers::inv_sqrtpi;
      Kokkos::atomic_add(&bodyView(iIndex).dadt[0], sum.myArray[0] * prefix);
      Kokkos::atomic_add(&bodyView(iIndex).dadt[1], sum.myArray[1] * prefix);
      Kokkos::atomic_add(&bodyView(iIndex).dadt[2], sum.myArray[2] * prefix);
    });
  });
}

template <typename CellView, typename ParticleView, typename MultipoleView>
KOKKOS_INLINE_FUNCTION void P2M(size_t cellIndex, CellView cellView, ParticleView bodyView, MultipoleView multipoleView)
{
  real_t     dX[3];
  complex_t  Ynm[P * P], YnmTheta[P * P];
  const auto startindex = cellView(cellIndex).BodyOffset;
  const auto nbody      = cellView(cellIndex).NBODY;
  for (size_t index = startindex; index < startindex + nbody; index++) {
    {
      for (int d = 0; d < 3; d++)
        dX[d] = bodyView(index).X[d] - cellView(cellIndex).X[d];
      real_t rho, alpha, beta;
      cart2sph(dX, rho, alpha, beta);
      evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
      for (int n = 0; n < P; n++) {
        for (int m = 0; m <= n; m++) {
          const int nm  = n * n + n - m;
          const int nms = n * (n + 1) / 2 + m;
          for (int d = 0; d < 3; d++) {
            Kokkos::atomic_add(&multipoleView(cellIndex, 3 * nms + d), bodyView(index).alpha[d] * Ynm[nm]);
          }
        }
      }
    }
  }
}

template <typename CellView, typename MultipoleView>
KOKKOS_INLINE_FUNCTION void M2M(size_t cellIndex, CellView cellView, MultipoleView multipoleView)
{
  complex_t Ynm[P * P], YnmTheta[P * P];
  real_t    dX[3];

  const size_t start  = cellView(cellIndex).ChildOffset;
  const size_t nChild = cellView(cellIndex).NCHILD;
  for (size_t childIndex = start; childIndex < start + nChild; childIndex++) {
    for (int d = 0; d < 3; d++)
      dX[d] = cellView(cellIndex).X[d] - cellView(childIndex).X[d];
    real_t rho, alpha, beta;
    cart2sph(dX, rho, alpha, beta);
    evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
    for (int j = 0; j < P; j++) {
      for (int k = 0; k <= j; k++) {
        int       jks = j * (j + 1) / 2 + k;
        complex_t M[3]{0, 0, 0};
        for (int n = 0; n <= j; n++) {
          for (int m = Kokkos::max(-n, -j + k + n); m <= Kokkos::min(k - 1, n); m++) {
            int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
            int nm    = n * n + n - m;
            for (int d = 0; d < 3; d++)
              M[d] += multipoleView(childIndex, 3 * jnkms + d) * Ynm[nm] * real_t(ipow2n(m) * oddOrEven(n));
          }
          for (int m = k; m <= Kokkos::min(n, j + k - n); m++) {
            int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
            int nm    = n * n + n - m;
            for (int d = 0; d < 3; d++)
              M[d] += Kokkos::conj(multipoleView(childIndex, 3 * jnkms + d)) * Ynm[nm] * real_t(oddOrEven(k + n + m));
          }
        }
        for (int d = 0; d < 3; d++)
          Kokkos::atomic_add(&multipoleView(cellIndex, 3 * jks + d), M[d]);
      }
    }
  }
}

template <typename CellView, typename MultipoleView, typename LocalView>
KOKKOS_INLINE_FUNCTION void M2L(const size_t i, const size_t jj, CellView cellView, MultipoleView multipoleView, LocalView localView)
{
  complex_t Ynm2[4 * P * P];
  real_t    dX[3];
  for (int d = 0; d < 3; d++)
    dX[d] = cellView(i).X[d] - cellView(jj).X[d];
  real_t rho, alpha, beta;
  cart2sph(dX, rho, alpha, beta);
  evalLocal(rho, alpha, beta, Ynm2);
  for (int j = 0; j < P; j++) {
    real_t Cnm = oddOrEven(j);
    for (int k = 0; k <= j; k++) {
      int       jks = j * (j + 1) / 2 + k;
      complex_t L[3]{0, 0, 0};
      for (int n = 0; n < P; n++) {
        for (int m = -n; m < 0; m++) {
          int nms  = n * (n + 1) / 2 - m;
          int jnkm = (j + n) * (j + n) + j + n + m - k;
          for (int d = 0; d < 3; d++)
            L[d] += Kokkos::conj(multipoleView(jj, 3 * nms + d)) * Cnm * Ynm2[jnkm];
        }
        for (int m = 0; m <= n; m++) {
          int    nms  = n * (n + 1) / 2 + m;
          int    jnkm = (j + n) * (j + n) + j + n + m - k;
          real_t Cnm2 = Cnm * oddOrEven((k - m) * (k < m) + m);
          for (int d = 0; d < 3; d++)
            L[d] += multipoleView(jj, 3 * nms + d) * Cnm2 * Ynm2[jnkm];
        }
      }
      for (int d = 0; d < 3; d++)
        Kokkos::atomic_add(&localView(i, 3 * jks + d), L[d]);
    }
  }
}

template <typename CellView, typename MultipoleView, typename LocalView>
KOKKOS_INLINE_FUNCTION void M2L(const size_t i, const size_t jj, CellView cellView, CellView cellViewSensors, MultipoleView multipoleView, LocalView localView)
{
  complex_t Ynm2[4 * P * P];
  real_t    dX[3];
  for (int d = 0; d < 3; d++)
    dX[d] = cellViewSensors(i).X[d] - cellView(jj).X[d];
  real_t rho, alpha, beta;
  cart2sph(dX, rho, alpha, beta);
  evalLocal(rho, alpha, beta, Ynm2);
  for (int j = 0; j < P; j++) {
    real_t Cnm = oddOrEven(j);
    for (int k = 0; k <= j; k++) {
      int       jks = j * (j + 1) / 2 + k;
      complex_t L[3]{0, 0, 0};
      for (int n = 0; n < P; n++) {
        for (int m = -n; m < 0; m++) {
          int nms  = n * (n + 1) / 2 - m;
          int jnkm = (j + n) * (j + n) + j + n + m - k;
          for (int d = 0; d < 3; d++)
            L[d] += Kokkos::conj(multipoleView(jj, 3 * nms + d)) * Cnm * Ynm2[jnkm];
        }
        for (int m = 0; m <= n; m++) {
          int    nms  = n * (n + 1) / 2 + m;
          int    jnkm = (j + n) * (j + n) + j + n + m - k;
          real_t Cnm2 = Cnm * oddOrEven((k - m) * (k < m) + m);
          for (int d = 0; d < 3; d++) {
            L[d] += multipoleView(jj, 3 * nms + d) * Cnm2 * Ynm2[jnkm];
          }
        }
      }
      for (int d = 0; d < 3; d++)
        Kokkos::atomic_add(&localView(i, 3 * jks + d), L[d]);
    }
  }
}

template <typename CellView, typename LocalView>
KOKKOS_INLINE_FUNCTION void L2L(const size_t jIndex, CellView cellView, LocalView localView)
{
  complex_t Ynm[P * P], YnmTheta[P * P];
  real_t    dX[3];
  for (size_t i = 0; i < cellView(jIndex).NCHILD; i++) {
    const size_t iIndex = cellView(jIndex).ChildOffset + i;
    for (int d = 0; d < 3; d++)
      dX[d] = cellView(iIndex).X[d] - cellView(jIndex).X[d];
    real_t rho, alpha, beta;
    cart2sph(dX, rho, alpha, beta);
    evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
    for (int j = 0; j < P; j++) {
      for (int k = 0; k <= j; k++) {
        const int jks = j * (j + 1) / 2 + k;
        complex_t L[3]{0, 0, 0};
        for (int n = j; n < P; n++) {
          for (int m = j + k - n; m < 0; m++) {
            const int jnkm = (n - j) * (n - j) + n - j + m - k;
            const int nms  = n * (n + 1) / 2 - m;
            for (int d = 0; d < 3; d++)
              L[d] += Kokkos::conj(localView(jIndex, 3 * nms + d)) * Ynm[jnkm] * real_t(oddOrEven(k));
          }
          for (int m = 0; m <= n; m++) {
            if (n - j >= abs(m - k)) {
              const int jnkm = (n - j) * (n - j) + n - j + m - k;
              const int nms  = n * (n + 1) / 2 + m;
              for (int d = 0; d < 3; d++)
                L[d] += localView(jIndex, 3 * nms + d) * Ynm[jnkm] * real_t(oddOrEven((m - k) * (m < k)));
            }
          }
        }
        for (int d = 0; d < 3; d++)
          Kokkos::atomic_add(&localView(iIndex, 3 * jks + d), L[d]);
      }
    }
  }
}

template <typename CellView, typename LocalView, typename ParticleView>
KOKKOS_INLINE_FUNCTION void L2P(const size_t cellIndex, CellView cellView, LocalView localView, ParticleView bodyView)
{
  const complex_t I(0., 1.); //!< Imaginary unit
  multicomplex    Ynm[P * P], YnmTheta[P * P];
  const real_t    COMPLEX_STEP = 1e-32;
  const size_t    start        = cellView(cellIndex).BodyOffset;
  const size_t    end          = cellView(cellIndex).BodyOffset + cellView(cellIndex).NBODY;
  for (size_t bodyIndex = start; bodyIndex < end; bodyIndex++) {
    real_t velocity[3]{0.0, 0.0, 0.0};
    real_t J[3][3]{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int H_ind = 0; H_ind < 3; H_ind++) {
      cplx dX[3];
      for (int d = 0; d < 3; d++)
        dX[d] = bodyView(bodyIndex).X[d] - cellView(cellIndex).X[d];
      dX[H_ind] += cplx(0, COMPLEX_STEP);
      cplx spherical1[3];
      cplx spherical2[3];
      cplx spherical3[3];
      cplx cartesian[3];
      for (int d = 0; d < 3; d++) {
        spherical1[d] = 0;
        spherical2[d] = 0;
        spherical3[d] = 0;
        cartesian[d]  = 0;
      }
      cplx r, theta, phi;
      cart2sph(dX, r, theta, phi);
      evalMultipole(r, theta, phi, Ynm, YnmTheta);
      for (int n = 0; n < P; n++) {
        int nm  = n * n + n;
        int nms = n * (n + 1) / 2;
        spherical1[0] += (product(init_from_C1(localView(cellIndex, 3 * nms + 0)), Ynm[nm])).A / r * n;
        spherical1[1] += (product(init_from_C1(localView(cellIndex, 3 * nms + 0)), YnmTheta[nm])).A;
        spherical2[0] += (product(init_from_C1(localView(cellIndex, 3 * nms + 1)), Ynm[nm])).A / r * n;
        spherical2[1] += (product(init_from_C1(localView(cellIndex, 3 * nms + 1)), YnmTheta[nm])).A;
        spherical3[0] += (product(init_from_C1(localView(cellIndex, 3 * nms + 2)), Ynm[nm])).A / r * n;
        spherical3[1] += (product(init_from_C1(localView(cellIndex, 3 * nms + 2)), YnmTheta[nm])).A;
        for (int m = 1; m <= n; m++) {
          nm  = n * n + n + m;
          nms = n * (n + 1) / 2 + m;
          spherical1[0] += 2 * (product(init_from_C1(localView(cellIndex, 3 * nms + 0)), Ynm[nm])).A / r * n;
          spherical1[1] += 2 * (product(init_from_C1(localView(cellIndex, 3 * nms + 0)), YnmTheta[nm])).A;
          spherical1[2] += 2 * (product(product(init_from_C1(localView(cellIndex, 3 * nms + 0)), Ynm[nm]), init_from_C1(I))).A * m;
          spherical2[0] += 2 * (product(init_from_C1(localView(cellIndex, 3 * nms + 1)), Ynm[nm])).A / r * n;
          spherical2[1] += 2 * (product(init_from_C1(localView(cellIndex, 3 * nms + 1)), YnmTheta[nm])).A;
          spherical2[2] += 2 * (product(product(init_from_C1(localView(cellIndex, 3 * nms + 1)), Ynm[nm]), init_from_C1(I))).A * m;
          spherical3[0] += 2 * (product(init_from_C1(localView(cellIndex, 3 * nms + 2)), Ynm[nm])).A / r * n;
          spherical3[1] += 2 * (product(init_from_C1(localView(cellIndex, 3 * nms + 2)), YnmTheta[nm])).A;
          spherical3[2] += 2 * (product(product(init_from_C1(localView(cellIndex, 3 * nms + 2)), Ynm[nm]), init_from_C1(I))).A * m;
        }
      }
      for (int ind = 0; ind < 3; ind++)
        cartesian[ind] = 0;
      sph2cart(r, theta, phi, spherical1, cartesian);

      if (H_ind == 0) {
        velocity[2] -= real_t(cartesian[1]);
        velocity[1] += real_t(cartesian[2]);
        J[1][0] += imag(cartesian[2]) / COMPLEX_STEP;
        J[2][0] -= imag(cartesian[1]) / COMPLEX_STEP;
      } else if (H_ind == 1) {
        J[1][1] += imag(cartesian[2]) / COMPLEX_STEP;
        J[2][1] -= imag(cartesian[1]) / COMPLEX_STEP;
      } else if (H_ind == 2) {
        J[1][2] += imag(cartesian[2]) / COMPLEX_STEP;
        J[2][2] -= imag(cartesian[1]) / COMPLEX_STEP;
      }

      for (int ind = 0; ind < 3; ind++)
        cartesian[ind] = 0;
      sph2cart(r, theta, phi, spherical2, cartesian);

      if (H_ind == 0) {
        velocity[2] += real_t(cartesian[0]);
        velocity[0] -= real_t(cartesian[2]);
        J[0][0] -= imag(cartesian[2]) / COMPLEX_STEP;
        J[2][0] += imag(cartesian[0]) / COMPLEX_STEP;
      } else if (H_ind == 1) {
        J[0][1] -= imag(cartesian[2]) / COMPLEX_STEP;
        J[2][1] += imag(cartesian[0]) / COMPLEX_STEP;
      } else if (H_ind == 2) {
        J[0][2] -= imag(cartesian[2]) / COMPLEX_STEP;
        J[2][2] += imag(cartesian[0]) / COMPLEX_STEP;
      }

      for (int ind = 0; ind < 3; ind++)
        cartesian[ind] = 0;
      sph2cart(r, theta, phi, spherical3, cartesian);

      if (H_ind == 0) {
        velocity[1] -= real_t(cartesian[0]);
        velocity[0] += real_t(cartesian[1]);
        J[0][0] += imag(cartesian[1]) / COMPLEX_STEP;
        J[1][0] -= imag(cartesian[0]) / COMPLEX_STEP;
      } else if (H_ind == 1) {
        J[0][1] += imag(cartesian[1]) / COMPLEX_STEP;
        J[1][1] -= imag(cartesian[0]) / COMPLEX_STEP;
      } else if (H_ind == 2) {
        J[0][2] += imag(cartesian[1]) / COMPLEX_STEP;
        J[1][2] -= imag(cartesian[0]) / COMPLEX_STEP;
      }
    }
    bodyView(bodyIndex).velocity[0] += 0.25 * Kokkos::numbers::inv_pi * velocity[0];
    bodyView(bodyIndex).velocity[1] += 0.25 * Kokkos::numbers::inv_pi * velocity[1];
    bodyView(bodyIndex).velocity[2] += 0.25 * Kokkos::numbers::inv_pi * velocity[2];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        bodyView(bodyIndex).J[i][j] += 0.25 * Kokkos::numbers::inv_pi * J[i][j];
      }
    }
  }
}

} // namespace exafmm
#endif
