#ifndef EXAFMMTYPES_HPP
#define EXAFMMTYPES_HPP
#include <Kokkos_Core.hpp>
#include <vector>

namespace exafmm {
//! Basic type definitions
typedef double                  real_t;    //!< Floating point type
typedef Kokkos::complex<real_t> complex_t; //!< Complex type

//! Global variables
static constexpr int    P         = 4;                     //!< Order of expansions
static constexpr int    NTERM     = 3 * (P * (P + 1) / 2); //!< Number of coefficients
static constexpr int    ncrit     = 128;                    //!< Number of bodies per leaf cell
static constexpr real_t MAC_theta = 0.3;                   //!< Multipole acceptance criterion

//! Structure of bodies
struct Body {
  real_t J[3][3]{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
  real_t X[3]{0.0,0.0,0.0};            //!< Location
  real_t alpha[3]{0.0,0.0,0.0};        //!< Strength
  real_t velocity[3]{0.0,0.0,0.0};     //!< Velocity
  real_t dadt[3]{0.0,0.0,0.0};         //!< Rate of change of Strength
  real_t velocity_old[3]{0.0,0.0,0.0}; //!< Storage for RK Schemes
  real_t dadt_old[3]{0.0,0.0,0.0};     //!< Storage for RK Schemes
  real_t X_old[3]{0.0,0.0,0.0};        //!< Initial state storage for RK Schemes
  real_t alpha_old[3]{0.0,0.0,0.0};    //!< Initial state storage for RK Schemes
  real_t radius{0.0};          //!< Radius
  real_t drdt{0.0};            //!< Rate of change of radius
  real_t radius_old{0.0};      //!< Initial state storage for RK Schemes
  real_t drdt_old{0.0};        //!< Initial state storage for RK Schemes
};
typedef std::vector<Body> Bodies; //!< Vector of bodies

//! Structure of cells
struct Cell {
  size_t NCHILD;      //!< Number of child cells
  size_t NBODY;       //!< Number of descendant bodies
  size_t ChildOffset; //!< Offset of first child cell
  size_t BodyOffset;  //!< Offset of first body
  real_t X[3];        //!< Cell center
  Cell  *CHILD;       //!< Pointer of first child cell
  Body  *BODY;        //!< Pointer of first body
  real_t R;           //!< Cell radius
  size_t level;       //!< Level of cell in tree
};
typedef std::vector<Cell> Cells; //!< Vector of cells
} // namespace exafmm

#endif
