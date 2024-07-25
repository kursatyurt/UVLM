#ifndef RELAXATION_HPP
#define RELAXATION_HPP
#include "Kokkos_Core.hpp"

namespace Vortex{


template <typename ParticleView>
KOKKOS_FUNCTION void relaxation(const size_t i,ParticleView particleView){
    nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                    (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                    (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
    nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)

    p.Gamma[1] = (1-rlxf)*p.Gamma[1] + rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
    p.Gamma[2] = (1-rlxf)*p.Gamma[2] + rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
    p.Gamma[3] = (1-rlxf)*p.Gamma[3] + rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

}

}

#endif