#ifndef TRAVERSE_HPP
#define TRAVERSE_HPP
#include <iostream>
#include "Kokkos_Core.hpp"
#include "Kokkos_Pair.hpp"
#include "exafmmTypes.hpp"
#include "multipoleKernels.hpp"
namespace exafmm {

//! Recursive call to dual tree traversal for list construction
// void getList(Cell *Ci, Cell *Cj)
// {
//   real_t dX[3];
//   for (int d = 0; d < 3; d++)
//     dX[d] = Ci->X[d] - Cj->X[d];                                       // Distance vector from source to target
//   real_t R2 = norm(dX) * MAC_theta * MAC_theta;                        // Scalar distance squared
//   if (R2 > (Ci->R + Cj->R) * (Ci->R + Cj->R)) {                        // If distance is far enough
//     Ci->listM2L.push_back(Cj);                                         //  Add to M2L list
//   } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {                     // Else if both cells are leafs
//     Ci->listP2P.push_back(Cj);                                         //  Add to P2P list
//   } else if (Cj->NCHILD == 0 || (Ci->R >= Cj->R && Ci->NCHILD != 0)) { // If Cj is leaf or Ci is larger
//     for (Cell *ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++) {   // Loop over Ci's children
//       getList(ci, Cj);                                                 //   Recursive call to target child cells
//     } //  End loop over Ci's children
//   } else {                                                           // Else if Ci is leaf or Cj is larger
//     for (Cell *cj = Cj->CHILD; cj != Cj->CHILD + Cj->NCHILD; cj++) { // Loop over Cj's children
//       getList(Ci, cj);                                               //   Recursive call to source child cells
//     } //  End loop over Cj's children
//   } // End if for leafs and Ci Cj size
// }

Kokkos::pair<std::vector<Kokkos::pair<size_t, size_t>>, std::vector<Kokkos::pair<size_t, size_t>>> getList(const Cell *targetRootCell, const Cell *sourceRootCell)
{
  std::vector<Kokkos::pair<size_t, size_t>> P2P;
  std::vector<Kokkos::pair<size_t, size_t>> M2L;
  P2P.reserve(1'000'000);
  M2L.reserve(1'000'000);

  using NodePair = Kokkos::pair<Cell *, Cell *>;

  if (targetRootCell->NCHILD == 0 && sourceRootCell->NCHILD == 0) {
    P2P.push_back(Kokkos::make_pair(0, 0));
    return {std::move(P2P), std::move(M2L)};
  }

  NodePair stack[128];
  stack[0] = Kokkos::make_pair(const_cast<Cell *>(targetRootCell), const_cast<Cell *>(sourceRootCell));

  int stackPos = 1;

  auto interact = [&stackPos, &P2P, &M2L, &targetRootCell, sourceRootCell](Cell *Cii, Cell *Cjj, NodePair *stack_) {
    real_t dX[3];
    for (int d = 0; d < 3; d++) {
      dX[d] = Cii->X[d] - Cjj->X[d]; // Distance vector from source to target
    }
    real_t R2 = norm(dX) * MAC_theta * MAC_theta; // Scalar distance squared
    if (R2 < (Cii->R + Cjj->R) * (Cii->R + Cjj->R)) {
      if (Cii->NCHILD == 0 && Cjj->NCHILD == 0) { // Only Leaf Cells do P2P
        P2P.push_back(Kokkos::make_pair(Cii - targetRootCell, Cjj - sourceRootCell));
      } else {
        assert(stackPos < 128);
        stack_[stackPos++] = NodePair{Cii, Cjj};
      }
    } else {
      M2L.push_back(Kokkos::make_pair(Cii - targetRootCell, Cjj - sourceRootCell));
    }
  };

  while (stackPos > 0) {
    NodePair nodePair = stack[--stackPos];
    auto     target   = nodePair.first;
    auto     source   = nodePair.second;

    if (source->NCHILD == 0 || (target->R >= source->R && target->NCHILD != 0)) {
      for (Cell *ci = target->CHILD; ci != target->CHILD + target->NCHILD; ci++) { // Loop over Ci's children
        interact(ci, source, stack);
      }
    } else {
      for (Cell *cj = source->CHILD; cj != source->CHILD + source->NCHILD; cj++) { // Loop over Cj's children
        interact(target, cj, stack);
      }
    }
  }
  return {std::move(P2P), std::move(M2L)};
}

//! Direct summation
void direct(Bodies &bodies, Bodies &jbodies)
{
  Cells cells(2);             // Define a pair of cells to pass to P2P kernel
  Cell *Ci  = &cells[0];      // Allocate single target
  Cell *Cj  = &cells[1];      // Allocate single source
  Ci->BODY  = &bodies[0];     // Iterator of first target body
  Ci->NBODY = bodies.size();  // Number of target bodies
  Cj->BODY  = &jbodies[0];    // Iterator of first source body
  Cj->NBODY = jbodies.size(); // Number of source bodies
  P2P(Ci, Cj);                // Evaluate P2P kenrel
}
} // namespace exafmm
#endif