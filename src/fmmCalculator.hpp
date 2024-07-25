#ifndef FMMCALCULATOR_HPP
#define FMMCALCULATOR_HPP

#include "Kokkos_NumericTraits.hpp"
#include "exafmmTypes.hpp"
#include "traverse.hpp"
#include "tree.hpp"
namespace Vortex {
// Viscosity
struct Inviscid {};
struct PSE {};
struct CSM {};
// Time integration Scheme
struct Euler {};
struct RK2 {};
struct RK3 {};
struct RK4 {};
// VPM Formulation
struct cVPM {};
struct rVPM {};
// Stretching formulation
struct Transposed {};
struct Classic {};

template <typename Host, typename Device, typename IntegrationScheme, typename ViscousScheme, typename Formulation, typename Stretching>
struct FMMCalculator {
  /// @brief  Cell Definition to Transfer to GPU
  struct Cell {
    size_t         NCHILD;
    size_t         NBODY;
    size_t         ChildOffset;
    size_t         BodyOffset;
    exafmm::real_t X[3];
  };
  typedef Kokkos::pair<size_t, size_t>                              IterPair;
  typedef Kokkos::View<exafmm::complex_t * [exafmm::NTERM], Device> MultipoleView;
  typedef Kokkos::View<exafmm::complex_t * [exafmm::NTERM], Device> LocalView;
  typedef std::map<size_t, std::vector<size_t>>                     LevelMap;
  typedef Kokkos::View<IterPair *, Device>                          PairView;
  typedef Kokkos::View<Cell *, Device>                              CellView;
  typedef Kokkos::View<exafmm::Body *, Device>                      ParticleView;
  typedef Kokkos::View<size_t *>                                    LevelMapView;
  typedef exafmm::Cells                                             HostExaFMMCells;

  LevelMap      _levelMap;
  CellView      _cellsView;
  MultipoleView _multipoleView;
  LocalView     _localView;
  ParticleView  _bodiesView;
  PairView      _M2LPairsView;
  PairView      _P2PPairsView;
  LevelMapView  _levelMapView;
  /// @brief Time step for the integration
  exafmm::real_t _dt;
  /// @brief Viscosity of the fluid
  exafmm::real_t _nu;
  /// @brief The Tree for the particles
  HostExaFMMCells _particleCells;
  /// @brief The Tree for the sensor points
  HostExaFMMCells _sensorCells;
  // Sensor Tree
  LocalView    _localViewSensors;
  CellView     _cellsViewSensors;
  ParticleView _sensorsView;
  // RBF and CG Related
  exafmm::real_t _alpha[3]{0.0, 0.0, 0.0};
  exafmm::real_t _beta[3]{0.0, 0.0, 0.0};
  // RBF2
  Kokkos::View<int *, Device> _P2PCounts;

  void createTree(exafmm::Bodies &bodies)
  {
    Kokkos::Profiling::pushRegion("Tree Construction on CPU");
    _particleCells = std::move(exafmm::buildTree(bodies)); // Build tree
    LevelMap levelMap;
    for (size_t i = 0; i < _particleCells.size(); i++) {
      levelMap[_particleCells[i].level].push_back(i);
    }
    _levelMap = std::move(levelMap);

    // Copy Cell information to Device
    _cellsView         = CellView("CellView", _particleCells.size());
    auto cellsViewHost = Kokkos::create_mirror_view(_cellsView);
    Kokkos::parallel_for("Cell Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, _particleCells.size()), [&](const size_t i) {
      const auto Ci                = &_particleCells[i];
      cellsViewHost(i).NCHILD      = Ci->NCHILD;
      cellsViewHost(i).NBODY       = Ci->NBODY;
      cellsViewHost(i).BodyOffset  = Ci->BodyOffset;
      cellsViewHost(i).ChildOffset = Ci->ChildOffset;
      cellsViewHost(i).X[0]        = Ci->X[0];
      cellsViewHost(i).X[1]        = Ci->X[1];
      cellsViewHost(i).X[2]        = Ci->X[2];
    });

    auto x         = exafmm::getList(&_particleCells[0], &_particleCells[0]); // Pass root cell to recursive call
    auto _P2PPairs = std::move(x.first);
    auto _M2LPairs = std::move(x.second);
    std::sort(_M2LPairs.begin(), _M2LPairs.end(), [](const auto &left, const auto &right) { return left.second < right.second; });
    const auto M2LPairsSize = _M2LPairs.size();
    const auto P2PPairsSize = _P2PPairs.size();
    // Copy Pairs to Device
    _M2LPairsView         = PairView("M2LPairsView", M2LPairsSize);
    _P2PPairsView         = PairView("P2PPairsView", P2PPairsSize);
    auto M2LPairsViewHost = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _M2LPairsView);
    auto P2PPairsViewHost = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _P2PPairsView);
    Kokkos::parallel_for("M2L Pair Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, M2LPairsSize), [&](const size_t i) { M2LPairsViewHost(i) = _M2LPairs[i]; });
    Kokkos::parallel_for("P2P Pair Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, P2PPairsSize), [&](const size_t i) { P2PPairsViewHost(i) = _P2PPairs[i]; });

    Kokkos::fence();
    Kokkos::deep_copy(_cellsView, cellsViewHost);
    Kokkos::deep_copy(_M2LPairsView, M2LPairsViewHost);
    Kokkos::deep_copy(_P2PPairsView, P2PPairsViewHost);
    copyBodies2Device(bodies);
    Kokkos::Profiling::popRegion();
  }

  void createSensorTree(exafmm::Bodies &sensors)
  {
    Kokkos::Profiling::pushRegion("Tree Construction on CPU (Sensors)");
    _sensorCells = std::move(exafmm::buildTree(sensors)); // Build tree
    LevelMap levelMap;
    for (size_t i = 0; i < _sensorCells.size(); i++) {
      levelMap[_sensorCells[i].level].push_back(i);
    }
    _levelMap = std::move(levelMap);

    // Copy Cell information to Device
    _cellsViewSensors  = CellView("CellViewSensors", _sensorCells.size());
    auto cellsViewHost = Kokkos::create_mirror_view(_cellsViewSensors);
    Kokkos::parallel_for("Cell Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, _sensorCells.size()), [&](const size_t i) {
      const auto Ci                = &_sensorCells[i];
      cellsViewHost(i).NCHILD      = Ci->NCHILD;
      cellsViewHost(i).NBODY       = Ci->NBODY;
      cellsViewHost(i).BodyOffset  = Ci->BodyOffset;
      cellsViewHost(i).ChildOffset = Ci->ChildOffset;
      cellsViewHost(i).X[0]        = Ci->X[0];
      cellsViewHost(i).X[1]        = Ci->X[1];
      cellsViewHost(i).X[2]        = Ci->X[2];
    });

    auto x         = exafmm::getList(&_sensorCells[0], &_particleCells[0]); // Pass root cell to recursive call
    auto _P2PPairs = std::move(x.first);
    auto _M2LPairs = std::move(x.second);
    std::sort(_M2LPairs.begin(), _M2LPairs.end(), [](const auto &left, const auto &right) { return left.second < right.second; });
    const auto M2LPairsSize = _M2LPairs.size();
    const auto P2PPairsSize = _P2PPairs.size();
    // Copy Pairs to Device
    _M2LPairsView         = PairView("M2LPairsView", M2LPairsSize);
    _P2PPairsView         = PairView("P2PPairsView", P2PPairsSize);
    auto M2LPairsViewHost = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _M2LPairsView);
    auto P2PPairsViewHost = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _P2PPairsView);
    Kokkos::parallel_for("M2L Pair Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, M2LPairsSize), [&](const size_t i) { M2LPairsViewHost(i) = _M2LPairs[i]; });
    Kokkos::parallel_for("P2P Pair Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, P2PPairsSize), [&](const size_t i) { P2PPairsViewHost(i) = _P2PPairs[i]; });

    Kokkos::fence();
    Kokkos::deep_copy(_cellsViewSensors, cellsViewHost);
    Kokkos::deep_copy(_M2LPairsView, M2LPairsViewHost);
    Kokkos::deep_copy(_P2PPairsView, P2PPairsViewHost);
    copySensors2Device(sensors);
    Kokkos::Profiling::popRegion();
  }

  struct sensorOnly {};
  void createSensorTree(sensorOnly, exafmm::Bodies &sensors)
  {
    Kokkos::Profiling::pushRegion("Tree Construction on CPU (Sensors)");
    _sensorCells = std::move(exafmm::buildTree(sensors)); // Build tree
    LevelMap levelMap;
    for (size_t i = 0; i < _sensorCells.size(); i++) {
      levelMap[_sensorCells[i].level].push_back(i);
    }
    _levelMap = std::move(levelMap);

    // Copy Cell information to Device
    _cellsViewSensors  = CellView("CellViewSensors", _sensorCells.size());
    auto cellsViewHost = Kokkos::create_mirror_view(_cellsViewSensors);
    Kokkos::parallel_for("Cell Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, _sensorCells.size()), [&](const size_t i) {
      const auto Ci                = &_sensorCells[i];
      cellsViewHost(i).NCHILD      = Ci->NCHILD;
      cellsViewHost(i).NBODY       = Ci->NBODY;
      cellsViewHost(i).BodyOffset  = Ci->BodyOffset;
      cellsViewHost(i).ChildOffset = Ci->ChildOffset;
      cellsViewHost(i).X[0]        = Ci->X[0];
      cellsViewHost(i).X[1]        = Ci->X[1];
      cellsViewHost(i).X[2]        = Ci->X[2];
    });

    auto       x            = exafmm::getList(&_sensorCells[0], &_sensorCells[0]); // Pass root cell to recursive call
    auto       _P2PPairs    = std::move(x.first);
    const auto P2PPairsSize = _P2PPairs.size();
    // Copy Pairs to Device
    _P2PPairsView         = PairView("P2PPairsView", P2PPairsSize);
    auto P2PPairsViewHost = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _P2PPairsView);
    Kokkos::parallel_for("P2P Pair Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, P2PPairsSize), [&](const size_t i) { P2PPairsViewHost(i) = _P2PPairs[i]; });

    Kokkos::fence();
    Kokkos::deep_copy(_cellsViewSensors, cellsViewHost);
    Kokkos::deep_copy(_P2PPairsView, P2PPairsViewHost);
    copySensors2Device(sensors);
    Kokkos::Profiling::popRegion();
  }

  void copySensors2Device(exafmm::Bodies &sensors)
  {
    Kokkos::Profiling::pushRegion("copySensors2Device");
    // Copy Body Information to Device
    _sensorsView         = ParticleView("SensorsView", sensors.size());
    auto sensorsViewHost = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _sensorsView);
    Kokkos::parallel_for("Body Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, sensors.size()), [&](const size_t i) { sensorsViewHost(i) = sensors[i]; });
    Kokkos::deep_copy(_sensorsView, sensorsViewHost);
    Kokkos::Profiling::popRegion();
  }

  void copyBodies2Device(exafmm::Bodies &bodies)
  {
    Kokkos::Profiling::pushRegion("copyBodies2Device");
    // Copy Body Information to Device
    _bodiesView         = ParticleView("BodiesView", bodies.size());
    auto bodiesViewHost = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _bodiesView);
    Kokkos::parallel_for("Body Initialization on Host", Kokkos::RangePolicy<Host>(Host(), 0, bodies.size()), [&](const size_t i) { bodiesViewHost(i) = bodies[i]; });
    Kokkos::deep_copy(_bodiesView, bodiesViewHost);
    Kokkos::Profiling::popRegion();
  }

  void copySensors2Host(exafmm::Bodies &sensors)
  {
    Kokkos::Profiling::pushRegion("copySensors2Host");
    // Copy Body Information to Host
    auto sensorsViewHost = Kokkos::create_mirror_view(_sensorsView);
    Kokkos::deep_copy(sensorsViewHost, _sensorsView);
    Kokkos::parallel_for("Sensor Replacement on Host", Kokkos::RangePolicy<Host>(Host(), 0, sensors.size()), [&](const size_t i) { sensors[i] = sensorsViewHost(i); });
    Kokkos::Profiling::popRegion();
  }

  void copyBodies2Host(exafmm::Bodies &bodies)
  {
    Kokkos::Profiling::pushRegion("copyBodies2Host");
    // Copy Body Information to Host
    auto bodiesViewHost = Kokkos::create_mirror_view(_bodiesView);
    Kokkos::deep_copy(bodiesViewHost, _bodiesView);
    Kokkos::parallel_for("Body Replacement on Host", Kokkos::RangePolicy<Host>(Host(), 0, bodies.size()), [&](const size_t i) { bodies[i] = bodiesViewHost(i); });
    Kokkos::Profiling::popRegion();
  }

  struct resetPoles {};
  KOKKOS_INLINE_FUNCTION void operator()(resetPoles, const size_t i) const
  {
    for (size_t j = 0; j < exafmm::NTERM; j++) {
      _multipoleView(i, j) = 0.0;
      _localView(i, j)     = 0.0;
    }
  }

  struct resetSensorPoles {};
  KOKKOS_INLINE_FUNCTION void operator()(resetSensorPoles, const size_t i) const
  {
    for (size_t j = 0; j < exafmm::NTERM; j++) {
      _localViewSensors(i, j) = 0.0;
    }
  }

  void initMultipoles()
  {
    // Initialize The Multipole and Local View-
    _multipoleView = MultipoleView("MultipoleView", _cellsView.extent(0));
    _localView     = LocalView("LocalView", _cellsView.extent(0));
  }

  void initSensorMultipoles()
  {
    _localViewSensors = LocalView("LocalViewSensors", _cellsViewSensors.extent(0));
  }

  void initSensors(exafmm::Bodies &sensors)
  {
    Kokkos::Profiling::pushRegion("Initialization of Sensors");
    createSensorTree(sensors);
    initSensorMultipoles();
    Kokkos::Profiling::popRegion();
  }

  void init(exafmm::Bodies &bodies)
  {
    Kokkos::Profiling::pushRegion("Initialization of FMM");
    createTree(bodies);
    initMultipoles();
    Kokkos::Profiling::popRegion();
  }

  struct upwardPass {};
  KOKKOS_INLINE_FUNCTION void operator()(upwardPass, const size_t i) const
  {
    const auto index = _levelMapView(i);
    if (_cellsView(index).NCHILD == 0) {
      exafmm::P2M(index, _cellsView, _bodiesView, _multipoleView);
    }
    exafmm::M2M(index, _cellsView, _multipoleView);
  }

  struct m2l {};
  KOKKOS_INLINE_FUNCTION void operator()(m2l, const size_t i) const
  {
    const auto [index1, index2] = _M2LPairsView(i);
    exafmm::M2L(index1, index2, _cellsView, _multipoleView, _localView);
  }

  struct M2LSensors {};
  KOKKOS_INLINE_FUNCTION void operator()(M2LSensors, const size_t i) const
  {
    const auto [index1, index2] = _M2LPairsView(i);
    exafmm::M2L(index1, index2, _cellsView, _cellsViewSensors, _multipoleView, _localViewSensors);
  }

  struct P2PSensors {};
  KOKKOS_INLINE_FUNCTION void operator()(P2PSensors, const Kokkos::TeamPolicy<>::member_type &teamMember) const
  {
    const auto [index1, index2] = _P2PPairsView(teamMember.league_rank());
    exafmm::P2P(index1, index2, _bodiesView, _sensorsView, _cellsView, _cellsViewSensors, teamMember);
  }

  struct p2p {};
  KOKKOS_INLINE_FUNCTION void operator()(p2p, const Kokkos::TeamPolicy<>::member_type &teamMember) const
  {
    const auto [index1, index2] = _P2PPairsView(teamMember.league_rank());
    exafmm::P2P(index1, index2, _bodiesView, _cellsView, teamMember);
  }

  KOKKOS_INLINE_FUNCTION void operator()(PSE, const Kokkos::TeamPolicy<>::member_type &teamMember) const
  {
    const auto [index1, index2] = _P2PPairsView(teamMember.league_rank());
    exafmm::PSE(index1, index2, _bodiesView, _cellsView, _nu, teamMember);
  }

  KOKKOS_INLINE_FUNCTION void operator()(CSM, const size_t i) const
  {
    _bodiesView(i).drdt += (Kokkos::sqrt((_bodiesView(i).radius * _bodiesView(i).radius) + (2.0 * _nu * _dt)) - _bodiesView(i).radius) / _dt;
  }

  struct downwardPass {};
  KOKKOS_INLINE_FUNCTION void operator()(downwardPass, const size_t i) const
  {
    const auto index = _levelMapView(i);
    exafmm::L2L(index, _cellsView, _localView);
    if (_cellsView(index).NCHILD == 0) {
      exafmm::L2P(index, _cellsView, _localView, _bodiesView);
    }
  }

  struct downwardPassSensors {};
  KOKKOS_INLINE_FUNCTION void operator()(downwardPassSensors, const size_t i) const
  {
    const auto index = _levelMapView(i);
    exafmm::L2L(index, _cellsViewSensors, _localViewSensors);
    if (_cellsViewSensors(index).NCHILD == 0) {
      exafmm::L2P(index, _cellsViewSensors, _localViewSensors, _sensorsView);
    }
  }

  struct resetRates {};
  KOKKOS_INLINE_FUNCTION void operator()(resetRates, const size_t i) const
  {
    _bodiesView(i).velocity[0] = 0.0;
    _bodiesView(i).velocity[1] = 0.0;
    _bodiesView(i).velocity[2] = 0.0;
    _bodiesView(i).dadt[0]     = 0.0;
    _bodiesView(i).dadt[1]     = 0.0;
    _bodiesView(i).dadt[2]     = 0.0;
    _bodiesView(i).J[0][0]     = 0.0;
    _bodiesView(i).J[0][1]     = 0.0;
    _bodiesView(i).J[0][2]     = 0.0;
    _bodiesView(i).J[1][0]     = 0.0;
    _bodiesView(i).J[1][1]     = 0.0;
    _bodiesView(i).J[1][2]     = 0.0;
    _bodiesView(i).J[2][0]     = 0.0;
    _bodiesView(i).J[2][1]     = 0.0;
    _bodiesView(i).J[2][2]     = 0.0;
    _bodiesView(i).drdt        = 0.0;
  }

  struct resetSensors {};
  KOKKOS_INLINE_FUNCTION void operator()(resetSensors, const size_t i) const
  {
    _sensorsView(i).velocity[0] = 0.0;
    _sensorsView(i).velocity[1] = 0.0;
    _sensorsView(i).velocity[2] = 0.0;
    _sensorsView(i).dadt[0]     = 0.0;
    _sensorsView(i).dadt[1]     = 0.0;
    _sensorsView(i).dadt[2]     = 0.0;
  }

  struct StretchingTransposed {};
  KOKKOS_INLINE_FUNCTION void operator()(StretchingTransposed, const size_t i) const
  {
    _bodiesView(i).dadt[0] += _bodiesView(i).J[0][0] * _bodiesView(i).alpha[0] + _bodiesView(i).J[1][0] * _bodiesView(i).alpha[1] + _bodiesView(i).J[2][0] * _bodiesView(i).alpha[2];
    _bodiesView(i).dadt[1] += _bodiesView(i).J[0][1] * _bodiesView(i).alpha[0] + _bodiesView(i).J[1][1] * _bodiesView(i).alpha[1] + _bodiesView(i).J[2][1] * _bodiesView(i).alpha[2];
    _bodiesView(i).dadt[2] += _bodiesView(i).J[0][2] * _bodiesView(i).alpha[0] + _bodiesView(i).J[1][2] * _bodiesView(i).alpha[1] + _bodiesView(i).J[2][2] * _bodiesView(i).alpha[2];
  }

  struct StretchingClassic {};
  KOKKOS_INLINE_FUNCTION void operator()(StretchingClassic, const size_t i) const
  {
    _bodiesView(i).dadt[0] += _bodiesView(i).J[0][0] * _bodiesView(i).alpha[0] + _bodiesView(i).J[0][1] * _bodiesView(i).alpha[1] + _bodiesView(i).J[0][2] * _bodiesView(i).alpha[2];
    _bodiesView(i).dadt[1] += _bodiesView(i).J[1][0] * _bodiesView(i).alpha[0] + _bodiesView(i).J[1][1] * _bodiesView(i).alpha[1] + _bodiesView(i).J[1][2] * _bodiesView(i).alpha[2];
    _bodiesView(i).dadt[2] += _bodiesView(i).J[2][0] * _bodiesView(i).alpha[0] + _bodiesView(i).J[2][1] * _bodiesView(i).alpha[1] + _bodiesView(i).J[2][2] * _bodiesView(i).alpha[2];
  }

  struct rVPMTerms {};
  KOKKOS_INLINE_FUNCTION void operator()(rVPMTerms, const size_t i) const
  {
    const exafmm::real_t normAlpha = Kokkos::sqrt(_bodiesView(i).alpha[0] * _bodiesView(i).alpha[0] + _bodiesView(i).alpha[1] * _bodiesView(i).alpha[1] + _bodiesView(i).alpha[2] * _bodiesView(i).alpha[2]);
    const exafmm::real_t alphaHat[3]{_bodiesView(i).alpha[0] / normAlpha, _bodiesView(i).alpha[1] / normAlpha, _bodiesView(i).alpha[2] / normAlpha};
    const exafmm::real_t dotProductTerm = _bodiesView(i).dadt[0] * alphaHat[0] + _bodiesView(i).dadt[1] * alphaHat[1] + _bodiesView(i).dadt[2] * alphaHat[2];

    _bodiesView(i).dadt[0] -= 3. / 5. * dotProductTerm * alphaHat[0];
    _bodiesView(i).dadt[1] -= 3. / 5. * dotProductTerm * alphaHat[1];
    _bodiesView(i).dadt[2] -= 3. / 5. * dotProductTerm * alphaHat[2];
    _bodiesView(i).drdt -= 1.0 / 5.0 * _bodiesView(i).radius / normAlpha * dotProductTerm;
  }

  void calculate()
  {
    Kokkos::Profiling::pushRegion("Calculation of FMM");
    Kokkos::parallel_for("Reset The Multipoles", Kokkos::RangePolicy<Device, resetPoles>(Device(), 0, _cellsView.size()), *this);
    Kokkos::parallel_for("Reset The Rates", Kokkos::RangePolicy<Device, resetRates>(Device(), 0, _bodiesView.size()), *this);
    Kokkos::fence();

    for (int level = _levelMap.size(); level > 0; level--) {
      const auto size = _levelMap[level].size();
      Kokkos::resize(Kokkos::WithoutInitializing, _levelMapView, size);
      auto levelMapKokkosHost = Kokkos::create_mirror_view(_levelMapView);
      for (size_t i = 0; i < size; i++) {
        levelMapKokkosHost(i) = _levelMap[level][i];
      }
      Kokkos::deep_copy(_levelMapView, levelMapKokkosHost);

      Kokkos::parallel_for("UpwardPass", Kokkos::RangePolicy<Device, upwardPass>(Device(), 0, size), *this);
    }

    Kokkos::parallel_for("M2L", Kokkos::RangePolicy<Device, m2l>(Device(), 0, _M2LPairsView.extent(0)), *this);
    Kokkos::parallel_for("P2P", Kokkos::TeamPolicy<Device, p2p>(Device(), _P2PPairsView.extent(0), Kokkos::AUTO, Kokkos::TeamPolicy<Device>::vector_length_max()), *this);

    for (size_t level = 0; level < _levelMap.size(); level++) {
      const auto size = _levelMap[level].size();
      Kokkos::resize(Kokkos::WithoutInitializing, _levelMapView, size);
      auto levelMapKokkosHost = Kokkos::create_mirror_view(_levelMapView);
      for (size_t i = 0; i < size; i++) {
        levelMapKokkosHost(i) = _levelMap[level][i];
      }
      Kokkos::deep_copy(_levelMapView, levelMapKokkosHost);

      Kokkos::parallel_for("downwardPass", Kokkos::RangePolicy<Device, downwardPass>(Device(), 0, size), *this);
    }

    if constexpr (std::is_same<Stretching, Transposed>::value) {
      Kokkos::parallel_for("CalculateTransposedStrecthing", Kokkos::RangePolicy<Device, StretchingTransposed>(Device(), 0, _bodiesView.size()), *this);
    } else if constexpr (std::is_same<Stretching, Classic>::value) {
      Kokkos::parallel_for("CalculateClassicStrecthing", Kokkos::RangePolicy<Device, StretchingClassic>(Device(), 0, _bodiesView.size()), *this);
    } else {
      static_assert(std::is_same<Stretching, Transposed>::value || std::is_same<Stretching, Classic>::value, "Stretching should be either Transposed or Classic");
    }

    if constexpr (std::is_same<Formulation, rVPM>::value) {
      Kokkos::fence();
      Kokkos::parallel_for("rVPMTerms", Kokkos::RangePolicy<Device, rVPMTerms>(Device(), 0, _bodiesView.size()), *this);
    }
    Kokkos::Profiling::popRegion();
  }

  void calculateSensorPoints()
  {
    Kokkos::Profiling::pushRegion("Calculation of Sensors");
    Kokkos::parallel_for("Reset The Sensor Rates", Kokkos::RangePolicy<Device, resetSensors>(Device(), 0, _sensorsView.size()), *this);
    Kokkos::parallel_for("Reset The Sensor Multipoles", Kokkos::RangePolicy<Device, resetSensorPoles>(Device(), 0, _localViewSensors.extent(0)), *this);
    Kokkos::fence();

    Kokkos::parallel_for("M2LSensors", Kokkos::RangePolicy<Device, M2LSensors>(Device(), 0, _M2LPairsView.extent(0)), *this);
    Kokkos::parallel_for("P2PSensors", Kokkos::TeamPolicy<Device, P2PSensors>(Device(), _P2PPairsView.extent(0), Kokkos::AUTO, Kokkos::TeamPolicy<Device>::vector_length_max()), *this);

    for (size_t level = 0; level < _levelMap.size(); level++) {
      const auto size = _levelMap[level].size();
      Kokkos::resize(Kokkos::WithoutInitializing, _levelMapView, size);
      auto levelMapKokkosHost = Kokkos::create_mirror_view(_levelMapView);
      for (size_t i = 0; i < size; i++) {
        levelMapKokkosHost(i) = _levelMap[level][i];
      }
      Kokkos::deep_copy(_levelMapView, levelMapKokkosHost);
      Kokkos::parallel_for("downwardPassSensors", Kokkos::RangePolicy<Device, downwardPassSensors>(Device(), 0, size), *this);
    }
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
  }

  struct storeInitialState {};
  KOKKOS_INLINE_FUNCTION void operator()(storeInitialState, const size_t i) const
  {
    _bodiesView(i).velocity_old[0] = 0.0;
    _bodiesView(i).velocity_old[1] = 0.0;
    _bodiesView(i).velocity_old[2] = 0.0;
    _bodiesView(i).dadt_old[0]     = 0.0;
    _bodiesView(i).dadt_old[1]     = 0.0;
    _bodiesView(i).dadt_old[2]     = 0.0;
    _bodiesView(i).X_old[0]        = _bodiesView(i).X[0];
    _bodiesView(i).X_old[1]        = _bodiesView(i).X[1];
    _bodiesView(i).X_old[2]        = _bodiesView(i).X[2];
    _bodiesView(i).alpha_old[0]    = _bodiesView(i).alpha[0];
    _bodiesView(i).alpha_old[1]    = _bodiesView(i).alpha[1];
    _bodiesView(i).alpha_old[2]    = _bodiesView(i).alpha[2];
    _bodiesView(i).radius_old      = _bodiesView(i).radius;
  }

  struct stepEuler {};
  KOKKOS_INLINE_FUNCTION void operator()(stepEuler, const size_t i) const
  {
    _bodiesView(i).X[0] += _bodiesView(i).velocity[0] * _dt;
    _bodiesView(i).X[1] += _bodiesView(i).velocity[1] * _dt;
    _bodiesView(i).X[2] += _bodiesView(i).velocity[2] * _dt;
    _bodiesView(i).alpha[0] += _bodiesView(i).dadt[0] * _dt;
    _bodiesView(i).alpha[1] += _bodiesView(i).dadt[1] * _dt;
    _bodiesView(i).alpha[2] += _bodiesView(i).dadt[2] * _dt;
    _bodiesView(i).radius += _bodiesView(i).drdt * _dt;
  }

  struct stepRK21 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK21, const size_t i) const
  {
    // z_1 = dt * f(y_0)
    _bodiesView(i).velocity_old[0] = _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] = _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] = _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0]     = _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1]     = _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2]     = _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old        = _bodiesView(i).drdt;
    _bodiesView(i).X[0] += _dt * _bodiesView(i).velocity_old[0];
    _bodiesView(i).X[1] += _dt * _bodiesView(i).velocity_old[1];
    _bodiesView(i).X[2] += _dt * _bodiesView(i).velocity_old[2];
    _bodiesView(i).alpha[0] += _dt * _bodiesView(i).dadt_old[0];
    _bodiesView(i).alpha[1] += _dt * _bodiesView(i).dadt_old[1];
    _bodiesView(i).alpha[2] += _dt * _bodiesView(i).dadt_old[2];
    _bodiesView(i).radius += _dt * _bodiesView(i).drdt_old;
  }

  struct stepRK22 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK22, const size_t i) const

  {
    _bodiesView(i).velocity_old[0] = 0.5 * _bodiesView(i).velocity[0] + 0.5 * _bodiesView(i).velocity_old[0];
    _bodiesView(i).velocity_old[1] = 0.5 * _bodiesView(i).velocity[1] + 0.5 * _bodiesView(i).velocity_old[1];
    _bodiesView(i).velocity_old[2] = 0.5 * _bodiesView(i).velocity[2] + 0.5 * _bodiesView(i).velocity_old[2];
    _bodiesView(i).dadt_old[0]     = 0.5 * _bodiesView(i).dadt[0] + 0.5 * _bodiesView(i).dadt_old[0];
    _bodiesView(i).dadt_old[1]     = 0.5 * _bodiesView(i).dadt[1] + 0.5 * _bodiesView(i).dadt_old[1];
    _bodiesView(i).dadt_old[2]     = 0.5 * _bodiesView(i).dadt[2] + 0.5 * _bodiesView(i).dadt_old[2];
    _bodiesView(i).drdt_old        = 0.5 * _bodiesView(i).drdt + 0.5 * _bodiesView(i).drdt_old;

    _bodiesView(i).X[0]     = _bodiesView(i).X_old[0] + _dt * _bodiesView(i).velocity_old[0];
    _bodiesView(i).X[1]     = _bodiesView(i).X_old[1] + _dt * _bodiesView(i).velocity_old[1];
    _bodiesView(i).X[2]     = _bodiesView(i).X_old[2] + _dt * _bodiesView(i).velocity_old[2];
    _bodiesView(i).alpha[0] = _bodiesView(i).alpha_old[0] + _dt * _bodiesView(i).dadt_old[0];
    _bodiesView(i).alpha[1] = _bodiesView(i).alpha_old[1] + _dt * _bodiesView(i).dadt_old[1];
    _bodiesView(i).alpha[2] = _bodiesView(i).alpha_old[2] + _dt * _bodiesView(i).dadt_old[2];
    _bodiesView(i).radius   = _bodiesView(i).radius_old + _dt * _bodiesView(i).drdt_old;
  }

  struct stepRK31 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK31, const size_t i) const
  {
    // z_1 = dt * f(y_0)
    _bodiesView(i).velocity_old[0] = _dt * _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] = _dt * _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] = _dt * _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0]     = _dt * _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1]     = _dt * _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2]     = _dt * _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old        = _dt * _bodiesView(i).drdt;
    // y_1 = y_0 + 1/3 * z_1
    _bodiesView(i).X[0] += (1.0 / 3.0) * _bodiesView(i).velocity_old[0];
    _bodiesView(i).X[1] += (1.0 / 3.0) * _bodiesView(i).velocity_old[1];
    _bodiesView(i).X[2] += (1.0 / 3.0) * _bodiesView(i).velocity_old[2];
    _bodiesView(i).alpha[0] += (1.0 / 3.0) * _bodiesView(i).dadt_old[0];
    _bodiesView(i).alpha[1] += (1.0 / 3.0) * _bodiesView(i).dadt_old[1];
    _bodiesView(i).alpha[2] += (1.0 / 3.0) * _bodiesView(i).dadt_old[2];
    _bodiesView(i).radius += (1.0 / 3.0) * _bodiesView(i).drdt_old;
  }

  struct stepRK32 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK32, const size_t i) const

  {
    // z_2 = -5/9 z_1 + dt*f(y_1)
    _bodiesView(i).velocity_old[0] = -(5.0 / 9.0) * _bodiesView(i).velocity_old[0] + _dt * _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] = -(5.0 / 9.0) * _bodiesView(i).velocity_old[1] + _dt * _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] = -(5.0 / 9.0) * _bodiesView(i).velocity_old[2] + _dt * _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0]     = -(5.0 / 9.0) * _bodiesView(i).dadt_old[0] + _dt * _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1]     = -(5.0 / 9.0) * _bodiesView(i).dadt_old[1] + _dt * _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2]     = -(5.0 / 9.0) * _bodiesView(i).dadt_old[2] + _dt * _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old        = -(5.0 / 9.0) * _bodiesView(i).drdt_old + _dt * _bodiesView(i).drdt;

    _bodiesView(i).X[0] += (15.0 / 16.0) * _bodiesView(i).velocity_old[0];
    _bodiesView(i).X[1] += (15.0 / 16.0) * _bodiesView(i).velocity_old[1];
    _bodiesView(i).X[2] += (15.0 / 16.0) * _bodiesView(i).velocity_old[2];
    _bodiesView(i).alpha[0] += (15.0 / 16.0) * _bodiesView(i).dadt_old[0];
    _bodiesView(i).alpha[1] += (15.0 / 16.0) * _bodiesView(i).dadt_old[1];
    _bodiesView(i).alpha[2] += (15.0 / 16.0) * _bodiesView(i).dadt_old[2];
    _bodiesView(i).radius += (15.0 / 16.0) * _bodiesView(i).drdt_old;
  }

  struct stepRK33 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK33, const size_t i) const

  {
    // z_3 = -153/128 z_2 + dt *f(y2)
    _bodiesView(i).velocity_old[0] = (-153.0 / 128.0) * _bodiesView(i).velocity_old[0] + _dt * _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] = (-153.0 / 128.0) * _bodiesView(i).velocity_old[1] + _dt * _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] = (-153.0 / 128.0) * _bodiesView(i).velocity_old[2] + _dt * _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0]     = (-153.0 / 128.0) * _bodiesView(i).dadt_old[0] + _dt * _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1]     = (-153.0 / 128.0) * _bodiesView(i).dadt_old[1] + _dt * _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2]     = (-153.0 / 128.0) * _bodiesView(i).dadt_old[2] + _dt * _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old        = (-153.0 / 128.0) * _bodiesView(i).drdt_old + _dt * _bodiesView(i).drdt;
    // y_3 = y_2 + 8/15 z_3
    _bodiesView(i).X[0] += (8.0 / 15.0) * _bodiesView(i).velocity_old[0];
    _bodiesView(i).X[1] += (8.0 / 15.0) * _bodiesView(i).velocity_old[1];
    _bodiesView(i).X[2] += (8.0 / 15.0) * _bodiesView(i).velocity_old[2];
    _bodiesView(i).alpha[0] += (8.0 / 15.0) * _bodiesView(i).dadt_old[0];
    _bodiesView(i).alpha[1] += (8.0 / 15.0) * _bodiesView(i).dadt_old[1];
    _bodiesView(i).alpha[2] += (8.0 / 15.0) * _bodiesView(i).dadt_old[2];
    _bodiesView(i).radius += (8.0 / 15.0) * _bodiesView(i).drdt_old;
  }

  struct stepRK41 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK41, const size_t i) const
  {
    // z_1 = dt * f(y_0)
    _bodiesView(i).velocity_old[0] = _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] = _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] = _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0]     = _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1]     = _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2]     = _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old        = _bodiesView(i).drdt;
    // y_1 = y_0 + 1/3 * z_1
    _bodiesView(i).X[0] += 0.5 * _dt * _bodiesView(i).velocity_old[0];
    _bodiesView(i).X[1] += 0.5 * _dt * _bodiesView(i).velocity_old[1];
    _bodiesView(i).X[2] += 0.5 * _dt * _bodiesView(i).velocity_old[2];
    _bodiesView(i).alpha[0] += 0.5 * _dt * _bodiesView(i).dadt_old[0];
    _bodiesView(i).alpha[1] += 0.5 * _dt * _bodiesView(i).dadt_old[1];
    _bodiesView(i).alpha[2] += 0.5 * _dt * _bodiesView(i).dadt_old[2];
    _bodiesView(i).radius += 0.5 * _dt * _bodiesView(i).drdt_old;
  }

  struct stepRK42 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK42, const size_t i) const

  {
    _bodiesView(i).velocity_old[0] += 2.0 * _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] += 2.0 * _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] += 2.0 * _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0] += 2.0 * _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1] += 2.0 * _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2] += 2.0 * _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old += 2.0 * _bodiesView(i).drdt;

    _bodiesView(i).X[0]     = _bodiesView(i).X_old[0] + 0.5 * _dt * _bodiesView(i).velocity[0];
    _bodiesView(i).X[1]     = _bodiesView(i).X_old[1] + 0.5 * _dt * _bodiesView(i).velocity[1];
    _bodiesView(i).X[2]     = _bodiesView(i).X_old[2] + 0.5 * _dt * _bodiesView(i).velocity[2];
    _bodiesView(i).alpha[0] = _bodiesView(i).alpha_old[0] + 0.5 * _dt * _bodiesView(i).dadt[0];
    _bodiesView(i).alpha[1] = _bodiesView(i).alpha_old[1] + 0.5 * _dt * _bodiesView(i).dadt[1];
    _bodiesView(i).alpha[2] = _bodiesView(i).alpha_old[2] + 0.5 * _dt * _bodiesView(i).dadt[2];
    _bodiesView(i).radius   = _bodiesView(i).radius_old + 0.5 * _dt * _bodiesView(i).drdt;
  }

  struct stepRK43 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK43, const size_t i) const

  {
    _bodiesView(i).velocity_old[0] += 2.0 * _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] += 2.0 * _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] += 2.0 * _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0] += 2.0 * _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1] += 2.0 * _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2] += 2.0 * _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old += 2.0 * _bodiesView(i).drdt;

    _bodiesView(i).X[0]     = _bodiesView(i).X_old[0] + 0.5 * _dt * _bodiesView(i).velocity[0];
    _bodiesView(i).X[1]     = _bodiesView(i).X_old[1] + 0.5 * _dt * _bodiesView(i).velocity[1];
    _bodiesView(i).X[2]     = _bodiesView(i).X_old[2] + 0.5 * _dt * _bodiesView(i).velocity[2];
    _bodiesView(i).alpha[0] = _bodiesView(i).alpha_old[0] + 0.5 * _dt * _bodiesView(i).dadt[0];
    _bodiesView(i).alpha[1] = _bodiesView(i).alpha_old[1] + 0.5 * _dt * _bodiesView(i).dadt[1];
    _bodiesView(i).alpha[2] = _bodiesView(i).alpha_old[2] + 0.5 * _dt * _bodiesView(i).dadt[2];
    _bodiesView(i).radius   = _bodiesView(i).radius_old + 0.5 * _dt * _bodiesView(i).drdt;
  }

  struct stepRK44 {};
  KOKKOS_INLINE_FUNCTION void operator()(stepRK44, const size_t i) const

  {
    _bodiesView(i).velocity_old[0] += _bodiesView(i).velocity[0];
    _bodiesView(i).velocity_old[1] += _bodiesView(i).velocity[1];
    _bodiesView(i).velocity_old[2] += _bodiesView(i).velocity[2];
    _bodiesView(i).dadt_old[0] += _bodiesView(i).dadt[0];
    _bodiesView(i).dadt_old[1] += _bodiesView(i).dadt[1];
    _bodiesView(i).dadt_old[2] += _bodiesView(i).dadt[2];
    _bodiesView(i).drdt_old += _bodiesView(i).drdt;

    _bodiesView(i).X[0]     = _bodiesView(i).X_old[0] + (_dt / 6.0) * _bodiesView(i).velocity_old[0];
    _bodiesView(i).X[1]     = _bodiesView(i).X_old[1] + (_dt / 6.0) * _bodiesView(i).velocity_old[1];
    _bodiesView(i).X[2]     = _bodiesView(i).X_old[2] + (_dt / 6.0) * _bodiesView(i).velocity_old[2];
    _bodiesView(i).alpha[0] = _bodiesView(i).alpha_old[0] + (_dt / 6.0) * _bodiesView(i).dadt_old[0];
    _bodiesView(i).alpha[1] = _bodiesView(i).alpha_old[1] + (_dt / 6.0) * _bodiesView(i).dadt_old[1];
    _bodiesView(i).alpha[2] = _bodiesView(i).alpha_old[2] + (_dt / 6.0) * _bodiesView(i).dadt_old[2];
    _bodiesView(i).radius   = _bodiesView(i).radius_old + (_dt / 6.0) * _bodiesView(i).drdt_old;
  }

  void calculateViscosity()
  {
    if constexpr (std::is_same<ViscousScheme, Inviscid>::value) {
      return;
    }
    if constexpr (std::is_same<ViscousScheme, PSE>::value) {
      Kokkos::parallel_for("PSE", Kokkos::TeamPolicy<Device, PSE>((_P2PPairsView.extent(0)), Kokkos::AUTO, Kokkos::TeamPolicy<Device>::vector_length_max()), *this);
    }
    if constexpr (std::is_same<ViscousScheme, CSM>::value) {
      Kokkos::parallel_for("CSM", Kokkos::RangePolicy<Device, CSM>(Device(), 0, _bodiesView.extent(0)), *this);
    }
  };

  void getSensorData(exafmm::Bodies &sensors)
  {
    initSensors(sensors);
    calculateSensorPoints();
    copySensors2Host(sensors);
  }

  void calculateVorticityDualTree()
  {
    Kokkos::Profiling::pushRegion("Calculation of Vorticity");
    Kokkos::parallel_for("Reset The Sensor Rates", Kokkos::RangePolicy<Device, resetSensorsOldVelocity>(Device(), 0, _sensorsView.size()), *this);
    Kokkos::fence();
    Kokkos::parallel_for("P2PVorticity", Kokkos::TeamPolicy<Device, P2PVorticityDual>(Device(), _P2PPairsView.extent(0), Kokkos::AUTO, Kokkos::TeamPolicy<Device>::vector_length_max()), *this);
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
  }

  struct resetSensorsOldVelocity {};
  KOKKOS_INLINE_FUNCTION void operator()(resetSensorsOldVelocity, const size_t i) const
  {
    _sensorsView(i).velocity_old[0] = 0.0;
    _sensorsView(i).velocity_old[1] = 0.0;
    _sensorsView(i).velocity_old[2] = 0.0;
  }

  struct P2PVorticityDual {};
  KOKKOS_INLINE_FUNCTION void operator()(P2PVorticityDual, const Kokkos::TeamPolicy<>::member_type &teamMember) const
  {
    const auto [index1, index2] = _P2PPairsView(teamMember.league_rank());
    exafmm::P2P_vorticity(index1, index2, _bodiesView, _sensorsView, _cellsView, _cellsViewSensors, teamMember);
  }

  struct P2PVorticitySingle {};
  KOKKOS_INLINE_FUNCTION void operator()(P2PVorticitySingle, const Kokkos::TeamPolicy<>::member_type &teamMember) const
  {
    const auto [index1, index2] = _P2PPairsView(teamMember.league_rank());
    exafmm::P2P_vorticity(index1, index2, _sensorsView, _cellsViewSensors, teamMember);
  }

  struct initAlphaValues {};
  KOKKOS_INLINE_FUNCTION void operator()(initAlphaValues, const size_t i) const
  {
    // Assume spherical volume
    const exafmm::real_t vol = (4. / 3. * Kokkos::numbers::pi * _sensorsView(i).radius * _sensorsView(i).radius * _sensorsView(i).radius);
    _sensorsView(i).alpha[0] = _sensorsView(i).velocity_old[0] * vol;
    _sensorsView(i).alpha[1] = _sensorsView(i).velocity_old[1] * vol;
    _sensorsView(i).alpha[2] = _sensorsView(i).velocity_old[2] * vol;
  }

  void calculateVorticitySingleTree()
  {
    Kokkos::Profiling::pushRegion("Calculation of Vorticity");
    Kokkos::parallel_for("Reset The Sensor Rates", Kokkos::RangePolicy<Device, resetSensors>(Device(), 0, _sensorsView.size()), *this);
    Kokkos::fence();
    Kokkos::parallel_for("P2PVorticity", Kokkos::TeamPolicy<Device, P2PVorticitySingle>(Device(), _P2PPairsView.extent(0), Kokkos::AUTO, Kokkos::TeamPolicy<Device>::vector_length_max()), *this);
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
  }

  struct setInitialResidualAndUpdateAlpha {};
  KOKKOS_INLINE_FUNCTION void operator()(setInitialResidualAndUpdateAlpha, const size_t i, exafmm::real_t &lsum1, exafmm::real_t &lsum2, exafmm::real_t &lsum3) const
  {
    // The residual is target_vorticity - current_vorticity
    const auto r00 = _sensorsView(i).velocity_old[0] - _sensorsView(i).velocity[0];
    const auto r01 = _sensorsView(i).velocity_old[1] - _sensorsView(i).velocity[1];
    const auto r02 = _sensorsView(i).velocity_old[2] - _sensorsView(i).velocity[2];
    lsum1 += r00 * r00;
    lsum2 += r01 * r01;
    lsum3 += r02 * r02;
    // Init solution field
    _sensorsView(i).dadt_old[0] = _sensorsView(i).alpha[0];
    _sensorsView(i).dadt_old[1] = _sensorsView(i).alpha[1];
    _sensorsView(i).dadt_old[2] = _sensorsView(i).alpha[2];
    // Set Alpha values to p0 = r0 here on we iterate over the residual field
    _sensorsView(i).alpha[0] = r00;
    _sensorsView(i).alpha[1] = r01;
    _sensorsView(i).alpha[2] = r02;
    // Store residuals in X_old
    _sensorsView(i).X_old[0] = r00;
    _sensorsView(i).X_old[1] = r01;
    _sensorsView(i).X_old[2] = r02;
  }

  struct calculatePAP {};
  KOKKOS_INLINE_FUNCTION void operator()(calculatePAP, const size_t i, exafmm::real_t &lsum1, exafmm::real_t &lsum2, exafmm::real_t &lsum3) const
  {
    // The residual is p is alpha and Ap is velocity
    lsum1 += _sensorsView(i).alpha[0] * _sensorsView(i).velocity[0];
    lsum2 += _sensorsView(i).alpha[1] * _sensorsView(i).velocity[1];
    lsum3 += _sensorsView(i).alpha[2] * _sensorsView(i).velocity[2];
  }

  struct updateAlphasAndResiduals {};
  KOKKOS_INLINE_FUNCTION void operator()(updateAlphasAndResiduals, const size_t i) const
  {
    // Update alphas (x_(k+1) = x_k + alpha_k * p_k) // Here we store x in dadt_old (alpha) and p in alpha
    _sensorsView(i).dadt_old[0] += _alpha[0] * _sensorsView(i).alpha[0];
    _sensorsView(i).dadt_old[1] += _alpha[1] * _sensorsView(i).alpha[1];
    _sensorsView(i).dadt_old[2] += _alpha[2] * _sensorsView(i).alpha[2];
    // Update residuals (r_(k+1) = r_k - alpha_k * Ap_k) // Here we store r in X_old and Ap in velocity
    _sensorsView(i).X_old[0] -= _alpha[0] * _sensorsView(i).velocity[0];
    _sensorsView(i).X_old[1] -= _alpha[1] * _sensorsView(i).velocity[1];
    _sensorsView(i).X_old[2] -= _alpha[2] * _sensorsView(i).velocity[2];
  }

  struct calculateResidualNormSquared {};
  KOKKOS_INLINE_FUNCTION void operator()(calculateResidualNormSquared, const size_t i, exafmm::real_t &lsum1, exafmm::real_t &lsum2, exafmm::real_t &lsum3) const
  {
    lsum1 += _sensorsView(i).X_old[0] * _sensorsView(i).X_old[0];
    lsum2 += _sensorsView(i).X_old[1] * _sensorsView(i).X_old[1];
    lsum3 += _sensorsView(i).X_old[2] * _sensorsView(i).X_old[2];
  }

  struct updateP {};
  KOKKOS_INLINE_FUNCTION void operator()(updateP, const size_t i) const
  {
    // Update alphas (x_(k+1) = x_k + alpha_k * p_k) // Here we store x in dadt_old (alpha) and p in alpha
    _sensorsView(i).alpha[0] = _sensorsView(i).X_old[0] + _beta[0] * _sensorsView(i).alpha[0];
    _sensorsView(i).alpha[1] = _sensorsView(i).X_old[1] + _beta[1] * _sensorsView(i).alpha[1];
    _sensorsView(i).alpha[2] = _sensorsView(i).X_old[2] + _beta[2] * _sensorsView(i).alpha[2];
  }

  struct storeAlphaValues {};
  KOKKOS_INLINE_FUNCTION void operator()(storeAlphaValues, const size_t i) const
  {
    _sensorsView(i).alpha[0] = _sensorsView(i).dadt_old[0];
    _sensorsView(i).alpha[1] = _sensorsView(i).dadt_old[1];
    _sensorsView(i).alpha[2] = _sensorsView(i).dadt_old[2];
  }

  void RBF(exafmm::Bodies &particles, exafmm::Bodies &newPoints)
  {
    const exafmm::real_t tol = 1e-10;
    // Init both trees reset particles to newPoints
    init(particles);
    initSensors(newPoints);
    // The vorticity data stored inside velocity_old here
    calculateVorticityDualTree();
    copySensors2Host(newPoints);
    newPoints.erase(std::remove_if(newPoints.begin(), newPoints.end(), [](const auto &b) { return (std::abs(b.velocity_old[0]) < 1e-6) && (std::abs(b.velocity_old[1]) < 1e-6) && (std::abs(b.velocity_old[2]) < 1e-6); }), newPoints.end());
    createSensorTree(sensorOnly(), newPoints);
    // Set the initial alpha values by vorticity * vol
    Kokkos::parallel_for("initAlphaValues", Kokkos::RangePolicy<Device, initAlphaValues>(Device(), 0, _sensorsView.extent(0)), *this);
    // calculate the new vorcities of new points. dat stored in the velocity
    calculateVorticitySingleTree();
    // Residuals in all three dimension, each dimension has a separate residual
    exafmm::real_t res0 = 0.0;
    exafmm::real_t res1 = 0.0;
    exafmm::real_t res2 = 0.0;
    // Store residuals in X_old why not?
    // Update alpha values (alpha= omega_target - omega_current) set initial residual // ParReduce
    Kokkos::parallel_reduce("setInitialResidualAndUpdateAlpha", Kokkos::RangePolicy<Device, setInitialResidualAndUpdateAlpha>(Device(), 0, _sensorsView.extent(0)), *this, res0, res1, res2);
    //  Now CG calcualtion
    for (int i = 0; i < 100; i++) {
      //  Ap calculation :)
      calculateVorticitySingleTree();
      // p^T * Ap
      exafmm::real_t pAp0 = 0.0;
      exafmm::real_t pAp1 = 0.0;
      exafmm::real_t pAp2 = 0.0;
      Kokkos::parallel_reduce("calculatePAP", Kokkos::RangePolicy<Device, calculatePAP>(Device(), 0, _sensorsView.extent(0)), *this, pAp0, pAp1, pAp2);
      // calculate CG alphas
      _alpha[0] = (res0 > Kokkos::Experimental::epsilon_v<exafmm::real_t> && Kokkos::abs(pAp0) > Kokkos::Experimental::epsilon_v<exafmm::real_t>) ? res0 / pAp0 : 0.0;
      _alpha[1] = (res1 > Kokkos::Experimental::epsilon_v<exafmm::real_t> && Kokkos::abs(pAp1) > Kokkos::Experimental::epsilon_v<exafmm::real_t>) ? res1 / pAp1 : 0.0;
      _alpha[2] = (res2 > Kokkos::Experimental::epsilon_v<exafmm::real_t> && Kokkos::abs(pAp2) > Kokkos::Experimental::epsilon_v<exafmm::real_t>) ? res2 / pAp2 : 0.0;
      //   Update alphas stored in dadt_old x_(k+1) = x_k + alpha_k * p_k
      //  Update residuals stored in X_old r_(k+1) = r_k - alpha_k * Ap_k
      Kokkos::parallel_for("updateAlphasAndResiduals", Kokkos::RangePolicy<Device, updateAlphasAndResiduals>(Device(), 0, _sensorsView.extent(0)), *this);
      exafmm::real_t res0_kp1 = 0.0;
      exafmm::real_t res1_kp1 = 0.0;
      exafmm::real_t res2_kp1 = 0.0;
      Kokkos::parallel_reduce("calculateResidualNormSquared", Kokkos::RangePolicy<Device, calculateResidualNormSquared>(Device(), 0, _sensorsView.extent(0)), *this, res0_kp1, res1_kp1, res2_kp1);
      std::cout << "Iteration " << i << " Residuals " << res0_kp1 << " " << res1_kp1 << " " << res2_kp1 << std::endl;
      if (res0_kp1 < tol && res1_kp1 < tol && res2_kp1 < tol) {
        break;
      }
      // Calculate beta_k = r_(k+1) * r_(k+1) / r_k * r_k
      _beta[0] = (res0_kp1 > Kokkos::Experimental::epsilon_v<exafmm::real_t> && res0 > Kokkos::Experimental::epsilon_v<exafmm::real_t>) ? res0_kp1 / res0 : 0.0;
      _beta[1] = (res1_kp1 > Kokkos::Experimental::epsilon_v<exafmm::real_t> && res1 > Kokkos::Experimental::epsilon_v<exafmm::real_t>) ? res1_kp1 / res1 : 0.0;
      _beta[2] = (res2_kp1 > Kokkos::Experimental::epsilon_v<exafmm::real_t> && res2 > Kokkos::Experimental::epsilon_v<exafmm::real_t>) ? res2_kp1 / res2 : 0.0;
      // Store new residuals
      res0 = res0_kp1;
      res1 = res1_kp1;
      res2 = res2_kp1;
      // Update p_k+1 = r_(k+1) + beta_k * p_k
      Kokkos::parallel_for("updateP", Kokkos::RangePolicy<Device, updateP>(Device(), 0, _sensorsView.extent(0)), *this);
    }
    // Set alpha values of newPoints
    Kokkos::parallel_for("storeAlphaValues", Kokkos::RangePolicy<Device, storeAlphaValues>(Device(), 0, _sensorsView.extent(0)), *this);
    Kokkos::fence();
    copySensors2Host(newPoints);
  }

  void advance(exafmm::Bodies &bodies, const exafmm::real_t dt)
  {
    if constexpr (std::is_same<IntegrationScheme, Euler>::value) {
      _dt = dt;
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("EulerStep", Kokkos::RangePolicy<Device, stepEuler>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
    }
    if constexpr (std::is_same<IntegrationScheme, RK2>::value) {
      _dt = dt;
      init(bodies);
      Kokkos::parallel_for("storeInitialState", Kokkos::RangePolicy<Device, storeInitialState>(Device(), 0, _bodiesView.extent(0)), *this);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK21", Kokkos::RangePolicy<Device, stepRK21>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK22", Kokkos::RangePolicy<Device, stepRK22>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
    }
    if constexpr (std::is_same<IntegrationScheme, RK3>::value) {
      _dt = dt;
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK31", Kokkos::RangePolicy<Device, stepRK31>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK32", Kokkos::RangePolicy<Device, stepRK32>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK33", Kokkos::RangePolicy<Device, stepRK33>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
    }
    if constexpr (std::is_same<IntegrationScheme, RK4>::value) {
      _dt = dt;
      init(bodies);
      Kokkos::parallel_for("storeInitialState", Kokkos::RangePolicy<Device, storeInitialState>(Device(), 0, _bodiesView.extent(0)), *this);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK41", Kokkos::RangePolicy<Device, stepRK41>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK42", Kokkos::RangePolicy<Device, stepRK42>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK43", Kokkos::RangePolicy<Device, stepRK43>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
      init(bodies);
      calculate();
      calculateViscosity();
      Kokkos::fence();
      Kokkos::parallel_for("stepRK44", Kokkos::RangePolicy<Device, stepRK44>(Device(), 0, _bodiesView.extent(0)), *this);
      Kokkos::fence();
      copyBodies2Host(bodies);
    }
  }
};
} // namespace Vortex
#endif
