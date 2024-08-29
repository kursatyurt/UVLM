#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <matplot/matplot.h>
#include <vector>
#include "Kokkos_Timer.hpp"
#include "fmmCalculator.hpp"
#include "traverse.hpp"

Eigen::Vector3d vortexLineUnitVelocity(std::pair<const Eigen::Vector3d, const Eigen::Vector3d> line, const Eigen::Vector3d &targetPoint)
{
  // see Katz and Plotkin p.255
  Eigen::Vector3d     r0     = line.second - line.first;
  Eigen::Vector3d     r1     = targetPoint - line.first;
  Eigen::Vector3d     r2     = targetPoint - line.second;
  Eigen::Vector3d     d      = r1.cross(r2);
  double              d_len  = d.norm();
  double              r1_len = r1.norm();
  double              r2_len = r2.norm();
  static const double eps    = 1e-10; // cut off length
  if (d_len * d_len < eps || r1_len < eps || r2_len < eps) {
    return Eigen::Vector3d::Zero();
  }
  const double K = 1. / 4. / M_PI / d.squaredNorm() *
                   r0.dot(r1 / r1.norm() - r2 / r2.norm());

  return K * d;
}

struct Wing {
  std::vector<Eigen::Vector3d> LE_Vertices;
  std::vector<Eigen::Vector3d> TE_Vertices;
  std::vector<Eigen::Vector3d> normals;
  std::vector<double>          areas;
  std::vector<Eigen::Vector3d> controlPoints;

  inline auto getPanelCount() const
  {
    return LE_Vertices.size() - 1;
  }

  void addVertexCouple(const Eigen::Vector3d &LE, const Eigen::Vector3d &TE)
  {
    LE_Vertices.push_back(LE);
    TE_Vertices.push_back(TE);
  }

  void calculateTopology()
  {
    normals.resize(getPanelCount());
    areas.resize(getPanelCount());
    controlPoints.resize(getPanelCount());

    for (auto i = 0ul; i < getPanelCount(); i++) {
      controlPoints[i]   = (LE_Vertices[i] + LE_Vertices[i + 1] + TE_Vertices[i] + TE_Vertices[i + 1]) / 4.0;
      Eigen::Vector3d v1 = LE_Vertices[i + 1] - TE_Vertices[i];
      Eigen::Vector3d v2 = LE_Vertices[i] - TE_Vertices[i + 1];
      normals[i]         = v2.cross(v1);
      areas[i]           = 0.5 * normals[i].norm();
      normals[i].normalize();
    }
  };

  std::pair<const Eigen::Vector3d, const Eigen::Vector3d> getPanelVortexLine(const unsigned panelID, const unsigned lineID)
  {
    assert(lineID < 4);
    if (lineID == 0) {
      return {TE_Vertices[panelID], TE_Vertices[panelID + 1]};
    } else if (lineID == 1) {
      return {TE_Vertices[panelID + 1], LE_Vertices[panelID + 1]};
    } else if (lineID == 2) {
      return {LE_Vertices[panelID + 1], LE_Vertices[panelID]};
    } else {
      return {LE_Vertices[panelID], TE_Vertices[panelID]};
    }
  }
};

inline double deg2rad(double deg)
{
  return deg * M_PI / 180.0;
}
// https://csimaoferreira.github.io/Rotor-Wake-Aerodynamics-Lifting-Line/#/8

void testVelocity()
{
  std::pair<Eigen::Vector3d, Eigen::Vector3d> line = {Eigen::Vector3d{0, 0, 0}, Eigen::Vector3d{0, 0, 1}};
  Eigen::Vector3d                             targetPoint{0.0, 0.5, 0.5};
  Eigen::Vector3d                             velocity = vortexLineUnitVelocity(line, targetPoint);
  std::cout << "Velocity = " << velocity.transpose() << std::endl;
  std::cout << "Expected = [-0.225, 0, 0]" << std::endl;

  line        = {Eigen::Vector3d{1.0, 0.0, 0.0}, Eigen::Vector3d{0.0, 1.0, 0.0}};
  targetPoint = {0.0, 0.0, 1.0};
  velocity    = vortexLineUnitVelocity(line, targetPoint);
  std::cout << "Velocity = " << velocity.transpose() << std::endl;
  std::cout << "Expected = [0.038, 0.038, 0.038]" << std::endl;

  line        = {Eigen::Vector3d{1.0, 0.0, 0.0}, Eigen::Vector3d{0.0, 1.0, 0.0}};
  targetPoint = {0.5, 0.5, 0.5};
  velocity    = vortexLineUnitVelocity(line, targetPoint);
  std::cout << "Velocity = " << velocity.transpose() << std::endl;
  std::cout << "Expected = [0.184, 0.184, 0.000]" << std::endl;

  line        = {Eigen::Vector3d{1.0, 0.0, 1.0}, Eigen::Vector3d{0.0, 0.0, 1.0}};
  targetPoint = {0.5, 0.5, 0.5};
  velocity    = vortexLineUnitVelocity(line, targetPoint);
  std::cout << "Velocity = " << velocity.transpose() << std::endl;
  std::cout << "Expected = [0.000, -0.092, -0.092]" << std::endl;

  line        = {Eigen::Vector3d{1.0, 0.0, 1.0}, Eigen::Vector3d{0.0, 0.0, 1.0}};
  targetPoint = {1.5, 1.5, 1.5};
  velocity    = vortexLineUnitVelocity(line, targetPoint);
  std::cout << "Velocity = " << velocity.transpose() << std::endl;
  std::cout << "Expected = [0.000, 0.006, -0.018]" << std::endl;
}

int main(int argc, char **argv)
{
  // testVelocity();
  // return 0;
  typedef Vortex::FMMCalculator<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace, Vortex::RK3, Vortex::Inviscid, Vortex::rVPM, Vortex::Transposed> FMMCalculator;

  Kokkos::ScopeGuard guard(argc, argv);
  FMMCalculator      fmmCalculator;
  exafmm::Bodies     particles;

  auto writeTovtk = [&particles](int step) {
    std::ofstream file;
    file.open("output" + std::to_string(step) + ".vtk");
    file << "# vtk DataFile Version 3.0\n";
    file << "vtk output\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";
    file << "POINTS " << particles.size() << " double\n";
    for (size_t b = 0; b < particles.size(); b++) {
      file << particles[b].X[0] << " " << particles[b].X[1] << " " << particles[b].X[2] << "\n";
    }
    // Add points as vertices
    file << "VERTICES " << particles.size() << " " << 2 * particles.size() << "\n";
    for (size_t b = 0; b < particles.size(); b++) {
      file << "1 " << b << "\n";
    }
    file << "POINT_DATA " << particles.size() << "\n";

    // Add alpha vectors
    file << "VECTORS alpha double\n";
    for (const auto &body : particles) {
      file << body.alpha[0] << " " << body.alpha[1] << " " << body.alpha[2] << "\n";
    }

    // Add velocity vectors
    file << "VECTORS velocity double\n";
    for (const auto &body : particles) {
      file << body.velocity[0] << " " << body.velocity[1] << " " << body.velocity[2] << "\n";
    }

    // Add radius scalars
    file << "SCALARS radius double\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto &body : particles) {
      file << body.radius << "\n";
    }
    file.close();
  };

  fmmCalculator._nu = 1e-5;

  static const Eigen::Vector3d bodyVelocity(-1.0, 0.0, -0.1);

  static const int numPanels = 64;
  const double     dt        = 5.0 / numPanels;

  Eigen::VectorXd gamma_old;
  gamma_old.resize(numPanels);
  gamma_old.setZero();
  Eigen::MatrixXd AIC = Eigen::MatrixXd::Zero(numPanels, numPanels);
  Eigen::MatrixXd AIC_inverse;
  for (int time = 0; time < 100000; time++) {
    Eigen::Matrix3d rotation;

    // const double angle = std::sin(0.1 * time) * 5.0;

    const double angle = 0.0;
    Wing         wing;

    rotation = Eigen::AngleAxisd(deg2rad(-angle), Eigen::Vector3d::UnitY()).toRotationMatrix();

    const double span  = 5.0;
    const double dx    = span / numPanels;
    const double chord = 1.0;

    for (int i = 0; i < numPanels + 1; i++) {
      // wing.addVertexCouple(rotation * Eigen::Vector3d{0, i * dx, 0} - dt * time * bodyVelocity, rotation * Eigen::Vector3d{chord, i * dx, 0} - dt * time * bodyVelocity);
      // wing.addVertexCouple(rotation * Eigen::Vector3d{0, i * dx, 0} - dt * time * bodyVelocity, rotation * Eigen::Vector3d{std::max(0.0, chord * std::sqrt(1.0 - (2.0 * i * dx / span - 1.0) * (2.0 * i * dx / span - 1.0))), i * dx, 0} - dt * time * bodyVelocity);
      // wing.addVertexCouple(rotation * Eigen::Vector3d{std::min(-0.01 * chord, -chord * std::sqrt(1.0 - (2.0 * i * dx / span - 1.0) * (2.0 * i * dx / span - 1.0))), i * dx, 0} - dt * time * bodyVelocity, rotation * Eigen::Vector3d{0, i * dx, 0} - dt * time * bodyVelocity);
      wing.addVertexCouple(rotation * Eigen::Vector3d{0.25 * std::min(-0.025 * chord, -chord * std::sqrt(1.0 - (2.0 * i * dx / span - 1.0) * (2.0 * i * dx / span - 1.0))), i * dx, 0} + dt * time * bodyVelocity,
                           rotation * Eigen::Vector3d{0.75 * std::max(0.025 * chord, chord * std::sqrt(1.0 - (2.0 * i * dx / span - 1.0) * (2.0 * i * dx / span - 1.0))), i * dx, 0} + dt * time * bodyVelocity);
    }

    std::cout << "Time = " << time << std::endl;

    wing.calculateTopology();

    exafmm::Bodies sensors;

    for (auto k = 0ul; k < wing.controlPoints.size(); k++) {
      exafmm::Body sensor;
      sensor.X[0] = wing.controlPoints[k][0];
      sensor.X[1] = wing.controlPoints[k][1];
      sensor.X[2] = wing.controlPoints[k][2];
      sensors.push_back(sensor);
    }
    if (time > 0) {
      fmmCalculator.getSensorData(sensors);
    }

    if (time == 0) {
      AIC.setZero();
#pragma omp parallel for collapse(2)
      for (unsigned i = 0; i < wing.getPanelCount(); i++) {
        for (unsigned j = 0; j < wing.getPanelCount(); j++) {
          const Eigen::Vector3d &normal_j         = wing.normals[j];
          const Eigen::Vector3d &control_point_j  = wing.controlPoints[j];
          Eigen::Vector3d        induced_velocity = vortexLineUnitVelocity(wing.getPanelVortexLine(i, 0), control_point_j) + vortexLineUnitVelocity(wing.getPanelVortexLine(i, 1), control_point_j) + vortexLineUnitVelocity(wing.getPanelVortexLine(i, 2), control_point_j) + vortexLineUnitVelocity(wing.getPanelVortexLine(i, 3), control_point_j);
          AIC(i, j)                               = induced_velocity.dot(normal_j);
        }
      }
    }
    Eigen::VectorXd rhs;
    rhs.resize(wing.getPanelCount());

    for (unsigned i = 0; i < wing.getPanelCount(); i++) {
      rhs[i] = bodyVelocity.dot(wing.normals[i]) - sensors[i].velocity[0] * wing.normals[i][0] - sensors[i].velocity[1] * wing.normals[i][1] - sensors[i].velocity[2] * wing.normals[i][2];
    }

    for (unsigned i = 0; i < wing.getPanelCount(); i++) {
      for (unsigned j = 0; j < wing.getPanelCount(); j++) {
        const Eigen::Vector3d &normal_j        = wing.normals[j];
        const Eigen::Vector3d &control_point_j = wing.controlPoints[j];
        rhs[i] += gamma_old[i] * vortexLineUnitVelocity(wing.getPanelVortexLine(i, 0), control_point_j).dot(normal_j);
      }
    }

    if (time == 0) {
      AIC_inverse = AIC.inverse();
    }
    Eigen::VectorXd gamma = AIC_inverse * rhs;

    std::cout << "Maximum GAMMA " << gamma.maxCoeff() << std::endl;
    std::cout << "Minimum GAMMA " << gamma.minCoeff() << std::endl;

    // Shed particles
    for (auto p = 0ul; p < wing.getPanelCount(); p++) {
      // Get TE line
      const auto            TE       = wing.getPanelVortexLine(p, 0);
      const Eigen::Vector3d dxx      = TE.second - TE.first;
      const Eigen::Vector3d midpoint = (TE.first + TE.second) / 2.0;
      exafmm::Body          particle;
      particle.X[0]              = midpoint[0] - 0.5 * bodyVelocity[0] * dt;
      particle.X[1]              = midpoint[1] - 0.5 * bodyVelocity[1] * dt;
      particle.X[2]              = midpoint[2] - 0.5 * bodyVelocity[2] * dt;
      particle.alpha[0]          = dxx[0] * (gamma[p] - gamma_old[p]);
      particle.alpha[1]          = dxx[1] * (gamma[p] - gamma_old[p]);
      particle.alpha[2]          = dxx[2] * (gamma[p] - gamma_old[p]);
      const Eigen::Vector3d dxx2 = -bodyVelocity * dt;
      particle.radius            = std::min(dxx.norm(), dxx2.norm()) * 2.5;
      particles.push_back(particle);
    }
    // Trailing particles !
    {
      auto         p = 0ul;
      exafmm::Body particle;
      const auto   TE            = wing.getPanelVortexLine(p, 0);
      particle.X[0]              = TE.first[0] + 0.5 * bodyVelocity[0] * dt;
      particle.X[1]              = TE.first[1] + 0.5 * bodyVelocity[1] * dt;
      particle.X[2]              = TE.first[2] + 0.5 * bodyVelocity[2] * dt;
      const Eigen::Vector3d dxx  = TE.second - TE.first;
      const Eigen::Vector3d dxx2 = -bodyVelocity * dt;
      particle.radius            = std::min(dxx.norm(), dxx2.norm()) * 2.5;
      particle.alpha[0] += dxx2[0] * gamma[p];
      particle.alpha[1] += dxx2[1] * gamma[p];
      particle.alpha[2] += dxx2[2] * gamma[p];
      particles.push_back(particle);
    }
    for (auto p = 1ul; p < wing.getPanelCount(); p++) {
      exafmm::Body particle;
      const auto   TE            = wing.getPanelVortexLine(p, 0);
      particle.X[0]              = TE.first[0] + 0.5 * bodyVelocity[0] * dt;
      particle.X[1]              = TE.first[1] + 0.5 * bodyVelocity[1] * dt;
      particle.X[2]              = TE.first[2] + 0.5 * bodyVelocity[2] * dt;
      const Eigen::Vector3d dxx  = TE.second - TE.first;
      const Eigen::Vector3d dxx2 = -bodyVelocity * dt;
      particle.radius            = std::min(dxx.norm(), dxx2.norm()) * 2.5;
      particle.alpha[0] += dxx2[0] * (gamma[p] - gamma[p - 1]);
      particle.alpha[1] += dxx2[1] * (gamma[p] - gamma[p - 1]);
      particle.alpha[2] += dxx2[2] * (gamma[p] - gamma[p - 1]);
      particles.push_back(particle);
    }
    {
      auto         p = wing.getPanelCount() - 1;
      exafmm::Body particle;
      const auto   TE            = wing.getPanelVortexLine(p, 0);
      particle.X[0]              = TE.second[0] - 0.5 * bodyVelocity[0] * dt;
      particle.X[1]              = TE.second[1] - 0.5 * bodyVelocity[1] * dt;
      particle.X[2]              = TE.second[2] - 0.5 * bodyVelocity[2] * dt;
      const Eigen::Vector3d dxx  = TE.second - TE.first;
      const Eigen::Vector3d dxx2 = -bodyVelocity * dt;
      particle.radius            = std::min(dxx.norm(), dxx2.norm()) * 2.5;
      particle.alpha[0] -= dxx2[0] * gamma[p];
      particle.alpha[1] -= dxx2[1] * gamma[p];
      particle.alpha[2] -= dxx2[2] * gamma[p];
      particles.push_back(particle);
    }

    gamma_old = gamma;

    if (time % 10 == 0) {
      writeTovtk(time);
    }
    if (!particles.empty()) {
      fmmCalculator.advance(particles, dt);
    }
  }

  return 0;
}
