#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <matplot/matplot.h>
#include <vector>
#include "Kokkos_Timer.hpp"
#include "fmmCalculator.hpp"
#include "traverse.hpp"

Eigen::Vector3d vortexLineUnitVelocity(std::pair<const Eigen::Vector3d &, const Eigen::Vector3d &> line, const Eigen::Vector3d &targetPoint)
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

  std::pair<const Eigen::Vector3d &, const Eigen::Vector3d &> getPanelVortexLine(const unsigned panelID, const unsigned lineID)
  {
    assert(lineID < 4);
    if (lineID == 0) {
      return {TE_Vertices[panelID + 1], TE_Vertices[panelID]};
    } else if (lineID == 1) {
      return {TE_Vertices[panelID], LE_Vertices[panelID]};
    } else if (lineID == 2) {
      return {LE_Vertices[panelID], LE_Vertices[panelID + 1]};
    } else {
      return {LE_Vertices[panelID + 1], TE_Vertices[panelID + 1]};
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
  typedef Vortex::FMMCalculator<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace, Vortex::RK3, Vortex::Inviscid, Vortex::cVPM, Vortex::Transposed> FMMCalculator;

  Kokkos::ScopeGuard           guard(argc, argv);
  FMMCalculator                fmmCalculator;
  static const Eigen::Vector3d freestreamVelocity(0.0, 1.0, 0.0);

  exafmm::Bodies particles;

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

  const double     dt        = 0.01;
  static const int numPanels = 251;

  Eigen::VectorXd gamma_old;
  gamma_old.resize(numPanels);
  gamma_old.setZero();

  for (int time = 0; time < 100000; time++) {
    Eigen::Matrix3d rotation;

    const double angle = 10;

    Wing wing;

    rotation = Eigen::AngleAxisd(deg2rad(-angle), Eigen::Vector3d::UnitX()).toRotationMatrix();

    const double span  = 2.0;
    const double dx    = span / numPanels;
    const double chord = 0.1;

    for (int i = 0; i < numPanels + 1; i++) {
      wing.addVertexCouple(rotation * Eigen::Vector3d{i * dx, 0, 0} - dt * time * freestreamVelocity, rotation * Eigen::Vector3d{i * dx, chord, 0} - dt * time * freestreamVelocity);
    }

    Eigen::MatrixXd AIC = Eigen::MatrixXd::Zero(wing.getPanelCount(), wing.getPanelCount());

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
      int mid = sensors.size() / 2;
      std::cout << "Induced Velocity " << sensors[mid].velocity[0] << " " << sensors[mid].velocity[1] << " " << sensors[mid].velocity[2] << std::endl;
    }

    AIC.setZero(); // Ensure AIC is initialized properly
    for (unsigned i = 0; i < wing.getPanelCount(); i++) {
      for (unsigned j = 0; j < wing.getPanelCount(); j++) {
        const Eigen::Vector3d &normal_j        = wing.normals[j];
        const Eigen::Vector3d &control_point_j = wing.controlPoints[j];
        // std::cout << "Control Point " << control_point_j.transpose() << std::endl;
        for (unsigned e = 0; e < 4; e++) {
          auto            vortex_line      = wing.getPanelVortexLine(i, e);
          Eigen::Vector3d induced_velocity = vortexLineUnitVelocity(vortex_line, control_point_j);
          // std::cout << "Induced Velocity " << induced_velocity.transpose() << std::endl;
          AIC(i, j) += induced_velocity.dot(normal_j);
        }
      }
    }
    // std::cout << "AIC" << std::endl;
    // std::cout << AIC << std::endl;

    Eigen::VectorXd rhs;
    rhs.resize(wing.getPanelCount());

    for (unsigned i = 0; i < wing.getPanelCount(); i++)
      rhs[i] = -freestreamVelocity.dot(wing.normals[i]) - sensors[i].velocity[0] * wing.normals[i][0] - sensors[i].velocity[1] * wing.normals[i][1] - sensors[i].velocity[2] * wing.normals[i][2];
    //
    Eigen::MatrixXd     AIC_inverse = AIC.inverse();
    Eigen::VectorXd     gamma       = AIC_inverse * rhs;
    std::vector<double> inflow;
    for (unsigned i = 0; i < wing.getPanelCount(); i++) {
      inflow.push_back(sensors[i].velocity[2]);
    }

    // matplot::plot(gamma);
    // matplot::title("Gamma");
    // matplot::xlabel("Panel ID");
    // matplot::ylabel("Gamma");
    // matplot::grid(true);
    // matplot::show();

    std::cout << "Maximum GAMMA " << gamma.maxCoeff() << std::endl;
    std::cout << "Minimum GAMMA " << gamma.minCoeff() << std::endl;
    std::cout << "Angle = " << angle << " lift " << std::endl;

    Eigen::VectorXd aaa = AIC * gamma - rhs;
    std::cout << "Residual = " << aaa.norm() << std::endl;

    Eigen::Vector3d force = Eigen::Vector3d::Zero();

    for (unsigned i = 0; i < wing.getPanelCount(); i++) {
      const auto LE  = wing.getPanelVortexLine(i, 2);
      const auto dxx = LE.second - LE.first;
      // Induced velocities are required?
      Eigen::Vector3d pointVelocity = Eigen::Vector3d::Zero();
      for (unsigned j = 0; j < wing.getPanelCount(); j++) {
        pointVelocity += vortexLineUnitVelocity(wing.getPanelVortexLine(j, 0), wing.controlPoints[i]) * gamma[j];
        pointVelocity += vortexLineUnitVelocity(wing.getPanelVortexLine(j, 1), wing.controlPoints[i]) * gamma[j];
        pointVelocity += vortexLineUnitVelocity(wing.getPanelVortexLine(j, 2), wing.controlPoints[i]) * gamma[j];
        pointVelocity += vortexLineUnitVelocity(wing.getPanelVortexLine(j, 3), wing.controlPoints[i]) * gamma[j];
      }
      pointVelocity[0] += freestreamVelocity[0];
      pointVelocity[1] += freestreamVelocity[1];
      pointVelocity[2] += freestreamVelocity[2];
      pointVelocity[0] += sensors[i].velocity[0];
      pointVelocity[1] += sensors[i].velocity[1];
      pointVelocity[2] += sensors[i].velocity[2];
      force += gamma[i] * pointVelocity.cross(dxx);
    }
    std::cout << "Force = " << force.transpose() << std::endl;

    for (auto p = 0ul; p < wing.getPanelCount(); p++) {
      // Get TE line
      const auto            TE       = wing.getPanelVortexLine(p, 0);
      const Eigen::Vector3d dxx      = TE.second - TE.first;
      const Eigen::Vector3d midpoint = (TE.first + TE.second) / 2.0;
      exafmm::Body          particle;
      particle.X[0] = midpoint[0] + freestreamVelocity[0] * dt;
      particle.X[1] = midpoint[1] + freestreamVelocity[1] * dt;
      particle.X[2] = midpoint[2] + freestreamVelocity[2] * dt;
      particle.alpha[0] = dxx[0] * (gamma[p]-gamma_old[p]);
      particle.alpha[1] = dxx[1] * (gamma[p]-gamma_old[p]);
      particle.alpha[2] = dxx[2] * (gamma[p]-gamma_old[p]);
      // particle.alpha[0] = 0;
      // particle.alpha[1] = 0;
      // particle.alpha[2] = 0;
      //
      particle.radius = dxx.norm() * 2.5;
      if (p > 0) {
        const auto            left = wing.getPanelVortexLine(p, 1);
        const Eigen::Vector3d dxx2 = left.second - left.first;
        particle.alpha[0] += dxx2[0] * (gamma[p] - gamma[p - 1]) * 0.5;
        particle.alpha[1] += dxx2[1] * (gamma[p] - gamma[p - 1]) * 0.5;
        particle.alpha[2] += dxx2[2] * (gamma[p] - gamma[p - 1]) * 0.5;
      }
      if (p < wing.getPanelCount() - 1) {
        const auto            right = wing.getPanelVortexLine(p, 3);
        const Eigen::Vector3d dxx3  = right.second - right.first;
        particle.alpha[0] += dxx3[0] * (gamma[p] - gamma[p + 1]) * 0.5;
        particle.alpha[1] += dxx3[1] * (gamma[p] - gamma[p + 1]) * 0.5;
        particle.alpha[2] += dxx3[2] * (gamma[p] - gamma[p + 1]) * 0.5;
      }
      if (p == 0) {
        const auto            left = wing.getPanelVortexLine(p, 1);
        const Eigen::Vector3d dxx2 = left.second - left.first;
        particle.alpha[0] += dxx2[0] * gamma[p];
        particle.alpha[1] += dxx2[1] * gamma[p];
        particle.alpha[2] += dxx2[2] * gamma[p];
      }
      if (p == wing.getPanelCount() - 1) {
        const auto            right = wing.getPanelVortexLine(p, 3);
        const Eigen::Vector3d dxx2  = right.second - right.first;
        particle.alpha[0] += dxx2[0] * gamma[p];
        particle.alpha[1] += dxx2[1] * gamma[p];
        particle.alpha[2] += dxx2[2] * gamma[p];
      }
      particles.push_back(particle);
    }

    if (!particles.empty()) {
      fmmCalculator.advance(particles, dt);
    }

    gamma_old = gamma;
    writeTovtk(time);
  }

  return 0;
}
