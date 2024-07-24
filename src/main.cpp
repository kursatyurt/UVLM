#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <matplot/matplot.h>
#include <vector>

Eigen::Vector3d vortexLineUnitVelocity(std::pair<const Eigen::Vector3d &, const Eigen::Vector3d &> line, const Eigen::Vector3d &targetPoint)
{
  // Calculate the Biot-Savart Law for a unit vortex line
  const Eigen::Vector3d r1 = targetPoint - line.first;
  const Eigen::Vector3d r2 = targetPoint - line.second;
  const Eigen::Vector3d r0 = line.second - line.first;

  // Check singularity condition
  if (r1.norm() < 1e-6 || r2.norm() < 1e-6 || r1.cross(r2).norm() < 1e-6) {
    std::exit(1);
  }

  // Calculate the induced velocity
  return 0.25 / M_PI * r1.cross(r2) / (r1.cross(r2).squaredNorm()) * (r0.dot((r1 / r1.norm() - r2 / r2.norm())));
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
      normals[i]         = v1.cross(v2);
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

int main()
{
  // testVelocity();
  // return 0;
  int             numPanels = 500;
  Eigen::Matrix3d rotation;
  Eigen::VectorXd gamma_old;
  const double    angle = 5;

  Wing wing;

  rotation = Eigen::AngleAxisd(deg2rad(-angle), Eigen::Vector3d::UnitX()).toRotationMatrix();

  const double span  = 100000.0;
  const double dx    = span / numPanels;
  const double chord = 0.2;

  for (int i = 0; i < numPanels; i++) {
    wing.addVertexCouple(rotation * Eigen::Vector3d{i * dx, 0, 0}, rotation * Eigen::Vector3d{i * dx, chord, 0});
  }

  Eigen::MatrixXd AIC = Eigen::MatrixXd::Zero(wing.getPanelCount(), wing.getPanelCount());

  wing.calculateTopology();

  for (unsigned i = 0; i < wing.getPanelCount(); i++) {
    for (unsigned j = 0; j < wing.getPanelCount(); j++) {
      for (unsigned e = 0; e < 4; e++) {
        AIC(i, j) += vortexLineUnitVelocity(wing.getPanelVortexLine(i, e), wing.controlPoints[j]).dot(wing.normals[j]);
      }
    }
  }


  Eigen::VectorXd rhs;
  rhs.resize(wing.getPanelCount());

  Eigen::Vector3d freestreamVelocity(0, 1, 0);
  for (unsigned i = 0; i < wing.getPanelCount(); i++)
    rhs[i] = -freestreamVelocity.dot(wing.normals[i]);
  //
  Eigen::VectorXd gamma = AIC.fullPivLu().solve(rhs);

  // matplot::plot(gamma);
  // matplot::hold(matplot::on);
  // matplot::show();
  std::cout << "Maximum GAMMA " << gamma.maxCoeff() << std::endl;
  std::cout << "Angle = " << angle << " lift " << std::endl;

  Eigen::Vector3d force = Eigen::Vector3d::Zero();

  for (unsigned i = 0; i < wing.getPanelCount(); i++) {
    const auto LE  = wing.getPanelVortexLine(i, 2);
    const auto dxx = LE.second - LE.first;
    force += gamma[i] * freestreamVelocity.cross(dxx);
  }
  std::cout << "Force = " << force.transpose() << std::endl;

  return 0;
}