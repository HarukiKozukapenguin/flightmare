#pragma once

#include "flightlib/common/command.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/dynamics/quadrotor_dynamics.hpp"
#include <random>

namespace flightlib {

class LowLevelControllerSimple {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LowLevelControllerSimple(QuadrotorDynamics quad_dynamics);
  bool setCommand(const Command& cmd);
  Vector<4> run(const QuadState& state);
  bool updateQuadDynamics(const QuadrotorDynamics& quad_dynamics);
  bool randomizeKpeuler();
  inline Scalar getTime_constant(void) const { return time_constant_;}

 private:
  // Quadrotor properties
  Matrix<4, 4> B_allocation_;
  Matrix<4, 4> B_allocation_inv_;

  // P gain for body rate control
  Matrix<3, 3> Kp_rate_;

  // P gain for euler attitude control,  D gain for body rate
  Matrix<3, 3> Kp_euler_;
  Matrix<3, 3> Kd_rate_;
  Scalar small_time_constant_, large_time_constant_, init_time_constant_, init_kpeuler_, time_constant_;
  std::uniform_real_distribution<Scalar> uniform_dist_one_direction_{0.0, 1.0};
  std::random_device rd_;
  std::mt19937 random_gen_{rd_()};

  // const Matrix<3, 3> Kinv_ang_vel_tau_ =
  //   Vector<3>(20.0, 20.0, 40.0).asDiagonal();

  // Quadcopter to which the controller is applied
  QuadrotorDynamics quad_dynamics_;

  // Motor speeds calculated by the controller
  Vector<4> motor_omega_des_;

  // Command
  Command cmd_;
};

}  // namespace flightlib
