#include "flightlib/controller/lowlevel_controller_simple.hpp"

namespace flightlib {

LowLevelControllerSimple::LowLevelControllerSimple(QuadrotorDynamics quad) {
  updateQuadDynamics(quad);
}

bool LowLevelControllerSimple::updateQuadDynamics(
  const QuadrotorDynamics& quad) {
  quad_dynamics_ = quad;
  B_allocation_ = quad.getAllocationMatrix();
  B_allocation_inv_ = B_allocation_.inverse();

  Kp_rate_ = quad.kprate_.asDiagonal();
  Kd_rate_ = quad.kdrate_.asDiagonal();
  Kp_euler_ = quad.kpeuler_.asDiagonal();

  std::cout << "Kp_euler_: \n" << Kp_euler_ << std::endl;
  std::cout << "Kd_rate_: \n" << Kd_rate_ << std::endl;

  return true;
}

bool LowLevelControllerSimple::setCommand(const Command& cmd) {
  if (!cmd.valid()) return false;
  cmd_ = cmd;
  if (cmd_.isThrustRates()) {
    cmd_.collective_thrust =
      quad_dynamics_.clampCollectiveThrust(cmd_.collective_thrust);
    cmd_.omega = quad_dynamics_.clampBodyrates(cmd_.omega);
  }

  if (cmd_.isSingleRotorThrusts())
    cmd_.thrusts = quad_dynamics_.clampThrust(cmd_.thrusts);

  return true;
}


Vector<4> LowLevelControllerSimple::run(const QuadState& state) {
  Vector<4> motor_thrusts;
  if (cmd_.isThrustRates()) {
    const Vector<3> omega = state.w;
    const Scalar force = quad_dynamics_.getMass() * cmd_.collective_thrust;
    const Vector<3> omega_err = cmd_.omega - omega;
    const Vector<3> body_torque_des =
      quad_dynamics_.getJ() * Kp_rate_ * omega_err +
      omega.cross(quad_dynamics_.getJ() * omega);
    const Vector<4> thrust_torque(force, body_torque_des.x(),
                                  body_torque_des.y(), body_torque_des.z());

    motor_thrusts = B_allocation_inv_ * thrust_torque;

  } else if (cmd_.isThrustAttitude()) {

    // simple attitude controller
    // ICRA 2011, Minimum Snap Trajectory Generation and Control for Quadrotors, Mellinger

    const Vector<3> omega = state.w;
    const Scalar force = quad_dynamics_.getMass() * cmd_.collective_thrust;
    const Matrix<3, 3> R = state.R();
    const Matrix<3, 3> R_err = 0.5 * (cmd_.R.transpose() * R - R.transpose() * cmd_.R);
    const Vector<3> euler_err(-R_err(2, 1), -R_err(0, 2), -R_err(1, 0));
    const Vector<3> body_torque_des = quad_dynamics_.getJ() * (Kp_euler_ * euler_err - Kd_rate_ * omega); // we ignore term of  wxJw
    const Vector<4> thrust_torque(force, body_torque_des.x(),
                                  body_torque_des.y(), body_torque_des.z());

    motor_thrusts = B_allocation_inv_ * thrust_torque;

  } else {
    motor_thrusts = cmd_.thrusts;
  }

  motor_thrusts = quad_dynamics_.clampThrust(motor_thrusts);
  return motor_thrusts;
}


}  // namespace flightlib
