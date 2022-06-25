#include "flightlib/common/command.hpp"


namespace flightlib {

Command::Command()
  : t(0.0),
    thrusts(0.0, 0.0, 0.0, 0.0),
    collective_thrust(0.0),
    omega(0.0, 0.0, 0.0),
    p(0.0, 0.0, 0.0),
    v(0.0, 0.0, 0.0),
    cmd_mode(1),
    need_position_control_(false) {}

Command::~Command() {}

bool Command::setCmdMode(const int mode) {
  if (mode != 0 && mode != 1 && mode != 2) {
    return false;
  }
  cmd_mode = mode;
  return true;
}

void Command::setPostionControl(const bool flag) {
  need_position_control_ = flag;
}

bool Command::valid() const {
  return std::isfinite(t) &&
         (
          (p.allFinite() && v.allFinite() && std::isfinite(yaw) && need_position_control_) ||
          (std::isfinite(collective_thrust) && R.allFinite() &&
           (cmd_mode == quadcmd::THRUSTATT)) ||
          (std::isfinite(collective_thrust) && omega.allFinite() &&
           (cmd_mode == quadcmd::THRUSTRATE)) ||
          (thrusts.allFinite() && (cmd_mode == quadcmd::SINGLEROTOR)));
}

bool Command::isSingleRotorThrusts() const {
  return (cmd_mode == quadcmd::SINGLEROTOR) && thrusts.allFinite();
}

bool Command::isThrustRates() const {
  return (cmd_mode == quadcmd::THRUSTRATE) &&
         (std::isfinite(collective_thrust) && omega.allFinite());
}

bool Command::isThrustAttitude() const {
  return (cmd_mode == quadcmd::THRUSTATT) &&
         (std::isfinite(collective_thrust) && R.allFinite());
}

bool Command::needPositionControl() const {
  return need_position_control_;
}


void Command::setZeros() {
  t = 0.0;
  collective_thrust = 0;
  yaw = 0;

  R = Matrix<3, 3>::Zero();
  p = Vector<3>::Zero();
  v = Vector<3>::Zero();
  thrusts = Vector<4>::Zero();
  omega = Vector<3>::Zero();

}

}  // namespace flightlib
