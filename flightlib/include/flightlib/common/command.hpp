#pragma once

#include "flightlib/common/types.hpp"

namespace flightlib {

namespace quadcmd {

enum CMDMODE : int {
  SINGLEROTOR = 0,
  THRUSTRATE = 1,
  THRUSTATT = 2,
};

}  // namespace quadcmd
class Command {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Command();
  ~Command();

  //
  bool valid() const;
  bool isSingleRotorThrusts() const;
  bool isThrustRates() const;
  bool isThrustAttitude() const;
  bool needPositionControl() const;

  //
  void setZeros(void);
  bool setCmdMode(const int cmd_mode);
  void setPostionControl(const bool flag);

  /// time in [s]
  Scalar t;

  /// Single rotor thrusts in [N]
  Vector<4> thrusts;

  /// Collective mass-normalized thrust in [m/s^2]
  Scalar collective_thrust;

  /// Quaternion
  Matrix<3, 3> R;

  /// Bodyrates in [rad/s]
  Vector<3> omega;

  /// goal position p
  Vector<3> p;

  /// goal velocity v
  Vector<3> v;

  Scalar yaw;


  ///
  int cmd_mode;
  bool need_position_control_;
};

}  // namespace flightlib
