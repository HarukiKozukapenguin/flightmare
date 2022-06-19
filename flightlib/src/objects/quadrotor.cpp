#include "flightlib/objects/quadrotor.hpp"

namespace flightlib {

Quadrotor::Quadrotor(const std::string &cfg_path)
  : ctrl_(dynamics_),
    size_(0.2, 0.2, 0.2),
    collision_(false),
    world_box_(
      (Matrix<3, 2>() << -1000000, 1000000, -1000000, 1000000, 0, 1000000)
        .finished()) {
  // check if configuration file exist
  if (!(file_exists(cfg_path))) {
    logger_.error("Configuration file %s does not exists.", cfg_path);
  }

  YAML::Node cfg = YAML::LoadFile(cfg_path);

  // create quadrotor dynamics and update the parameters
  dynamics_.updateParams(cfg);
  init();
}

Quadrotor::Quadrotor(const QuadrotorDynamics &dynamics)
  : dynamics_(dynamics),
    ctrl_(dynamics_),
    size_(0.2, 0.2, 0.2),
    collision_(false),
    world_box_(
      (Matrix<3, 2>() << -1000000, 1000000, -1000000, 1000000, 0, 1000000)
        .finished()) {
  init();
}

Quadrotor::~Quadrotor() {}

bool Quadrotor::run(Command &cmd, const Scalar ctl_dt) {

  if (cmd.needPositionControl()) {
    updatePositionControl(state_, cmd);
  }

  if (!setCommand(cmd)) {
    logger_.error("Cannot Set Control Command");
    return false;
  };
  return run(ctl_dt);
}

bool Quadrotor::run(const Scalar ctl_dt) {
  if (!state_.valid()) {
    logger_.error("Not Valid states");
    return false;
  };
  if (!cmd_.valid()) {
    logger_.error("Not Valid action");
    return false;
  };

  QuadState old_state = state_;  // QuadState type
  QuadState next_state = state_;

  // time
  const Scalar max_dt = integrator_ptr_->dtMax();
  Scalar remain_ctl_dt = ctl_dt;

  ctrl_.setCommand(cmd_);

  // simulation loop
  while (remain_ctl_dt > 0.0) {
    const Scalar sim_dt = std::min(remain_ctl_dt, max_dt);

    const Vector<4> motor_thrusts_des = ctrl_.run(state_);
    runMotors(sim_dt, motor_thrusts_des);

    Vector<4> force_torques = B_allocation_ * motor_thrusts_;

    // Compute linear acceleration and body torque
    const Vector<3> force(0.0, 0.0, force_torques[0]);
    // compute the body drag
    const Vector<3> body_vel =
      state_.q().toRotationMatrix().transpose() * state_.v;
    const Vector<3> force_bodydrag = dynamics_.getBodyDrag(body_vel);

    // compute accleration
    state_.a =
      state_.q() * (force - force_bodydrag) * 1.0 / dynamics_.getMass() + gz_;

    // compute body torque
    state_.tau = force_torques.segment<3>(1);

    // dynamics integration
    integrator_ptr_->step(state_.x, sim_dt, next_state.x);

    //
    state_.x = next_state.x;
    remain_ctl_dt -= sim_dt;
  }

  // update state and sim time
  state_.t += ctl_dt;

  state_.qx.normalize();
  //
  // constrainInWorldBox(old_state);

  return true;
}

bool Quadrotor::updatePositionControl(const QuadState &state, Command &cmd) {

  if (!state.valid()) {
    std::cout << "State is invalid" << std::endl;
    return false;
  }

  // acc command
  Vector<3> pos_error = clip(cmd.p - state.p, dynamics_.p_err_max_);  // set clipping for stable flight
  Vector<3> vel_error = clip(cmd.v - state.v, dynamics_.v_err_max_);
  Vector<3> acc_setpoint = {0, 0, 0};  // set 0
  Vector<3> acc_cmd = dynamics_.kpacc_.cwiseProduct(pos_error) +
    dynamics_.kdacc_.cwiseProduct(vel_error) + acc_setpoint - GVEC;
  if (acc_cmd[2] < 1) acc_cmd[2] = 1;  // if 0, then unstable in z_B
  Scalar thrust_cmd = acc_cmd.norm() * dynamics_.getMass();

  // attitude command
  Quaternion q_c(Quaternion(Eigen::AngleAxis<Scalar>(cmd.yaw, Vector<3>::UnitZ())));
  Vector<3> y_c = q_c * Vector<3>::UnitY();
  Vector<3> z_B = acc_cmd.normalized();  // normalized desired z direction vector
  Vector<3> x_B = (y_c.cross(z_B)).normalized();  // normalized vector
  Vector<3> y_B = (z_B.cross(x_B)).normalized();
  Matrix<3, 3> R_W_B((Matrix<3, 3>() << x_B, y_B, z_B).finished());
  Quaternion q_des(R_W_B);  // desired quaternion by cmd.yaw and acc_cmd

  // Command command;
  cmd.t = state.t;
  cmd.collective_thrust = thrust_cmd / dynamics_.getMass();

  if (cmd.isThrustRates()) {
    // angular acceleration command
    // Attitude control method from Fohn 2020.
    const Quaternion q_e = state.q().inverse() * q_des;
    Matrix<3, 3> T_att = (Matrix<3, 3>() << dynamics_.kpatt_xy_, 0.0, 0.0, 0.0,
                          dynamics_.kpatt_xy_, 0.0, 0.0, 0.0, dynamics_.kpatt_z_)
      .finished();
    Vector<3> tmp = Vector<3>(q_e.w() * q_e.x() - q_e.y() * q_e.z(),
                              q_e.w() * q_e.y() + q_e.x() * q_e.z(), q_e.z());
    if (q_e.w() <= 0) tmp(2) *= -1.0;
    Vector<3> omega_cmd = 2.0 / std::sqrt(q_e.w() * q_e.w() + q_e.z() * q_e.z()) * T_att * tmp;

    // Command command;
    cmd.omega = omega_cmd;
  }

  if (cmd.isThrustAttitude()) {
    cmd.R = q_des.toRotationMatrix();
    // std::cout << "cmd.q: " << q_des.x() << "," << q_des.y() << ", " << q_des.z() << ", " << q_des.w() << "; cmd_euler: " << cmd.R.eulerAngles(2, 1, 0).transpose()  << "; euler: " << state_.Euler().transpose() << std::endl;
  }

  return true;
}

void Quadrotor::init() {
  // reset
  updateDynamics(dynamics_);
  reset();
}

bool Quadrotor::reset() {
  state_.setZero();
  motor_omega_.setZero();
  motor_thrusts_.setZero();
  return true;
}

bool Quadrotor::reset(const QuadState &state) {
  // std::cout << "reset in Quadrotor is called" << std::endl;
  if (!state.valid()) return false;
  state_ = state;
  motor_omega_.setZero();
  motor_thrusts_.setZero();
  // std::cout << "reset in Quadrotor is finished" << std::endl;
  return true;
}

void Quadrotor::runMotors(const Scalar sim_dt,
                          const Vector<4> &motor_thruts_des) {
  const Vector<4> motor_omega_des =
    dynamics_.motorThrustToOmega(motor_thruts_des);
  const Vector<4> motor_omega_clamped =
    dynamics_.clampMotorOmega(motor_omega_des);

  // simulate motors as a first-order system
  const Scalar c = std::exp(-sim_dt * dynamics_.getMotorTauInv());
  motor_omega_ = c * motor_omega_ + (1.0 - c) * motor_omega_clamped;

  motor_thrusts_ = dynamics_.motorOmegaToThrust(motor_omega_);
  motor_thrusts_ = dynamics_.clampThrust(motor_thrusts_);
}

bool Quadrotor::setCommand(const Command &cmd) {
  if (!cmd.valid()) {  // checked as "quadcmd::LINVEL" we cannot check whether
    logger_.error("Not Valid action");
    return false;
  }
  cmd_ = cmd;

  if (std::isfinite(cmd_.collective_thrust))
    cmd_.collective_thrust =
      dynamics_.clampCollectiveThrust(cmd_.collective_thrust);

  if (cmd_.omega.allFinite()) cmd_.omega = dynamics_.clampBodyrates(cmd_.omega);

  if (cmd_.thrusts.allFinite())
    cmd_.thrusts = dynamics_.clampThrust(cmd_.thrusts);


  return true;
}

bool Quadrotor::setState(const QuadState &state) {
  if (!state.valid()) return false;
  state_ = state;
  return true;
}

bool Quadrotor::setWorldBox(const Ref<Matrix<3, 2>> box) {
  if (box(0, 0) >= box(0, 1) || box(1, 0) >= box(1, 1) ||
      box(2, 0) >= box(2, 1)) {
    return false;
  }
  world_box_ = box;
  return true;
}

bool Quadrotor::updateBodyDragCoeff1(const Vector<3> cd1) {
  return dynamics_.updateBodyDragCoeff1(cd1);
}

bool Quadrotor::constrainInWorldBox(const QuadState &old_state) {
  if (!old_state.valid()) return false;

  // violate world box constraint in the x-axis
  if (state_.x(QS::POSX) <= world_box_(0, 0) ||
      state_.x(QS::POSX) >= world_box_(0, 1)) {
    state_.x(QS::POSX) = old_state.x(QS::POSX);
    state_.x(QS::VELX) = 0.0;
  }

  // violate world box constraint in the y-axis
  if (state_.x(QS::POSY) <= world_box_(1, 0) ||
      state_.x(QS::POSY) >= world_box_(1, 1)) {
    state_.x(QS::POSY) = old_state.x(QS::POSY);
    state_.x(QS::VELY) = 0.0;
  }

  // violate world box constraint in the z-axis
  if (state_.x(QS::POSZ) <= world_box_(2, 0) ||
      state_.x(QS::POSZ) >= world_box_(2, 1)) {
    //
    state_.x(QS::POSZ) = old_state.x(QS::POSZ);

    // reset velocity to zero
    state_.x(QS::VELZ) = 0.0;

    // reset acceleration to zero
    state_.a << 0.0, 0.0, 0.0;
    // reset angular velocity to zero
    state_.w << 0.0, 0.0, 0.0;
  }
  return true;
}

bool Quadrotor::getState(QuadState *const state) const {
  if (!state_.valid()) return false;

  *state = state_;
  return true;
}

bool Quadrotor::getMotorThrusts(Ref<Vector<4>> motor_thrusts) const {
  motor_thrusts = motor_thrusts_;
  return true;
}

Vector<4> Quadrotor::getMotorThrusts() const { return motor_thrusts_; }

bool Quadrotor::getMotorOmega(Ref<Vector<4>> motor_omega) const {
  motor_omega = motor_omega_;
  return true;
}

Vector<4> Quadrotor::getMotorOmega() const { return motor_omega_; }

bool Quadrotor::getDynamics(QuadrotorDynamics *const dynamics) const {
  if (!dynamics_.valid()) return false;
  *dynamics = dynamics_;
  return true;
}

const QuadrotorDynamics &Quadrotor::getDynamics() { return dynamics_; }

bool Quadrotor::updateDynamics(const QuadrotorDynamics &dynamics) {
  if (!dynamics.valid()) {
    std::cout << "[Quadrotor] dynamics is not valid!" << std::endl;
    return false;
  }
  dynamics_ = dynamics;
  ctrl_.updateQuadDynamics(dynamics_);

  integrator_ptr_ =
    std::make_unique<IntegratorRK4>(dynamics_.getDynamicsFunction(), 2.5e-3);
  B_allocation_ = dynamics_.getAllocationMatrix();
  B_allocation_inv_ << B_allocation_.inverse();
  return true;
}

bool Quadrotor::addRGBCamera(std::shared_ptr<RGBCamera> camera) {
  rgb_cameras_.push_back(camera);
  return true;
}

Vector<3> Quadrotor::getSize(void) const { return size_; }

Vector<3> Quadrotor::getPosition(void) const { return state_.p; }

std::vector<std::shared_ptr<RGBCamera>> Quadrotor::getCameras(void) const {
  return rgb_cameras_;
}

bool Quadrotor::getCamera(const size_t cam_id,
                          std::shared_ptr<RGBCamera> camera) const {
  if (cam_id <= rgb_cameras_.size()) {
    return false;
  }

  camera = rgb_cameras_[cam_id];
  return true;
}

bool Quadrotor::getCollision() const { return collision_; }

int Quadrotor::getNumCamera() const { return rgb_cameras_.size(); }

Vector<> Quadrotor::clip(const Vector<> &v, const Vector<> &bound) {
  return v.cwiseMin(bound).cwiseMax(-bound);
}

}  // namespace flightlib
