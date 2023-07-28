#include "flightlib/envs/vision_env/vision_env.hpp"

namespace flightlib {

VisionEnv::VisionEnv()
  : VisionEnv(getenv("FLIGHTMARE_PATH") +
                std::string("/flightpy/configs/vision/config.yaml"),
              0) {}

VisionEnv::VisionEnv(const std::string &cfg_path, const int env_id)
  : EnvBase() {
  // check if configuration file exist
  std::cout << "cfg_path: " << cfg_path << std::endl;
  if (!(file_exists(cfg_path))) {
    logger_.error("Configuration file %s does not exists.", cfg_path);
  }
  // load configuration file
  cfg_ = YAML::LoadFile(cfg_path);

  env_folder_ = cfg_["environment"]["env_folder"].as<int>();
  env_id_ = env_id;
  init();
}

VisionEnv::VisionEnv(const YAML::Node &cfg_node, const int env_id) : EnvBase() {
  cfg_ = cfg_node;
  //
  env_folder_ = cfg_["environment"]["env_folder"].as<int>();
  env_id_ = env_id;
  init();
}

void VisionEnv::init() {
  //
  is_collision_ = false;
  is_threshold_collision_ = false;
  unity_render_offset_ << 0.0, 0.0, 0.0;
  goal_linear_vel_ << 0.0, 0.0, 0.0;
  cmd_.setZeros();

  // create quadrotors

  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quad_ptr_ = std::make_shared<Quadrotor>(dynamics);


  // define input and output dimension for the environment
  obs_dim_ = visionenv::kNObs;
  act_dim_ = visionenv::kNAct;
  rew_dim_ = 0;
  num_detected_obstacles_ = visionenv::kNObstacles;

  // load parameters
  loadParam(cfg_);

  // additional paramter load
  cmd_.setCmdMode(cfg_["Control"]["cmd_mode"].as<int>());
  cmd_.setPostionControl(cfg_["Control"]["position_control"]);

  // add camera
  if (!configCamera(cfg_)) {
    logger_.error(
      "Cannot config RGB Camera. Something wrong with the config file");
  }
  changeLevel();

  // use single rotor control or bodyrate control
  // Scalar max_force = quad_ptr_->getDynamics().getForceMax();
  // Vector<3> max_omega = quad_ptr_->getDynamics().getOmegaMax();
  //
  // act_mean_ << (max_force / quad_ptr_->getMass()) / 2, 0.0, 0.0, 0.0;
  // act_std_ << (max_force / quad_ptr_->getMass()) / 2, max_omega.x(),
  //   max_omega.y(), max_omega.z();
  act_mean_ << 0, 0;
  act_std_ << max_gain_, max_gain_;  // set by my experience (cmd difference)

  collide_num = 0;
  wall_collide_num = 0;
  time_num = 0;
  bound_num = 0;
  goal_num = 0;
  iter = 0;
  current_env_steps_ = 0;
}

VisionEnv::~VisionEnv() {}

bool VisionEnv::reset(Ref<Vector<>> obs) {
  // std::cout << "reset call in VisionEnv" << std::endl;
  quad_state_.setZero();
  quad_old_state_.setZero();
  pi_act_.setZero();
  old_pi_act_.setZero();
  is_collision_ = false;
  is_threshold_collision_ = false;


  // std::uniform_real_distribution<Scalar> y_range_dist{y_lim_[0], y_lim_[1]};
  // std::uniform_real_distribution<Scalar> z_range_dist{z_lim_[0], z_lim_[1]};
  // Scalar y_lim = y_range_dist(random_gen_);
  // Scalar z_lim = z_range_dist(random_gen_);
  // world_box_[2] = -y_lim;
  // world_box_[3] = y_lim;
  // world_box_[5] = z_lim;
  // randomly reset the quadrotor state
  // reset position
  // std::uniform_real_distribution<Scalar> tree_range_dist{tree_size_range_[0],
  //                                                        tree_size_range_[1]};
  // tree_size_ = tree_range_dist(random_gen_);
  // changeLevel();
  if (!quad_size_fix_){
    randomize_size();
  }
  if (!max_gain_fix_){
    randomize_gain();
  }
  vel_compensation_ = std::sqrt(learn_max_gain_/max_gain_);
  while (true) {
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_) * 10 + 10;
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_) * world_box_[2] * 0.2;
    quad_state_.x(QS::POSZ) =
      uniform_dist_(random_gen_) * (world_box_[5] - world_box_[4]) * 0.4 +
      (world_box_[4] + world_box_[5]) / 2;

    // quad_state_.x(QS::POSX) = 52.9;
    // quad_state_.x(QS::POSY) = 7.4;
    // quad_state_.x(QS::POSZ) = 7.6;
    // reset quadrotor with random states
    is_collision_ = false;
    is_threshold_collision_ = false;
    quad_ptr_->reset(quad_state_);
    time_constant_ = quad_ptr_->getTime_constant();
    time_constant_ /= vel_compensation_;

    init_isCollision();  // change is_collision depending on initial position

    if (!is_collision_) {
      // std::cout << "not initial collision" << std::endl;
      break;
    }
    // std::cout << "initial collision" << std::endl;
    // std::cout << "initial x" << quad_state_.x(QS::POSX) << std::endl;
    // std::cout << "initial y" << quad_state_.x(QS::POSY) << std::endl;
    // std::cout << "initial z" << quad_state_.x(QS::POSZ) << std::endl;
  }
  quad_old_state_ = quad_state_;

  // std::cout << "z_vel is " << quad_state_.x(QS::VELZ) << std::endl;
  // reset control command
  act_.setZero();
  // cmd_.setZeros();
  cmd_.p[0] = quad_state_.x(QS::POSX);
  cmd_.p[1] = quad_state_.x(QS::POSY);

      if (fly_result_){
    std::cout << "quad size is " << quad_size_ << std::endl;
    std::cout << "quad size threshold is " << quad_size_threshold_ << std::endl;
    std::cout << "quad time_constant is " << time_constant_ << std::endl;
    }

  // obtain observations
  getObs(obs);
  reset_delay_buffer();
  return true;
}

void VisionEnv::randomize_size(){
  // reset quadrotor size
  // size: the size of the quadrotor, 0.25 ~ 0.5[m]
  // Scalar size_div = std::min((0.50-0.25)*(500-env_id_)/250 ,0.25);
  Scalar size_div = 0.30;
  Scalar size_r = uniform_dist_one_direction_(random_gen_)*size_div + 0.25;
  quad_size_ = size_r;
  quad_size_threshold_ = quad_size_ - quad_size_threshold_dev_ * uniform_dist_one_direction_(random_gen_);
  // resetSize(size);
}
void VisionEnv::randomize_gain(){
  // reset acc_max when reset
  max_gain_ = uniform_dist_one_direction_(random_gen_)*(range_max_gain_[1] - range_max_gain_[0]) + range_max_gain_[0];
  act_std_ << max_gain_, max_gain_;
}

bool VisionEnv::reset(Ref<Vector<>> obs, bool random) { return reset(obs); }

void VisionEnv::resetSize(Scalar size) {
  cfg_["quadrotor_dynamics"]["quad_size"] = size;
  cfg_["quadrotor_dynamics"]["mass"] = std::pow(size/init_size_,3)*init_mass_;
  for (size_t i; i < init_inertia_.size(); i++) {
    cfg_["quadrotor_dynamics"]["inertia"][i] = std::pow(size/init_size_,5)*init_inertia_[i];
  }
  Scalar tbm_x_length = size/init_size_*init_tbm_x_length_;
  Scalar tbm_y_length = size/init_size_*init_tbm_y_length_;
  std::vector<Scalar> tbm_fr{tbm_x_length, -tbm_y_length, 0.0};
  cfg_["quadrotor_dynamics"]["tbm_fr"] = tbm_fr;
  std::vector<Scalar> tbm_bl{-tbm_x_length, tbm_y_length, 0.0};
  cfg_["quadrotor_dynamics"]["tbm_bl"] = tbm_bl;
  std::vector<Scalar> tbm_br{-tbm_x_length, -tbm_y_length, 0.0};
  cfg_["quadrotor_dynamics"]["tbm_br"] = tbm_br;
  std::vector<Scalar> tbm_fl{tbm_x_length, tbm_y_length, 0.0};
  cfg_["quadrotor_dynamics"]["tbm_fl"] = tbm_fl;

  cfg_["quadrotor_dynamics"]["motor_omega_min"] = std::pow(size/init_size_,-1/2)*init_motor_omega_min_;
  cfg_["quadrotor_dynamics"]["motor_omega_max"] = std::pow(size/init_size_,-1/2)*init_motor_omega_max_;
  for (size_t i; i < init_thrust_map_.size(); i++) {
    cfg_["quadrotor_dynamics"]["thrust_map"][i] = std::pow(size/init_size_,4)*init_thrust_map_[i];
  }
  quad_size_ = size;
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quad_ptr_->updateDynamics(dynamics);
}


void VisionEnv::init_isCollision(void) {
  Vector<visionenv::Theta_Cuts> unused1;
  Vector<visionenv::kNObstaclesState> unused2;
  getObstacleState(
    unused1, unused2);  // change "is_collision_" depending on current state
}

bool VisionEnv::getObs(Ref<Vector<>> obs) {
  if (obs.size() != obs_dim_) {
    logger_.error("Observation dimension mismatch. %d != %d", obs.size(),
                  obs_dim_);
    return false;
  }

  // std::cout << "getObs is called" << std::endl;
  // compute rotation matrix
  // Vector<9> ori = Map<Vector<>>(quad_state_.R().data(), quad_state_.R().size());
  Matrix<3, 3> R = quad_state_.R();
  Vector<2> body_tilt = Vector<2>(R(0,2), R(1,2)) + Vector<2>(att_noise_*uniform_dist_(random_gen_),att_noise_*uniform_dist_(random_gen_));
  // std::cout << "ori is called" << std::endl;

  // get N most closest obstacles as the observation
  Vector<visionenv::Theta_Cuts> sphericalboxel;
  Vector<visionenv::kNObstaclesState> unused;
  // std::cout << "getObstacleState is being called" << std::endl;
  getObstacleState(sphericalboxel, unused);

  Scalar beta = linear_transition_log_;
  Scalar a = -1/beta;
  Scalar b = 1-log(beta);

  Vector<visionenv::Theta_Cuts> logsphericalboxel;
  for (int i = 0; i < visionenv::Theta_Cuts; i++) {
    if (sphericalboxel[i]> linear_transition_log_){
    logsphericalboxel[i] = -log(sphericalboxel[i]);
    }
    else{
      logsphericalboxel[i] = a*sphericalboxel[i]+b;
    }
  }

  // calculate log of the depth
  Scalar average_depth = 0;
  for (int i = 0; i < visionenv::Theta_Cuts; i++) {
    average_depth += sphericalboxel[i];
  }
  average_depth /= visionenv::Theta_Cuts;

  // std::cout << "getObstacleState is called" << std::endl;

  // std::cout << sphericalboxel << std::endl;
  Vector<3> normalized_p;  //[0,1]
  for (size_t i = 0; i < 3; i++) {
    normalized_p[i] = (quad_state_.p[i] - world_box_[i * 2]) /
                      (world_box_[i * 2 + 1] - world_box_[i * 2]);
  }
  // Observations

  obs << quad_size_, time_constant_, max_gain_, act_, quad_state_.p[0], quad_state_.p[1],
    quad_state_.v[0]*vel_compensation_ ,quad_state_.v[1], body_tilt,
    quad_state_.w[0] + omega_noise_*uniform_dist_(random_gen_), quad_state_.w[1] + omega_noise_*uniform_dist_(random_gen_),
    toLog((wall_pos_ - quad_size_) - quad_state_.x(QS::POSY), beta),
    toLog((wall_pos_ - quad_size_) + quad_state_.x(QS::POSY), beta),
    logsphericalboxel, gain_normalized_act_distance_;
  // std::cout << "obs is called" << std::endl;
  return true;
}

Scalar VisionEnv::toLog(Scalar x, Scalar beta){
  Scalar a = -1/beta;
  Scalar b = 1-log(beta);
  if (x> beta){
    return -log(x);
  }
  else{
    return a*x+b;
  }
}

bool VisionEnv::getObstacleState(
  Ref<Vector<visionenv::Theta_Cuts>> sphericalboxel,
  Ref<Vector<visionenv::kNObstaclesState>> obs_state) {
  // Scalar safty_threshold = 0.2;
  if (static_objects_.size() < 0) {
    logger_.error("No dynamic or static obstacles.");
    return false;
  }
  // make sure to reset the collision penalty
  relative_pos_norm_.clear();
  relative_2d_pos_norm_.clear();
  obstacle_radius_.clear();

  //
  quad_ptr_->getState(&quad_state_);

  // std::cout << "get quad_state_" << std::endl;

  // compute relative distance to dynamic obstacles
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> relative_pos;
  // for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
  //   // compute relative position vector
  //   Vector<3> delta_pos = dynamic_objects_[i]->getPos() - quad_state_.p;
  //   relative_pos.push_back(delta_pos);

  //   // compute relative distance
  //   Scalar obstacle_dist = delta_pos.norm();
  //   Scalar obstacle_2d_dist =
  //     std::sqrt(std::pow(delta_pos[0], 2) + std::pow(delta_pos[1], 2));
  //   // limit observation range
  //   if (obstacle_dist > max_detection_range_) {
  //     obstacle_dist = max_detection_range_;
  //   }
  //   relative_pos_norm_.push_back(obstacle_dist);
  //   relative_2d_pos_norm_.push_back(obstacle_2d_dist);

  //   // store the obstacle radius
  //   Scalar obs_radius = dynamic_objects_[i]->getScale()[0] / 2;
  //   // due to think quadsize, change obs_radius to more smaller to
  //   // move forword
  //   obs_radius = obs_radius / 2;
  //   obstacle_radius_.push_back(obs_radius);

  //   //
  //   if (obstacle_2d_dist < obs_radius + quad_size_) {
  //     is_collision_ = true;
  //   }
  // }

  // std::cout << "get dynamic_objects_" << std::endl;

  is_collision_ = false;
  is_threshold_collision_ = false;
  is_wall_collision_ = false;
  Scalar y_pos = quad_state_.x(QS::POSY);
  Scalar wall_2d_dist = std::min(wall_pos_ - y_pos, wall_pos_ + y_pos);
  if (wall_2d_dist < quad_size_) {
      is_wall_collision_ = true;
    }
  if (wall_2d_dist < quad_size_threshold_) {
      is_threshold_collision_ = true;
    }
  // compute relatiev distance to static obstacles
  for (int i = 0; i < (int)static_objects_.size(); i++) {
    // compute relative position vector
    Vector<3> delta_pos = static_objects_[i]->getPos() - quad_state_.p;
    relative_pos.push_back(delta_pos);

    // compute relative distance
    Scalar obstacle_dist = delta_pos.norm();
    Scalar obstacle_2d_dist =
      std::sqrt(std::pow(delta_pos[0], 2) + std::pow(delta_pos[1], 2));
    if (obstacle_dist > max_detection_range_) {
      obstacle_dist = max_detection_range_;
    }
    relative_pos_norm_.push_back(obstacle_dist);
    relative_2d_pos_norm_.push_back(obstacle_2d_dist);


    // store the obstacle radius
    Scalar obs_radius = static_objects_[i]->getScale()[0];
    // obs_radius = obs_radius / 2;

    obstacle_radius_.push_back(obs_radius);

    if (obstacle_2d_dist < obs_radius + quad_size_) {
      is_collision_ = true;
    }
    if (obstacle_2d_dist < obs_radius + quad_size_threshold_) {
      is_threshold_collision_ = true;
    }
  }

  // std::cout << "get static_objects_" << std::endl;

  // std::cout << relative_pos_norm_ << std::endl;
  size_t idx = 0;
  obstacle_num_ = 0;  // obstacle_num_ is declared at hpp
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> pos_b_list;
  std::vector<Scalar> pos_norm_list;
  std::vector<Scalar> obs_radius_list;
  Matrix<3, 3> R = quad_state_.R();
  Vector<3> euler = R.eulerAngles(0, 1, 2);
  if (euler[2] > M_PI / 2) euler[2] = M_PI;
  else if (euler[2] < -M_PI / 2) euler[2] = -M_PI;
  else euler[2] = 0;
  R = Eigen::AngleAxis<Scalar>(euler[0], Vector<3>::UnitX()) *
        Eigen::AngleAxis<Scalar>(euler[1], Vector<3>::UnitY()) *
        Eigen::AngleAxis<Scalar>(euler[2], Vector<3>::UnitZ());
  Matrix<3, 3> R_T = R.transpose();
  Vector<3> poll_y(R_T(0, 1), R_T(1, 1), R_T(2, 1));
  Vector<3> poll_z(R_T(0, 2), R_T(1, 2), R_T(2, 2));

  // std::vector<Vector<3>> converted_positions;
  // for (const auto &tree_pos : relative_pos) {
  //   Eigen::Vector3d converted_pos = R_T * tree_pos;
  //   converted_positions.push_back(converted_pos);
  // }
  // std::cout << "finish declaration" << std::endl;

  for (size_t sort_idx : sort_indexes(relative_2d_pos_norm_)) {
    if (idx >= 1) {
      break;
    }
    // std::cout << idx << std::endl;
    // if enough obstacles in the environment
    if (idx < relative_pos.size() &&
        relative_2d_pos_norm_[sort_idx] <= max_detection_range_) {
      // std::cout << relative_pos[sort_idx][0] << std::endl;

      // std::cout << "inputing to obs_state" << std::endl;
      for (size_t i = 0; i < 3; ++i) {
        // std::cout << i << std::endl;
        obs_state[i] = relative_pos[sort_idx][i];
      }
      // std::cout << "input relative_pos" << std::endl;
      obs_state[3] = obstacle_radius_[sort_idx];
    }
    // std::cout << "input to obs_state" << std::endl;
    idx += 1;
  }

  // std::cout << "get obs_state" << std::endl;

  idx = 0;
  for (size_t sort_idx : sort_indexes(relative_2d_pos_norm_)) {
    if (idx >= visionenv::kNObstacles) {
      obstacle_num_ = idx;
      break;
    }
    // if enough obstacles in the environment
    if (idx < relative_pos.size() &&
        relative_2d_pos_norm_[sort_idx] <= max_detection_range_) {
      pos_b_list.push_back(R_T * relative_pos[sort_idx]);
      pos_norm_list.push_back(relative_2d_pos_norm_[sort_idx]);
      obs_radius_list.push_back(obstacle_radius_[sort_idx]);
    } else {
      obstacle_num_ = idx;
      break;
    }
    idx += 1;
  }

  // std::cout << "get pos_b_list" << std::endl;

  // std::cout << "obstacle_num is " << obstacle_num << std::endl;
  // std::cout << "getsphericalboxel is being called" << std::endl;
  sphericalboxel = getsphericalboxel(pos_b_list, obs_radius_list, poll_y, poll_z, R_T);
  gain_normalized_act_distance_ = get_vel_act_boxel(pos_b_list, obs_radius_list, poll_y, poll_z, R_T);

  // vel_obs_distance_: [0,10.0] [m]
  vel_obs_distance_ =
    get_vel_sphericalboxel(pos_b_list, obs_radius_list, poll_y, poll_z, R_T);

  return true;
}

Vector<visionenv::Theta_Cuts> VisionEnv::getsphericalboxel(
  const std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> &pos_b_list,
  const std::vector<Scalar> &obs_radius_list, const Vector<3> &poll_y, const Vector<3> &poll_z,
  const Matrix<3, 3> &R_T) {
  // Vector<3> vel_2d = {quad_state_.v[0], quad_state_.v[1], 0};
  // Vector<3> body_vel = R_T * vel_2d;
  // Scalar vel_theta = std::atan2(body_vel[1], body_vel[0]);
  // Scalar vel_phi = std::atan(body_vel[2] / std::sqrt(std::pow(body_vel[0], 2)
  // +
  //                                                    std::pow(body_vel[1],
  //                                                    2)));

  Vector<visionenv::Theta_Cuts> obstacle_obs;
  for (int t = -visionenv::Theta_Cuts / 2; t < visionenv::Theta_Cuts / 2; ++t) {
    Scalar theta = (t >= 0) ? dist_theta_list_[t] : -dist_theta_list_[(-t) - 1];  //[deg]
    Scalar tcell = theta * (PI / 180);
    obstacle_obs[(t + visionenv::Theta_Cuts / 2)] =
      getClosestDistance(pos_b_list, obs_radius_list, poll_y, poll_z, tcell, 0);
  }
  return obstacle_obs;
}

Scalar VisionEnv::getClosestDistance(
  const std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> &pos_b_list,
  const std::vector<Scalar> &obs_radius_list, const Vector<3> &poll_y, const Vector<3> &poll_z,
  const Scalar &tcell, const Scalar &fcell) const {
  Vector<3> Cell = getCartesianFromAng(tcell, fcell);
  Scalar y_p = calc_dist_from_wall(1, Cell, poll_y);
  Scalar y_n = calc_dist_from_wall(-1, Cell, poll_y);
  Scalar rmin = std::min(std::min(y_p, y_n), max_detection_range_);
  for (size_t i = 0; i < obstacle_num_; ++i) {
    Vector<3> pos = pos_b_list[i];
    // make large radius depending on quad size for avoidance direction
    Scalar radius = obs_radius_list[i] + quad_size_;
    Eigen::Vector3d alpha = cross_product(Cell, poll_z);
    Eigen::Vector3d beta = cross_product(pos, poll_z);
    Scalar a = std::pow(alpha.norm(), 2);
    if (a == 0) continue;
    Scalar b = inner_product(alpha, beta);
    Scalar c = std::pow(beta.norm(), 2) - std::pow(radius, 2);
    Scalar D = std::pow(b, 2) - a * c;
    if (0 <= D) {
      Scalar dist = (b - std::sqrt(D)) / a;
      if (dist >= 0) {
        rmin = std::min(dist, rmin);
      }
    }
  }

  return rmin / max_detection_range_;
}
Scalar VisionEnv::calc_dist_from_wall(Scalar sign, const Vector<3>& Cell, const Vector<3> &poll_y) const {
  Scalar y_d= (sign*(wall_pos_- quad_size_) - quad_state_.p[1]);
  Scalar cos_theta = inner_product(Cell,poll_y);
  if (cos_theta*y_d <= 0) return max_detection_range_;
  else return y_d/cos_theta;
}

Vector<3> VisionEnv::getCartesianFromAng(const Scalar &theta,
                                         const Scalar &phi) const {
  Vector<3> cartesian = {std::cos(theta) * std::cos(phi),
                         std::sin(theta) * std::cos(phi), std::sin(phi)};
  return cartesian;
}

Scalar VisionEnv::inner_product(const Vector<3> &a, const Vector<3> &b) const {
  Scalar inner_product = 0;
  for (int i = 0; i < 3; i++) {
    inner_product += a[i] * b[i];
  }
  return inner_product;
}

Vector<3> VisionEnv::cross_product(const Vector<3> &a,
                                   const Vector<3> &b) const {
  Vector<3> cross_product;
  for (int i = 0; i < 3; i++) {
    cross_product[i] =
      a[(i + 1) % 3] * b[(i + 2) % 3] - a[(i + 2) % 3] * b[(i + 1) % 3];
  }
  return cross_product;
}

Vector<visionenv::Vel_Theta_Cuts>
VisionEnv::get_vel_act_boxel(
  const std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> &pos_b_list,
  const std::vector<Scalar> &obs_radius_list, const Vector<3> &poll_y, const Vector<3> &poll_z,
  const Matrix<3, 3> &R_T) const {
  Vector<3> vel_2d = {quad_state_.v[0], quad_state_.v[1], 0};
  Vector<3> body_vel = R_T * vel_2d;
  Scalar vel_theta, vel_phi;
  Scalar vel = quad_state_.v.norm() * vel_compensation_;
  if (vel < vel_acc_cal_threshold_) {
    vel_theta = 0;
    vel_phi = 0;
  }
  else{
    vel_theta = std::atan2(body_vel[1], body_vel[0]);
    vel_phi = std::atan(body_vel[2] / std::sqrt(std::pow(body_vel[0], 2) +
                                                     std::pow(body_vel[1], 2)));
  }

  Vector<visionenv::Vel_Theta_Cuts> act_distance;
  // angle of the velocity direction in 2D map
  for (int t = -visionenv::Vel_Theta_Cuts / 2;
       t < visionenv::Vel_Theta_Cuts / 2; ++t) {
      Scalar theta = (t >= 0) ? acc_theta_list_[t] : -acc_theta_list_[(-t) - 1];  //[deg]
      theta = theta * (PI / 180); //[rad]
      Scalar tcell = theta + vel_theta;
      Scalar pcell = vel_phi;
      Scalar dist =  getClosestDistance(pos_b_list, obs_radius_list, poll_y, poll_z, tcell, pcell) *
        max_detection_range_;
      act_distance[t + (visionenv::Vel_Theta_Cuts) / 2] = calc_dist_to_gain_normalized_act(dist,theta);
  }
  return act_distance;
}

Scalar VisionEnv::calc_dist_to_gain_normalized_act(Scalar dist, Scalar theta) const {
  Scalar squared_vel_normalized_by_gain = quad_state_.v.squaredNorm()*std::pow(vel_compensation_,2);
  return 2*std::sin(theta)*squared_vel_normalized_by_gain/(dist*std::pow(std::cos(theta),2));
}

Vector<visionenv::RewardCuts * visionenv::RewardCuts>
VisionEnv::get_vel_sphericalboxel(
  const std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> &pos_b_list,
  const std::vector<Scalar> &obs_radius_list, const Vector<3> &poll_y, const Vector<3> &poll_z,
  const Matrix<3, 3> &R_T) const {
  Vector<3> vel_2d = {quad_state_.v[0], quad_state_.v[1], 0};
  Vector<3> body_vel = R_T * vel_2d;
  if (body_vel[0] == 0 && body_vel[1] == 0) {
    body_vel[0] = 0.0001;
  }
  Scalar vel_theta = std::atan2(body_vel[1], body_vel[0]);
  Scalar vel_phi = std::atan(body_vel[2] / std::sqrt(std::pow(body_vel[0], 2) +
                                                     std::pow(body_vel[1], 2)));

  Vector<visionenv::RewardCuts * visionenv::RewardCuts> vel_obstacle_obs;
  // angle of the velocity direction in 2D map
  for (int t = -(visionenv::RewardCuts - 1) / 2;
       t < (visionenv::RewardCuts - 1) / 2 + 1; ++t) {
    for (int p = -(visionenv::RewardCuts - 1) / 2;
         p < (visionenv::RewardCuts - 1) / 2 + 1; ++p) {
      Scalar tcell =
        t / ((visionenv::RewardCuts - 1) / 2) * vel_collision_angle_max_ +
        vel_theta;
      Scalar pcell =
        p / ((visionenv::RewardCuts - 1) / 2) * vel_collision_angle_max_ +
        vel_phi;
      vel_obstacle_obs[(t + (visionenv::RewardCuts - 1) / 2) *
                         visionenv::RewardCuts +
                       (p + (visionenv::RewardCuts - 1) / 2)] =
        getClosestDistance(pos_b_list, obs_radius_list, poll_y, poll_z, tcell, pcell) *
        max_detection_range_;
    }
  }
  return vel_obstacle_obs;
}

// void VisionEnv::comp(Scalar &rmin, Scalar r) {
//   if (r < rmin) {
//     rmin = r;
//   }
// }

// Scalar VisionEnv::getclosestpoint(Scalar distance, Scalar theta, Scalar size)
// {
//   return distance * std::cos(theta) -
//          sqrt(std::pow(size, 2) - std::pow(distance * std::sin(theta), 2));
// }

bool VisionEnv::step(Ref<Vector<>> act, Ref<Vector<>> obs,
                     Ref<Vector<>> reward) {
  current_env_steps_++;
  set_current_max_collide_vel();
  if (is_threshold_collision_){
    resetCollision();
  }
  if (!act.allFinite() || act.rows() != act_dim_ || rew_dim_ != reward.rows()) {
    return false;
    logger_.error(
      "Cannot run environment simulation. dimension mismatch or invalid "
      "actions.");
  }

  // time delay
  effect_act_delay(act);
  act_ = act;
  old_pi_act_ = pi_act_;

  // compute actual control actions
  // act has range between [-1, 1] due to Tanh layer of the NN policy
  pi_act_ = act.cwiseProduct(act_std_) + act_mean_;

  cmd_.t += sim_dt_;
  quad_state_.t += sim_dt_;

  // apply old actions to simulate delay
  // cmd_.collective_thrust = old_pi_act_(0);
  // cmd_.omega = old_pi_act_.segment<3>(1);

  Quaternion quaternion = quad_state_.q();
  Vector<3> euler;
  quaternionToEuler(quaternion, euler);


  if (acc_control_){
    cmd_.p[0] = quad_state_.p[0];
    cmd_.p[1] = quad_state_.p[1];
    cmd_.p[2] = 1.0;
    cmd_.v[0] = quad_state_.v[0];
    cmd_.v[1] = quad_state_.v[1];
    cmd_.v[2] = 0.0;
    cmd_.a[0] = pi_act_(0);
    cmd_.a[1] = pi_act_(1);
    cmd_.a[2] = 0.0;
    cmd_.yaw = 0.0;
  }
  else if (control_feedthrough_) {
    cmd_.p[0] = pi_act_(0);
    cmd_.p[1] = pi_act_(1);
    cmd_.p[2] = 1.0;
    cmd_.v[0] = pi_act_(2);
    cmd_.v[1] = pi_act_(3);
    cmd_.v[2] = 0.0;
    cmd_.yaw = 0.0;

  } else if (momentum_bool_) {
    cmd_.p[0] =
      (1 - momentum_) * (pi_act_(0) + quad_state_.p[0]) + momentum_ * cmd_.p[0];
    cmd_.p[1] =
      (1 - momentum_) * (pi_act_(1) + quad_state_.p[1]) + momentum_ * cmd_.p[1];
    cmd_.p[2] = 1.0;
    cmd_.v[0] = (1 - momentum_) * (pi_act_(2) + quad_state_.v[0]) + momentum_ * cmd_.v[0];
    cmd_.v[1] = (1 - momentum_) * (pi_act_(3) + quad_state_.v[1]) + momentum_ * cmd_.v[1];
    cmd_.v[2] = 0.0;
    cmd_.yaw = 0.0;
    // std::cout << "momentum now" << std::endl;
  }

  else {
    cmd_.p[0] = pi_act_(0) + quad_state_.p[0];
    cmd_.p[1] = pi_act_(1) + quad_state_.p[1];
    cmd_.p[2] = 1.0;
    cmd_.v[0] = pi_act_(2) + quad_state_.v[0];
    cmd_.v[1] = pi_act_(3) + quad_state_.v[1];
    cmd_.v[2] = 0.0;
    // std::cout << "cmd_.v" << std::endl;
    // std::cout << cmd_.v << std::endl;
    cmd_.yaw = 0.0;
  }

  // std::cout << euler[2] << std::endl;

  // simulate quadrotor
  quad_ptr_->run(cmd_, sim_dt_);

  // update quadrotor state and old quad_state
  quad_old_state_ = quad_state_;
  quad_ptr_->getState(&quad_state_);
  // std::cout << "x is " << quad_state_.p(quad_state_.IDX::POSX) << std::endl;
  // std::cout << "y is " << quad_state_.p(quad_state_.IDX::POSY) << std::endl;
  // std::cout << "z is " << quad_state_.p(quad_state_.IDX::POSZ) << std::endl;

  // simulate dynamic obstacles
  // simDynamicObstacles(sim_dt_);

  // update observations
  getObs(obs);
  // time delay
  effect_obs_delay(obs);

  bool compute_reward = computeReward(reward);
  return compute_reward;
}

bool VisionEnv::simDynamicObstacles(const Scalar dt) {
  if (dynamic_objects_.size() <= 0) {
    logger_.warn(
      "No Dynamic Obstacles defined. Skipping dynamic obstacles simulation.");
    return false;
  }
  for (int i = 0; i < int(dynamic_objects_.size()); i++) {
    dynamic_objects_[i]->run(sim_dt_);
  }
  return true;
}

bool VisionEnv::resetCollision(){
  // reset quadrotor state when they collide with obstacles, and this is not fatal.
    QuadState collide_state = quad_old_state_;
    quad_ptr_-> resetCollision(collide_state);
    is_threshold_collision_ = false;
    quad_state_ = collide_state;
    quad_ptr_->setState(quad_state_);
    return true;
}

bool VisionEnv::computeReward(Ref<Vector<>> reward) {
  // ---------------------- reward function design
  // - compute collision penalty
  Scalar collision_penalty = 0.0;
  size_t idx = 0;
  for (size_t sort_idx : sort_indexes(relative_2d_pos_norm_)) {
    if (idx >= visionenv::kNObstacles) break;

    // Scalar relative_dist =
    //   (relative_pos_norm_[sort_idx] > 0) &&
    //         (relative_pos_norm_[sort_idx] < max_detection_range_)
    //     ? relative_pos_norm_[sort_idx]
    //     : max_detection_range_; // Papameretes of how far the quadrotor can
    //     detect sphere?

    // std::cout << "dist margin is " << relative_dist -
    // obstacle_radius_[sort_idx]<< std::endl;

    // const Scalar dist_margin_ = 2;
    if (relative_2d_pos_norm_[sort_idx] - obstacle_radius_[sort_idx] -
          quad_size_ <=
        dist_margin_) {
      // compute distance penalty
      Scalar collision_distance = relative_2d_pos_norm_[sort_idx] -
                                  obstacle_radius_[sort_idx] - quad_size_;
      collision_penalty +=
        collision_coeff_ * std::exp(-collision_exp_coeff_ * collision_distance);
    }
    idx += 1;
  }
  {
    Scalar wall_dist;
    wall_dist = (quad_state_.p[1] - wall_pos_) - quad_size_;
    if (wall_dist <= dist_margin_){
      collision_penalty += collision_coeff_ * std::exp(-collision_exp_coeff_ * wall_dist);
    }
    wall_dist = (quad_state_.p[1] + wall_pos_) - quad_size_;
    if (wall_dist <= dist_margin_){
      collision_penalty += collision_coeff_ * std::exp(-collision_exp_coeff_ * wall_dist);
    }
  }

  Scalar when_collision_penlty = 0;
  Scalar vel_collision_penalty = 0;
  Scalar beta = vel_transition_fraction_;
  Scalar a = -1/(beta*beta);
  Scalar b = 2/beta;
  if (is_collision_ && (quad_state_.v).norm() * vel_compensation_ < max_collide_vel_){
    when_collision_penlty = when_collision_coeff_ * std::max((quad_state_.v).squaredNorm()*std::pow(vel_compensation_,2),0.01);
  }
  if (is_wall_collision_ && (quad_state_.v).norm() * vel_compensation_ < max_collide_vel_){
    when_collision_penlty = when_wall_collision_coeff_ * std::max((quad_state_.v).squaredNorm()*std::pow(vel_compensation_,2),0.01);
  }
  else{
    for (size_t i = 0; i < visionenv::RewardCuts * visionenv::RewardCuts; i++) {
      Scalar vel_obs_dist = vel_obs_distance_[i];
      int p = (i % visionenv::RewardCuts) - (visionenv::RewardCuts - 1) / 2;
      int t = (i / visionenv::RewardCuts) - (visionenv::RewardCuts - 1) / 2;
      Scalar angle_discount_factor = std::sqrt(std::pow(p, 2) + std::pow(t, 2)) /
                                    ((visionenv::RewardCuts - 1) / 2) *
                                    vel_collision_angle_max_;
      Scalar dist_factor;
      if (vel_obs_dist<vel_transition_fraction_){
        dist_factor = a*vel_obs_dist+b;
      }
      else{
        dist_factor = 1/vel_obs_dist;
      }
      vel_collision_penalty += vel_collision_coeff_ *
                              (quad_state_.v).squaredNorm() * std::pow(vel_compensation_,2) *
                              dist_factor * std::exp(-angle_discount_factor);
    }
  }

  // std::cout << vel_collision_penalty << std::endl;
  // std::cout << "collision_penalty is " << collision_penalty << std::endl;
  // std::cout << ' ' << std::endl;
  Scalar move_diff = quad_state_.p(QS::POSX) - quad_old_state_.p(QS::POSX);
  Scalar move_reward = 0;
  if (move_diff>0){
    move_reward = move_coeff_ * move_diff;
  }
  else{
    move_reward = move_back_coeff_ * move_diff;
  }
    // - tracking a constant linear velocity
  Scalar lin_vel_reward = 0;
  if ((quad_state_.v).norm() * vel_compensation_ > goal_linear_vel_(0)){
    Vector<3> compensate_vel;
    if (quad_state_.v[0]*vel_compensation_ > goal_linear_vel_(0)){
      compensate_vel = {quad_state_.v[0]*vel_compensation_, quad_state_.v[1], 0.0};
    }
    else{
      compensate_vel = {goal_linear_vel_(0), quad_state_.v[1], 0.0};
    }
    lin_vel_reward = vel_coeff_ * (compensate_vel - goal_linear_vel_).norm();
  }

  // - angular velocity penalty, to avoid oscillations
  const Scalar ang_vel_penalty = angular_vel_coeff_ * quad_state_.w.norm();

  // - world box penalty

  Scalar world_box_penalty =
    world_box_coeff_[0] *
      std::pow(quad_state_.x(QS::POSY) - world_box_center_[0], 4) +
    world_box_coeff_[1] *
      std::pow(quad_state_.x(QS::POSZ) - world_box_center_[1], 4);

  Scalar tilt = quad_state_.Horizontal_Tilt();
  Scalar attitude_penalty = attitude_coeff_ * std::pow(tilt, 2);

  Scalar command_penalty = 0;
  Vector<visionenv::kNAct> pi_act_diff_ = pi_act_ - old_pi_act_;
  for (int i = 0; i < visionenv::kNAct; i++) {
    command_penalty += command_coeff_[i] * std::pow(pi_act_diff_[i], 2);
  }

  Vector<3> vel_3d = (quad_state_.v).normalized();

  Matrix<3, 3> R = quad_state_.R();
  Vector<3> body_x(R(0, 0), R(1, 0), R(2, 0));
  // R's componet (in this case e_x) is always normalized, so we don't have to
  // normalize

  Scalar attitude_vel_penalty =
    attitude_vel_coeff_ * (vel_3d.cross(body_x)).norm();


  //  change progress reward as survive reward
  const Scalar total_reward =
    move_reward + lin_vel_reward + collision_penalty + vel_collision_penalty + when_collision_penlty +
    ang_vel_penalty + survive_rew_ + world_box_penalty + attitude_penalty +
    command_penalty + attitude_vel_penalty;

  // return all reward components for debug purposes
  // only the total reward is used by the RL algorithm
  reward << move_reward, lin_vel_reward, collision_penalty,
    vel_collision_penalty, when_collision_penlty, ang_vel_penalty, survive_rew_, world_box_penalty,
    attitude_penalty, command_penalty, attitude_vel_penalty, total_reward;
  return true;
}


bool VisionEnv::isTerminalState(Scalar &reward) {
  const Scalar safty_threshold = 0.1;
  bool x_valid = quad_state_.p(QS::POSX) >= world_box_[0] + safty_threshold &&
                 quad_state_.p(QS::POSX) <= world_box_[1] - safty_threshold;
  bool y_valid = quad_state_.p(QS::POSY) >= -wall_pos_+ quad_size_ &&
                 quad_state_.p(QS::POSY) <= wall_pos_- quad_size_;
  bool z_valid = quad_state_.x(QS::POSZ) >= world_box_[4] + safty_threshold &&
                 quad_state_.x(QS::POSZ) <= world_box_[5] - safty_threshold;
  Scalar vel = quad_state_.v.norm() * std::pow(vel_compensation_,2);


  if (((is_collision_ || is_wall_collision_ )&& max_collide_vel_< vel )|| cmd_.t >= max_t_ - sim_dt_ || !x_valid || !y_valid ||
      !z_valid || quad_state_.p(QS::POSX) > goal_) {
    if (is_collision_ && max_collide_vel_< vel) {
      reward = -100.0;
      if (fly_result_){
        std::cout << "terminate by collision" << std::endl;
      }

      // return true;
      // std::cout << "t is " << cmd_.t << std::endl;]
      Scalar init_t = 0;
      if (cmd_.t == init_t) {
        // std::cout << "initial state" << std::endl;
        return true;
      }
      collide_num += 1;
    }
    if (is_wall_collision_ && max_collide_vel_< vel) {
      reward = -200.0;
      if (fly_result_){
        std::cout << "terminate by wall collision" << std::endl;
      }

      // return true;
      // std::cout << "t is " << cmd_.t << std::endl;]
      Scalar init_t = 0;
      if (cmd_.t == init_t) {
        // std::cout << "initial state" << std::endl;
        return true;
      }
      wall_collide_num += 1;
    }

    // simulation time out
    if (cmd_.t >= max_t_ - sim_dt_) {
      reward = -20;
      if (fly_result_){
      std::cout << "terminate by time" << std::endl;
      }
      // return true;
      time_num += 1;
    }

    // world boundling box check
    // - x, y, and z
    if (!x_valid || !y_valid || !z_valid) {
      reward = -200;

      if (quad_state_.p(QS::POSX) < world_box_[0] + safty_threshold) {
        reward = -400.0;
      }
      if (fly_result_){
        std::cout << "terminate by box" << std::endl;
      }
      // return true;
      bound_num += 1;
    }

    if (quad_state_.p(QS::POSX) > goal_) {
      reward = 100 * move_coeff_;
      if (fly_result_){
        std::cout << "terminate by reaching the goal" << std::endl;
      }
      // return true;
      goal_num += 1;
    }

    iter += 1;
    // std::cout << iter << std::endl;
    // std::cout << "iter is " << iter << std::endl;
    if (iter == 80 && fly_result_) {
      std::cout << "collide_num is " << collide_num << std::endl;
      std::cout << "time_num is " << time_num << std::endl;
      std::cout << "bound_num is " << bound_num << std::endl;
      std::cout << "goal_num is " << goal_num << std::endl;
    }
    return true;
  }
  return false;
}


bool VisionEnv::getQuadAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && pi_act_.allFinite() && (act.size() == pi_act_.size())) {
    act = pi_act_;
    return true;
  }
  return false;
}

bool VisionEnv::getQuadState(Ref<Vector<>> obs) const {
  // std::cout << "visionenv" << obs.rows() << visionenv::kNQuadState <<
  // std::endl;
  if (quad_state_.t >= 0.0 && (obs.rows() == visionenv::kNQuadState)) {
    obs << quad_state_.t, quad_state_.p, quad_state_.qx,
      quad_state_.Horizontal_Tilt(), quad_state_.v, quad_state_.w,
      quad_state_.a, quad_ptr_->getMotorOmega(), quad_ptr_->getMotorThrusts();
    return true;
  }
  logger_.error("Get Quadrotor state failed.");
  return false;
}

// float VisionEnv::getXstate() const{
//   return quad_state_.p(QS::POSX);
// }

bool VisionEnv::getDepthImage(Ref<DepthImgVector<>> depth_img) {
  if (!rgb_camera_ || !rgb_camera_->getEnabledLayers()[0]) {
    logger_.error(
      "No RGB Camera or depth map is not enabled. Cannot retrieve depth "
      "images.");
    return false;
  }
  rgb_camera_->getDepthMap(depth_img_);

  depth_img = Map<DepthImgVector<>>((float_t *)depth_img_.data,
                                    depth_img_.rows * depth_img_.cols);
  return true;
}


bool VisionEnv::getImage(Ref<ImgVector<>> img, const bool rgb) {
  if (!rgb_camera_) {
    logger_.error("No Camera! Cannot retrieve Images.");
    return false;
  }

  rgb_camera_->getRGBImage(rgb_img_);

  if (rgb_img_.rows != img_height_ || rgb_img_.cols != img_width_) {
    logger_.error(
      "Image resolution mismatch. Aborting.. Image rows %d != %d, Image cols "
      "%d != %d",
      rgb_img_.rows, img_height_, rgb_img_.cols, img_width_);
    return false;
  }

  if (!rgb) {
    // converting rgb image to gray image
    cvtColor(rgb_img_, gray_img_, CV_RGB2GRAY);
    // map cv::Mat data to Eiegn::Vector
    img = Map<ImgVector<>>(gray_img_.data, gray_img_.rows * gray_img_.cols);
  } else {
    img = Map<ImgVector<>>(rgb_img_.data, rgb_img_.rows * rgb_img_.cols *
                                            rgb_camera_->getChannels());
  }
  return true;
}


bool VisionEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["environment"]) {
    difficulty_level_list_ =
      cfg["environment"]["level"].as<std::vector<std::string>>();
    world_box_ = cfg["environment"]["world_box"].as<std::vector<Scalar>>();
    world_box_center_.push_back((world_box_[2] + world_box_[3]) / 2);
    world_box_center_.push_back((world_box_[4] + world_box_[5]) / 2);
    y_lim_ = cfg["environment"]["y_lim"].as<std::vector<Scalar>>();
    z_lim_ = cfg["environment"]["z_lim"].as<std::vector<Scalar>>();
    wall_pos_ = cfg["environment"]["wall_pos"].as<Scalar>();
    tree_size_ = cfg["environment"]["tree_size"].as<Scalar>();
    // tree_size_range_ =
    //   cfg["environment"]["tree_size_range"].as<std::vector<Scalar>>();
    std::vector<Scalar> goal_vel_vec =
      cfg["environment"]["goal_vel"].as<std::vector<Scalar>>();
    goal_linear_vel_ = Vector<3>(goal_vel_vec.data());
    max_detection_range_ =
      cfg["environment"]["max_detection_range"].as<Scalar>();
    goal_ = cfg["environment"]["goal"].as<Scalar>();
    fly_result_ = cfg["environment"]["fly_result"].as<bool>();
    // std::cout << "fly_result_ is " << std::boolalpha << fly_result_
    //           << std::endl;
    control_feedthrough_ = cfg["environment"]["control_feedthrough"].as<bool>();
    momentum_bool_ = cfg["environment"]["momentum_bool"].as<bool>();
    momentum_ = cfg["environment"]["momentum"].as<Scalar>();
    acc_control_ = cfg["environment"]["acc_control"].as<bool>();
    dist_theta_list_ = cfg["environment"]["dist_theta"].as<std::vector<Scalar>>();
    acc_theta_list_ = cfg["environment"]["acc_theta"].as<std::vector<Scalar>>();
    init_max_collide_vel_ = cfg["environment"]["max_collide_vel"].as<Scalar>();
    linear_transition_log_ = cfg["environment"]["linear_transition_log"].as<Scalar>();
    vel_transition_fraction_ = cfg["environment"]["vel_transition_fraction"].as<Scalar>();
    vel_acc_cal_threshold_ = cfg["environment"]["vel_acc_cal_threshold"].as<Scalar>();

    att_noise_ = cfg["environment"]["att_noise"].as<Scalar>() * M_PI / 180; //deg2rad
    omega_noise_ = cfg["environment"]["omega_noise"].as<Scalar>();
    // phi_list_ = cfg["environment"]["phi_list"].as<std::vector<Scalar>>();
  }

  if (cfg["simulation"]) {
    sim_dt_ = cfg["simulation"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["simulation"]["max_t"].as<Scalar>();
    num_envs_ = cfg["simulation"]["num_envs"].as<int>();
    num_each_env_steps_ = (int)(1E8/num_envs_);
    act_delay_ = cfg["simulation"]["act_delay"].as<Scalar>();
    act_delay_width_ = cfg["simulation"]["act_delay_width"].as<Scalar>();
    obs_delay_ = cfg["simulation"]["obs_delay"].as<Scalar>();
    obs_delay_width_ = cfg["simulation"]["obs_delay_width"].as<Scalar>();

    act_buffer_size_ = ceil((act_delay_ + act_delay_width_) / sim_dt_);
    obs_buffer_size_ = ceil((obs_delay_ + obs_delay_width_) / sim_dt_);

  } else {
    logger_.error("Cannot load [quadrotor_env] parameters");
    return false;
  }

  if (cfg["quadrotor_dynamics"]){
    range_max_gain_ = Map<Vector<2>>(cfg["quadrotor_dynamics"]["range_max_gain"].as<std::vector<Scalar>>().data());
    learn_max_gain_ = range_max_gain_[1];
    max_gain_fix_ =
      cfg["quadrotor_dynamics"]["max_gain_fix"].as<bool>();
    max_gain_ = cfg["quadrotor_dynamics"]["fix_max_gain"].as<Scalar>();
    vel_compensation_ = std::sqrt(learn_max_gain_/max_gain_);
    act_std_ << max_gain_, max_gain_;
  }

  if (cfg["rewards"]) {
    // load reward coefficients for reinforcement learning
    move_coeff_ = cfg["rewards"]["move_coeff"].as<Scalar>();
    move_back_coeff_ = cfg["rewards"]["move_back_coeff"].as<Scalar>();
    vel_coeff_ = cfg["rewards"]["vel_coeff"].as<Scalar>();
    collision_coeff_ = cfg["rewards"]["collision_coeff"].as<Scalar>();
    vel_collision_coeff_ = cfg["rewards"]["vel_collision_coeff"].as<Scalar>();
    vel_collision_angle_max_ =
      cfg["rewards"]["vel_collision_angle_max"].as<Scalar>() * M_PI /
      180;  // vel_collision_angle_max_ [rad]
    collision_exp_coeff_ = cfg["rewards"]["collision_exp_coeff"].as<Scalar>();
    dist_margin_ = cfg["rewards"]["dist_margin"].as<Scalar>();
    angular_vel_coeff_ = cfg["rewards"]["angular_vel_coeff"].as<Scalar>();
    survive_rew_ = cfg["rewards"]["survive_rew"].as<Scalar>();
    world_box_coeff_ =
      cfg["rewards"]["world_box_coeff"].as<std::vector<Scalar>>();
    if ((int)world_box_coeff_.size() != 2) {
      logger_.error("world_box_coeff_ size is not equal to 2");
      return false;
    }
    attitude_coeff_ = cfg["rewards"]["attitude_coeff"].as<Scalar>();
    command_coeff_ = cfg["rewards"]["command_coeff"].as<std::vector<Scalar>>();
    if ((int)command_coeff_.size() != act_dim_) {
      logger_.error("command_coeff_ size is not equal to act_dim_");
      return false;
    }
    attitude_vel_coeff_ = cfg["rewards"]["attitude_vel_coeff"].as<Scalar>();
    when_collision_coeff_ =
      cfg["rewards"]["when_collision_coeff"].as<Scalar>();
    when_wall_collision_coeff_ =
      cfg["rewards"]["when_wall_collision_coeff"].as<Scalar>();

    // std::cout << dist_margin_ << std::endl;

    // load reward settings
    reward_names_ = cfg["rewards"]["names"].as<std::vector<std::string>>();
    rew_dim_ = reward_names_.size();
  } else {
    logger_.error("Cannot load [rewards] parameters");
    return false;
  }

  // environment
  if (cfg["unity"]) {
    unity_render_ = cfg["unity"]["render"].as<bool>();
    scene_id_ = cfg["unity"]["scene_id"].as<SceneID>();
  }

  if (cfg["quadrotor_dynamics"]) {
    quad_size_ = cfg["quadrotor_dynamics"]["quad_size"].as<Scalar>();
    quad_size_fix_ = cfg["quadrotor_dynamics"]["quad_size_fix"].as<bool>();
    quad_size_threshold_dev_ =
      cfg["quadrotor_dynamics"]["quad_size_threshold_dev"].as<Scalar>();
    quad_size_threshold_ = quad_size_ - quad_size_threshold_dev_ * uniform_dist_one_direction_(random_gen_);
    // init_size_ = cfg["quadrotor_dynamics"]["quad_size"].as<Scalar>();
    // init_mass_ = cfg["quadrotor_dynamics"]["mass"].as<Scalar>();
    // init_inertia_ = cfg["quadrotor_dynamics"]["inertia"].as<std::vector<Scalar>>();
    // std::vector tbm_fl = cfg["quadrotor_dynamics"]["tbm_fl"].as<std::vector<Scalar>>();
    // init_tbm_x_length_ = tbm_fl[0];
    // init_tbm_y_length_ = tbm_fl[1];
    // init_motor_omega_min_ = cfg["quadrotor_dynamics"]["motor_omega_min"].as<Scalar>();
    // init_motor_omega_max_ = cfg["quadrotor_dynamics"]["motor_omega_max"].as<Scalar>();
    // init_thrust_map_ = cfg["quadrotor_dynamics"]["thrust_map"].as<std::vector<Scalar>>();
  }
  //
  std::string scene_file =
    getenv("FLIGHTMARE_PATH") + std::string("/flightpy/configs/scene.yaml");
  // check if configuration file exist
  if (!(file_exists(scene_file))) {
    logger_.error("Unity scene configuration file %s does not exists.",
                  scene_file);
  }
  // load configuration file
  YAML::Node scene_cfg_node = YAML::LoadFile(scene_file);
  std::string scene_idx = "scene_" + std::to_string(scene_id_);

  std::vector<Scalar> render_offset =
    scene_cfg_node[scene_idx]["render_offset"].as<std::vector<Scalar>>();
  unity_render_offset_ = Vector<3>(render_offset.data());
  return true;
}
bool VisionEnv::changeLevel() {
  difficulty_level_ = difficulty_level_list_[0];
  // std::size_t size_ = difficulty_level_list_.size();
  // std::cout << size_ <<std::endl;
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // int env_id = gen() % 100;
  obstacle_cfg_path_ =
    getenv("FLIGHTMARE_PATH") + std::string("/flightpy/configs/vision/") +
    difficulty_level_ + std::string("/") + std::string("environment_") +
    std::to_string(env_id_ % 501);
  std::cout << obstacle_cfg_path_ << std::endl;
  // cfg_["environment"]["env_folder"] = (env_id_ + 1) % 500;

  // add dynamic objects
  // std::string dynamic_object_yaml =
  //   obstacle_cfg_path_ + std::string("/dynamic_obstacles.yaml");
  // if (!configDynamicObjects(dynamic_object_yaml)) {
  //   // std::cout << dynamic_object_yaml << std::endl;
  //   logger_.error(
  //     "Cannot config Dynamic Object Yaml. Something wrong with the config "
  //     "file");
  // }

  // add static objects
  static_object_csv_ =
    obstacle_cfg_path_ + std::string("/static_obstacles.csv");
  if (!configStaticObjects(static_object_csv_)) {
    logger_.error(
      "Cannot config Static Object. Something wrong with the config file");
  }
  return true;
}

bool VisionEnv::chooseLevel() {
  // std::size_t size_ = difficulty_level_list_.size();
  // // std::cout << size_ <<std::endl;
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // int rand_int_ = gen() % size_;
  difficulty_level_ = difficulty_level_list_[0];
  return true;
}

bool VisionEnv::configDynamicObjects(const std::string &yaml_file) {
  //
  if (!(file_exists(yaml_file))) {
    logger_.error("Configuration file %s does not exists.", yaml_file);
    return false;
  }
  YAML::Node cfg_node = YAML::LoadFile(yaml_file);

  // logger_.info("Configuring dynamic objects");

  int num_objects = cfg_node["N"].as<int>();
  // create static objects
  for (int i = 0; i < num_objects; i++) {
    std::string object_id = "Object" + std::to_string(i + 1);
    std::string prefab_id = cfg_node[object_id]["prefab"].as<std::string>();
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    // load location, rotation and size
    std::vector<Scalar> posvec =
      (cfg_node[object_id]["position"]).as<std::vector<Scalar>>();
    std::vector<Scalar> rotvec =
      (cfg_node[object_id]["rotation"]).as<std::vector<Scalar>>();
    std::vector<Scalar> scalevec =
      (cfg_node[object_id]["scale"]).as<std::vector<Scalar>>();

    obj->setPosition(Vector<3>(posvec.data()));
    obj->setRotation(Quaternion(rotvec.data()));
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(Vector<3>(scalevec.data()) * tree_size_);

    std::string csv_name = cfg_node[object_id]["csvtraj"].as<std::string>();
    std::string csv_file = obstacle_cfg_path_ + std::string("/csvtrajs/") +
                           csv_name + std::string(".csv");
    if (!(file_exists(csv_file))) {
      logger_.error("Configuration file %s does not exists.", csv_file);
      return false;
    }
    obj->loadTrajectory(csv_file);

    dynamic_objects_.push_back(obj);
  }
  num_dynamic_objects_ = dynamic_objects_.size();
  return true;
}

bool VisionEnv::configStaticObjects(const std::string &csv_file) {
  //
  if (!(file_exists(csv_file))) {
    logger_.error("Configuration file %s does not exists.", csv_file);
    return false;
  }
  std::ifstream infile(csv_file);
  int i = 0;
  for (auto &row : CSVRange(infile)) {
    // std::cout << row[0] << std::endl; // object_id, and 10 obs's data
    // std::cout << "RowType: " << typeid(row).name() << std::endl;

    // Read column 0 for time
    std::string object_id = "StaticObject" + std::to_string(i + 1);
    // std::cout << object_id << std::endl;
    std::string prefab_id = (std::string)row[0];

    //
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    //
    Vector<3> pos;
    pos << std::stod((std::string)row[1]), std::stod((std::string)row[2]),
      std::stod((std::string)row[3]);

    Quaternion quat;
    quat.w() = std::stod((std::string)row[4]);
    quat.x() = std::stod((std::string)row[5]);
    quat.y() = std::stod((std::string)row[6]);
    quat.z() = std::stod((std::string)row[7]);

    Vector<3> scale;
    scale << std::stod((std::string)row[8]), std::stod((std::string)row[9]),
      std::stod((std::string)row[10]);

    //
    obj->setPosition(pos);
    obj->setRotation(quat);
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(scale);
    static_objects_.push_back(obj);
  }
  num_static_objects_ = static_objects_.size();

  return true;
}

bool VisionEnv::configCamera(const YAML::Node &cfg) {
  if (!cfg["rgb_camera"]) {
    logger_.error("Cannot config RGB Camera");
    return false;
  }

  if (!cfg["rgb_camera"]["on"].as<bool>()) {
    logger_.warn("Camera is off. Please turn it on.");
    return false;
  }

  if (quad_ptr_->getNumCamera() >= 1) {
    logger_.warn("Camera has been added. Skipping the camera configuration.");
    return false;
  }

  // create camera
  rgb_camera_ = std::make_shared<RGBCamera>();

  // load camera settings
  std::vector<Scalar> t_BC_vec =
    cfg["rgb_camera"]["t_BC"].as<std::vector<Scalar>>();
  std::vector<Scalar> r_BC_vec =
    cfg["rgb_camera"]["r_BC"].as<std::vector<Scalar>>();

  //
  Vector<3> t_BC(t_BC_vec.data());
  Matrix<3, 3> r_BC =
    (AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ()) *
     AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
     AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()))
      .toRotationMatrix();
  std::vector<bool> post_processing = {false, false, false};
  post_processing[0] = cfg["rgb_camera"]["enable_depth"].as<bool>();
  post_processing[1] = cfg["rgb_camera"]["enable_segmentation"].as<bool>();
  post_processing[2] = cfg["rgb_camera"]["enable_opticalflow"].as<bool>();

  //
  rgb_camera_->setFOV(cfg["rgb_camera"]["fov"].as<Scalar>());
  rgb_camera_->setWidth(cfg["rgb_camera"]["width"].as<int>());
  rgb_camera_->setChannels(cfg["rgb_camera"]["channels"].as<int>());
  rgb_camera_->setHeight(cfg["rgb_camera"]["height"].as<int>());
  rgb_camera_->setRelPose(t_BC, r_BC);
  rgb_camera_->setPostProcessing(post_processing);


  // add camera to the quadrotor
  quad_ptr_->addRGBCamera(rgb_camera_);

  // adapt parameters
  img_width_ = rgb_camera_->getWidth();
  img_height_ = rgb_camera_->getHeight();
  rgb_img_ = cv::Mat::zeros(img_height_, img_width_,
                            CV_MAKETYPE(CV_8U, rgb_camera_->getChannels()));
  depth_img_ = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
  return true;
}

bool VisionEnv::addQuadrotorToUnity(const std::shared_ptr<UnityBridge> bridge) {
  if (!quad_ptr_) return false;
  bridge->addQuadrotor(quad_ptr_);

  // for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
  //   bridge->addDynamicObject(dynamic_objects_[i]);
  // }

  //
  bridge->setRenderOffset(unity_render_offset_);
  bridge->setObjectCSV(static_object_csv_);
  return true;
}

bool VisionEnv::setUnity(bool render) {
  unity_render_ = render;
  if (!unity_render_ || unity_bridge_ptr_ != nullptr) {
    logger_.warn(
      "Unity render is False or Flightmare Bridge has been already created. "
      "Cannot set Unity.");
    return false;
  }
  // create unity bridge
  unity_bridge_ptr_ = UnityBridge::getInstance();
  // add objects to Unity

  addQuadrotorToUnity(unity_bridge_ptr_);

  logger_.info("Flightmare Bridge created.");
  return true;
}


bool VisionEnv::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  // std::cout << "connectUnity" << std::endl;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}


FrameID VisionEnv::updateUnity(const FrameID frame_id) {
  if (unity_render_ && unity_ready_) {
    // std::cout << "updageUnity" << std::endl;
    unity_bridge_ptr_->getRender(frame_id);
    return unity_bridge_ptr_->handleOutput(frame_id);
  } else {
    return 0;
  }
}


void VisionEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

int VisionEnv::getNumDetectedObstacles(void) { return num_detected_obstacles_; }

void VisionEnv::effect_act_delay(Ref<Vector<>> act){
    act_buffer_.push_front(act); //std::deque<Vector<>> act_buffer_;

    Scalar act_time_delay = uniform_dist_one_direction_(random_gen_)*act_delay_width_ + act_delay_;
    if (act_past_delay_ - sim_dt_/4 < act_time_delay - sim_dt_){
      act_time_delay = act_past_delay_ - sim_dt_/4 + sim_dt_;
    }
    Scalar act_time_step = act_time_delay/sim_dt_;
    size_t act_time_idx = static_cast<size_t>(std::floor(act_time_step));
    Scalar act_time_frac = act_time_step - act_time_idx;

    act = act_buffer_[act_time_idx] * (1-act_time_frac) + act_buffer_[act_time_idx + 1] * act_time_frac;

    if (act_buffer_.size() > act_buffer_size_){
      act_buffer_.pop_back();
    }
    act_past_delay_ = act_time_delay;
}

void VisionEnv::effect_obs_delay(Ref<Vector<>> obs){
    obs_buffer_.push_front(obs); //std::deque<Vector<>> obs_buffer_;

    Scalar obs_time_delay = uniform_dist_one_direction_(random_gen_)*obs_delay_width_ + obs_delay_;
    if (obs_past_delay_ - sim_dt_/4 < obs_time_delay - sim_dt_){
      obs_time_delay = obs_past_delay_ - sim_dt_/4 + sim_dt_;
    }
    Scalar obs_time_step = obs_time_delay/sim_dt_;
    size_t obs_time_idx = static_cast<size_t>(std::floor(obs_time_step));
    Scalar obs_time_frac = obs_time_step - obs_time_idx;

    obs = obs_buffer_[obs_time_idx] * (1-obs_time_frac) + obs_buffer_[obs_time_idx + 1] * obs_time_frac;

    if (obs_buffer_.size() > obs_buffer_size_){
      obs_buffer_.pop_back();
    }
    obs_past_delay_ = obs_time_delay;
}

void VisionEnv::reset_delay_buffer(){
  act_buffer_.clear();
  obs_buffer_.clear();

  act_buffer_.insert(act_buffer_.begin(), act_buffer_size_, Vector<>::Zero(act_dim_));
  obs_buffer_.insert(obs_buffer_.begin(), obs_buffer_size_, Vector<>::Zero(obs_dim_));
  act_past_delay_ = 0;
  obs_past_delay_ = 0;
}

bool VisionEnv::set_current_max_collide_vel(){
  max_collide_vel_ = init_max_collide_vel_ * (1 - current_env_steps_/num_each_env_steps_);
  // for debug
  if (max_collide_vel_ < 0) {
    max_collide_vel_ = 0;
  }
  return true;
  }

std::ostream &operator<<(std::ostream &os, const VisionEnv &vision_env) {
  os.precision(3);
  os << "Vision Environment:\n"
     << "obs dim =            [" << vision_env.obs_dim_ << "]\n"
     << "act dim =            [" << vision_env.act_dim_
     << "]\n"
     //  << "#dynamic objects=    [" << vision_env.num_dynamic_objects_ << "]\n"
     << "#static objects=     [" << vision_env.num_static_objects_ << "]\n"
     << "obstacle dim =       [" << vision_env.num_detected_obstacles_ << "]\n"
     << "sim dt =             [" << vision_env.sim_dt_ << "]\n"
     << "max_t =              [" << vision_env.max_t_ << "]\n"
     << "act_mean =           [" << vision_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << vision_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << vision_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << vision_env.obs_std_.transpose() << "]"
     << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
