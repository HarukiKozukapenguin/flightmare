
#pragma once

// std lib
#include <stdlib.h>

#include <cmath>
#include <iostream>

// yaml cpp
#include <yaml-cpp/yaml.h>

// random choice of level and env
#include <random>

#include <typeinfo>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/common/utils.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/unity_object.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

#define PI 3.14159265359
namespace flightlib {

namespace visionenv {

enum Vision : int {
  //
  kNQuadState = 25,

  kNObstacles = 30,
  kNObstaclesState = 4,

  Cuts = 8,

  // observations
  kObs = 0,
  kNObs = 3+9+3+3+4+3 + Cuts*Cuts,

  // control actions
  kAct = 0,
  kNAct = 4,
};
}  // namespace visionenv

class VisionEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VisionEnv();
  VisionEnv(const std::string &cfg_path, const int env_id);
  VisionEnv(const YAML::Node &cfg_node, const int env_id);
  ~VisionEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs) override;
  bool reset(Ref<Vector<>> obs, bool random);
  bool step(const Ref<Vector<>> act, Ref<Vector<>> obs,
            Ref<Vector<>> reward) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getImage(Ref<ImgVector<>> img, const bool rgb = true) override;
  bool getDepthImage(Ref<DepthImgVector<>> img) override;

  bool getObstacleState(Ref<Vector<visionenv::Cuts*visionenv::Cuts>> sphericalboxel);
  Vector<visionenv::Cuts*visionenv::Cuts> getsphericalboxel(std::vector<Vector<3>,
  Eigen::aligned_allocator<Vector<3>>>& pos_b_list, std::vector<Scalar> pos_norm_list, std::vector<Scalar> obs_radius_list);
  Scalar getClosestDistance(std::vector<Vector<3>,
  Eigen::aligned_allocator<Vector<3>>>& pos_b_list, std::vector<Scalar> pos_norm_list, std::vector<Scalar> obs_radius_list, 
  Scalar tcell, Scalar fcell);
  Vector<3> getCartesianFromAng(Scalar t, Scalar f);
  Scalar inner_product(Vector<3> a, Vector<3> b);
  void comp(Scalar& rmin, Scalar r);
  Scalar getclosestpoint(Scalar distance, Scalar theta, Scalar size);

  // get quadrotor states
  bool getQuadAct(Ref<Vector<>> act) const;
  bool getQuadState(Ref<Vector<>> state) const;

  // float getXstate() const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  bool addQuadrotorToUnity(const std::shared_ptr<UnityBridge> bridge) override;

  friend std::ostream &operator<<(std::ostream &os,
                                  const VisionEnv &vision_env);


  bool configCamera(const YAML::Node &cfg_node);
  bool changeLevel();
  bool chooseLevel();
  bool configDynamicObjects(const std::string &yaml_file);
  bool configStaticObjects(const std::string &csv_file);

  bool simDynamicObstacles(const Scalar dt);

  // flightmare (visualization)
  bool setUnity(const bool render);
  bool connectUnity();
  void disconnectUnity();
  FrameID updateUnity(const FrameID frame_id);


  //
  int getNumDetectedObstacles(void);
  bool isCollision(void) { return is_collision_; };
  inline std::vector<std::string> getRewardNames() { return reward_names_; }
  inline void setSceneID(const SceneID id) { scene_id_ = id; }
  inline std::shared_ptr<Quadrotor> getQuadrotor() { return quad_ptr_; }
  inline std::vector<std::shared_ptr<UnityObject>> getDynamicObjects() {
    return dynamic_objects_;
  }
  inline std::vector<std::shared_ptr<UnityObject>> getStaticObjects() {
    return static_objects_;
  }

  std::unordered_map<std::string, float> extra_info_;

 private:
  bool computeReward(Ref<Vector<>> reward);
  void init();
  int env_id_;
  // quadrotor
  std::shared_ptr<Quadrotor> quad_ptr_;
  //
  std::vector<std::shared_ptr<UnityObject>> static_objects_;
  std::vector<std::shared_ptr<UnityObject>> dynamic_objects_;

  QuadState quad_state_, quad_old_state_;
  Command cmd_;
  Logger logger_{"VisionEnv"};

  // Define reward for training
  Scalar move_coeff_, vel_coeff_, collision_coeff_,collision_exp_coeff_, angular_vel_coeff_, survive_rew_, dist_margin;
  std::vector<Scalar> world_box_coeff_;
  Vector<3> goal_linear_vel_;
  bool is_collision_;

  size_t obstacle_num;

  // max detection range (meter)
  Scalar max_detection_range_;
  Scalar goal_;
  std::vector<Scalar> relative_pos_norm_;
  std::vector<Scalar> obstacle_radius_;


  int num_detected_obstacles_;
  std::vector<std::string> difficulty_level_list_;
  std::string difficulty_level_;
  int env_folder_;
  std::vector<Scalar> world_box_;
  std::vector<Scalar> world_box_center_;

  // observations and actions (for RL)
  Vector<visionenv::kNObs> pi_obs_;
  Vector<visionenv::kNAct> pi_act_;
  Vector<visionenv::kNAct> old_pi_act_;

  // action and observation normalization (for learning)
  Vector<visionenv::kNAct> act_mean_;
  Vector<visionenv::kNAct> act_std_;
  Vector<visionenv::kNObs> obs_mean_ = Vector<visionenv::kNObs>::Zero();
  Vector<visionenv::kNObs> obs_std_ = Vector<visionenv::kNObs>::Ones();

  // robot vision
  std::shared_ptr<RGBCamera> rgb_camera_;
  cv::Mat rgb_img_, gray_img_;
  cv::Mat depth_img_;

  // auxiliary variables
  bool use_camera_{false};
  YAML::Node cfg_;
  std::vector<std::string> reward_names_;

  // Unity Rendering
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  SceneID scene_id_{UnityScene::WAREHOUSE};
  bool unity_ready_{false};
  bool unity_render_{false};
  RenderMessage_t unity_output_;
  uint16_t receive_id_{0};
  Vector<3> unity_render_offset_;

  //
  std::string static_object_csv_;
  std::string obstacle_cfg_path_;
  int num_dynamic_objects_;
  int num_static_objects_;
};

}  // namespace flightlib
