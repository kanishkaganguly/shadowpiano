#include <shadowpiano/shadow_piano.hpp>

/**
 * @brief Holds finger IDs for arranging into notes to be played
 * FIRST, MIDDLE, RIGHT corresponds to fingers to be played
 * RIGHT, LEFT corresponds to moving the arm left/right by 3-key offsets
 */

enum fingerID { FIRST, MIDDLE, RING, RIGHT, LEFT };

// Variables for thread synchronization
std::atomic_int note_counter{0};
// Mutex for critical sections
std::mutex m;
// Conditional variables for thread signalling
std::condition_variable cv;
// Synchronize thread start
bool ready = false;

void reset_sequence(
    ros::NodeHandle &n,
    std::vector<std::reference_wrapper<shadow_finger::Finger>> finger_list,
    shadow_planning::PlanningOptions &planning_options) {
  // Turn off trajectory controller for hand
  shadow_utils::stopTrajectoryController(n, "rh_trajectory_controller");

  // Release thumb from grasp
  std::vector<double> thumb_reset_joints{shadow_utils::deg2rad(10.0),
                                         shadow_utils::deg2rad(0.0), 0.0, 0.0,
                                         shadow_utils::deg2rad(0.0)};
  // Release fingers from grasp
  std::vector<double> finger_reset_joints{0.0, 0.0, 0.0, 0.0};
  // Loop over all fingers and release them from grasp
  for (auto finger : finger_list) {
    std::string finger_name = finger.get().getFingerName();
    if (finger_name.compare("thumb") == 0) {
      shadow_planning::moveFingerJoints(finger.get(), thumb_reset_joints);
    } else {
      shadow_planning::moveFingerJoints(finger.get(), finger_reset_joints);
    }
  }
}

void press_key_thread(int note_idx, shadow_finger::Finger &finger,
                      ros::NodeHandle &n, const float &move_angle,
                      const float &touch_delta = 100) {
  std::string finger_name = finger.getFingerName();
  ROS_INFO("%s finger thread started...", finger_name.c_str());

  ROS_INFO("%s finger waiting for trigger...", finger_name.c_str());
  {
    std::unique_lock<std::mutex> lck(m);
    while (!(note_idx == note_counter.load()) || !ready) {
      cv.wait(lck);
    }
  }
  ROS_INFO("%s finger received trigger...", finger_name.c_str());

  std::vector<double> curr_joint_vals = shadow_finger::getJointValues(finger);
  int proximal_idx = 1;

  sr_robot_msgs::BiotacAll::ConstPtr biotacDataPtr;
  biotacDataPtr = ros::topic::waitForMessage<sr_robot_msgs::BiotacAll>(
      "/rh/tactile", n, ros::Duration(0.4));
  float curr_pressure =
      biotacDataPtr->tactiles[shadow_finger::getBiotacIdx(finger_name)].pdc;
  float touch_threshold = curr_pressure + touch_delta;

  std_msgs::Float64 move_command;
  ros::Rate loop_rate(1000);
  ros::Duration long_press(0.5), short_press(0.1);
  // Check pressure every pressure_check iterations
  int pub_iter = 0;
  int pressure_check = 10;

  while (curr_pressure <= touch_threshold) {
    move_command.data = curr_joint_vals[proximal_idx] + move_angle;
    finger._joint_controller_publishers[proximal_idx].publish(move_command);

    if ((pub_iter % pressure_check) == 0) {
      biotacDataPtr = ros::topic::waitForMessage<sr_robot_msgs::BiotacAll>(
          "/rh/tactile", n, ros::Duration(0.4));
      curr_pressure =
          biotacDataPtr->tactiles[shadow_finger::getBiotacIdx(finger_name)].pdc;

      // Log outputs
      std::stringstream t;
      t << "Playing with " << finger_name << " finger\n"
        << "CurrPressure: " << curr_pressure << "\n"
        << "PressureThreshold: " << touch_threshold << "\n"
        << "Command: " << move_command.data << "\n";
      shadow_utils::ROS_INFO_COLOR(t, shadow_utils::COLORS::COLOR_RED);
    }

    // Update for next iteration
    curr_joint_vals = shadow_finger::getJointValues(finger);
    // Sleep
    loop_rate.sleep();
    // Pressure Check Update
    pub_iter += 1;
  }
  ROS_INFO("%s finger finished playing ...", finger_name.c_str());

  {
    ROS_INFO("Triggering next note %d ...", note_counter.load());
    std::unique_lock<std::mutex> lck(m);
    note_counter++;
    cv.notify_all();
  }

  short_press.sleep();
  move_command.data = 0.0;
  finger._joint_controller_publishers[proximal_idx].publish(move_command);

  return;
}

void shift_arm(int note_idx, int dir) {
  ROS_INFO("Arm thread started...");

  ROS_INFO("Arm waiting for trigger...");
  {
    std::unique_lock<std::mutex> lck(m);
    while (!(note_idx == note_counter.load()) || !ready) {
      cv.wait(lck);
    }
  }

  moveit::planning_interface::MoveGroupInterface arm_move_group("right_arm");
  geometry_msgs::PoseStamped curr_pose = arm_move_group.getCurrentPose();
  geometry_msgs::Pose target_pose = curr_pose.pose;

  std::vector<geometry_msgs::Pose> waypoints;
  switch (dir) {
    case 0:
      // LEFT
      target_pose.position.x -= 0.06;
      break;
    case 1:
      // RIGHT
      target_pose.position.x += 0.06;
      break;
    default:
      break;
  }
  waypoints.push_back(target_pose);

  moveit_msgs::RobotTrajectory trajectory;
  const double jump_threshold = 3.0;
  const double eef_step = 0.01;
  double fraction = arm_move_group.computeCartesianPath(
      waypoints, eef_step, jump_threshold, trajectory);
  ROS_INFO("Planned %.2f%% of path...", (fraction * 100));

  if (fraction >= 0.9) arm_move_group.execute(trajectory);
  ROS_INFO("Arm done shifting...");

  {
    ROS_INFO("Triggering next note %d ...", note_counter.load());
    std::unique_lock<std::mutex> lck(m);
    note_counter++;
    cv.notify_all();
  }
}

/**
 * @brief Synchronized start of all threads
 */
void run() {
  std::unique_lock<std::mutex> lck(m);
  ready = true;
  cv.notify_all();
}

int main(int argc, char **argv) {
  // Argument Parser
  CLI::App argparse{"ShadowHand Piano Demo"};
  bool reset_arg{false};
  argparse.add_flag("--reset", reset_arg, "Reset the fingers");
  CLI11_PARSE(argparse, argc, argv);

  // Setup ROS node
  ros::init(argc, argv, "shadow_piano");
  ros::AsyncSpinner spinner(4);
  spinner.start();
  ros::NodeHandle n;

  // Setup robot model
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();

  // Create PlanningOptions
  shadow_planning::PlanningOptions planning_options =
      shadow_planning::PlanningOptions();
  planning_options.num_attempts = 10;
  planning_options.allow_replanning = true;
  planning_options.set_planning_time = 30.0;
  planning_options.goal_position_tolerance = 0.01;
  planning_options.goal_orientation_tolerance = 0.01;
  planning_options.goal_joint_tolerance = 0.01;
  planning_options.velocity_scaling_factor = 0.1;
  planning_options.acceleration_scaling_factor = 0.1;

  // MoveGroup name for Shadow Hand
  moveit::planning_interface::MoveGroupInterface hand_move_group("right_hand");
  moveit::planning_interface::MoveGroupInterface::Plan hand_reset_plan;
  std::string hand_reset = "open";
  shadow_utils::startTrajectoryController(n, "rh_trajectory_controller");
  shadow_planning::planToNamedTarget(planning_options, hand_move_group,
                                     hand_reset, hand_reset_plan);
  hand_move_group.execute(hand_reset_plan);

  // Stop here if reset argument is passed
  if (reset_arg) {
    // Start controller before resetting, so that everything goes back to
    // normal
    shadow_utils::startTrajectoryController(n, "rh_trajectory_controller");
    {
      std::stringstream t;
      t << "Hand has been reset...";
      shadow_utils::ROS_INFO_COLOR(t, shadow_utils::COLORS::COLOR_RED);
    }
    return 0;
  }

  // Create first finger object
  std::string ff_name = "first";
  shadow_finger::Finger first_finger(ff_name, n);
  // Create middle finger object
  std::string mf_name = "middle";
  shadow_finger::Finger middle_finger(mf_name, n);
  // Create ring finger object
  std::string rf_name = "ring";
  shadow_finger::Finger ring_finger(rf_name, n);

  // Stop trajectory controller
  shadow_utils::stopTrajectoryController(n, "rh_trajectory_controller");

  // Threads for each finger
  std::vector<std::thread> finger_threads;
  // Move angle for finger
  float move_angle = shadow_utils::deg2rad(6.0);
  // Threshold for keypress detection
  float touch_delta = 110.0;
  // Note order
  // std::vector<int> notes = {fingerID::LEFT,   fingerID::FIRST,
  // fingerID::MIDDLE,
  //                           fingerID::RING,   fingerID::RIGHT,
  //                           fingerID::FIRST, fingerID::MIDDLE,
  //                           fingerID::RING};
  std::vector<int> notes = {fingerID::LEFT};

  for (size_t i = 0; i < notes.size(); i++) {
    switch (notes[i]) {
      case fingerID::FIRST:
        finger_threads.emplace_back(std::thread{
            press_key_thread, i, std::ref(first_finger), std::ref(n),
            std::cref(move_angle), std::cref(touch_delta)});
        break;
      case fingerID::MIDDLE:
        finger_threads.emplace_back(std::thread{
            press_key_thread, i, std::ref(middle_finger), std::ref(n),
            std::cref(move_angle), std::cref(touch_delta)});
        break;
      case fingerID::RING:
        finger_threads.emplace_back(
            std::thread{press_key_thread, i, std::ref(ring_finger), std::ref(n),
                        std::cref(move_angle), std::cref(touch_delta)});
        break;
      case fingerID::LEFT:
        finger_threads.emplace_back(std::thread{shift_arm, i, 0});
        break;
      case fingerID::RIGHT:
        finger_threads.emplace_back(std::thread{shift_arm, i, 1});
        break;
      default:
        break;
    }
  }

  run();

  for (auto &i : finger_threads) {
    i.join();
  }

  return 0;
}
