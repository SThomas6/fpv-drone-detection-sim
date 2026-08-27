# radar — simulated radar detection (Phase 4)

Not started. Will compute range / bearing / elevation / radial velocity from
the Gazebo ground-truth pose stream (`/world/detection_world/dynamic_pose/info`)
relative to the station's `radar_link` pose, plus a noise/detection-probability
model, publishing radar plots as ROS 2 messages.
