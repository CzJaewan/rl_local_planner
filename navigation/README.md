
# SETTING
## Model
```
  git clone https://github.com/CzJaewan/servingbot.git
```
## Gazebo & Amcl & Globalplanner
```
  git clone https://github.com/CzJaewan/rl_avoidance_gazebo.git
```
nce test in gazebo simulator 

# rl_avoidance_gazebo ( single agent )

## RUN used Waitpoint 
- Servingbot Gazebo world
```
  roslaunch servingbot_gazebo servingbot_rl_world.launch
``` 
- Amcl & Map & Globalplanner
```
  roslaunch gazebo_rl_test servingbot_rl.launch 
```
- PPO local planner
```
  cd ~/catkin_ws/src/rl_avoidance_gazebo/GAZEBO_TEST_R
  change the from test_world import StageWorld' -> 'from test_world_wp import StageWorld' in test_run.py 
  python test_run.py
```

## RUN used Look a head
- Servingbot Gazebo world
```
  roslaunch servingbot_gazebo servingbot_rl_world.launch
``` 
- Amcl & Map & Globalplanner & waypoint generator
```
  roslaunch gazebo_rl_test servingbot_rl_lookahead.launch 
```
- PPO local planner
```
  cd ~/catkin_ws/src/rl_avoidance_gazebo/GAZEBO_TEST_R
  change the from test_world import StageWorld' -> 'from test_world_LAH import StageWorld' in test_run.py 
  python test_run_r.py
```


# rl_avoidance_gazebo ( Multi agent )

## RUN 
- Servingbot Gazebo world
```
  roslaunch servingbot_gazebo multi-servingbot_rl_world.launch
``` 
- Amcl & Map & Globalplanner & waypoint generator
```
  roslaunch gazebo_rl_test multi_servingbot_rl.launch 
```
- PPO local planner
```
  cd ~/catkin_ws/src/rl_avoidance_gazebo/GAZEBO_TEST_R
  change the from test_world import StageWorld' -> 'from test_world import StageWorld' in test_run.py 
  mpiexec -np 2 python test_run_r.py
```
- Goal pub
```
  rostopic pub robot_0/move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: -10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'  mpexec -np 2 python multi_goal_pub.py
  rostopic pub robot_1/move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: 10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

## RUN used Look a head <- 수정 중 아직안됨.
- Servingbot Gazebo world
```
  roslaunch servingbot_gazebo multi-servingbot_rl_world.launch
``` 
- Amcl & Map & Globalplanner & Look a head generator
```
  roslaunch gazebo_rl_test multi_servingbot_rl_lookahead.launch 
```
- PPO local planner
```
  cd ~/catkin_ws/src/rl_avoidance_gazebo/GAZEBO_TEST_R
  change the from test_world import StageWorld' -> 'from test_world_LAH import StageWorld' in test_run.py 
  mpiexec -np 2 python test_run_r.py
```
- Goal pub
```
  cd ~/catkin_ws/src/rl_avoidance_gazebo/
  mpexec -np 2 python multi_goal_pub.py

```


# Policy Data
- Policy Data link
```
  https://drive.google.com/drive/folders/1vQ1XKbU1Lid40Vm92H0UDRcvquYNSrYi?usp=sharing

  - origin : rl-collision-avoidance reward
  - rc : rl-collision-avoidance reward, collision reward(50)
  - rc2 : rl-collision-avoidance reward, collision reward(30)
  - rg : rl-collision-avoidance reward, time_distance reward(abs)
  - rt : rl-collision-avoidance reward, time reward(-0.1)
  - rt2 : rl-collision-avoidance reward, time reward(-0.5)
  - rt05 : rl-collision-avoidance reward, time reward(-0.05)
  - rw : rl-collision-avoidance reward, omega reward remove
```  
  
