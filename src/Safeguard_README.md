# Safeguard-system-for-autonomous-flights-in-Mocap-Arena
---

##  Description
This repo contains initials steps for the safeguard system implmented within Optitrack Motion capture system. Project includes ELRS based drone and TX module, remote PC control. Everything is wrapped with ROS2 Humble.

---
## ✨ Features
- PID controller for autonomous flights
- Safeguard to preserve crashing of the drone
- Connection to Mocap system
- ROS2
---
## Requirments
First of all, you should be able to implement the setup from this repo https://github.com/lantacisaza/joy2crsf_ros2.
It will give you manual remote control of the drone through the joystick. Once you have that working setup, you can continue.

### Software:
- Ubuntu (any vesion or any OS that supports ROS2 Humble)
- Betaflight software (for debugging)
- ROS 2 Humble (or newer)
- Python 3.8+
- Motive Software for Optitrack
- Packages:
  - `rclpy`
  - `sensor_msgs`
  - `pyserial`
  - `scipy`
  - `cvxpy`
  - `numpy`
### Hardware:
  - Drone (TinyWhoop Modula6 was used)
  - ELRS TX module 
  - Power supply (in case TX module consumes more than 5V)
  - USB cable for TX-PC connection
  - Joystick (Logitech was used)
  - Optitrack Mocap system
  - Optitrack markers (3 per drone) 
---
## ⚙️ Build
Clone into your workspace and build with `colcon`:

```bash
cd ~/ros2_ws/src
git clone https://github.com/lantacisaza/Safeguard-system-for-autonomous-flights-in-Mocap-Arena
rosdep install --from-paths . --ignore-src -r -y
cd ~/ros2_ws
colcon build 
source install/setup.bash
```
---
## Mocap system
In this project, Optitrack Mocap system was used. Follow procedure below to subscribe to the pos and orientation messeges:
On Mocap PC:
### 1. Turn on Mocap PC and start Motive software.
### 2. Glue 3 markers on the drone and put it in the origin of the arena, then add rigid body in Motive by selecting 3 dots in origin (drone).
### 3. In settings, make sure VRPN streamline is ON.
On your Laptop/PC:
### 1. Connect to the same network as PC connected
### 2. Install this package for VRPN https://docs.ros.org/en/rolling/p/vrpn_mocap/#vrpn-mocap
### 3. Then run (Check IP address):
```bash
cd ros2_ws/
ros2 launch vrpn_mocap client.launch.yaml server:=192.168.1.202 port:=3883
```
### 4. Open another terminal and check whether you see data from the vrpn:
```bash
cd ros2_ws/
ros2 topic list
```
You should see all rigid bodies that are streaming 
### 5. In same terminal, subscribe to the needed rigid body, for example:
```bash
ros2 topic echo /vrpn_mocap/AzamatDrone/pose
```
You should see continous data stream of pose and orientation 
---
## Usage
You will need 5 terminals opened at the same time. 
### 1. Two terminals for the joystick connection, see this repo https://github.com/lantacisaza/joy2crsf_ros2 
### 2. Two terminals for the mocap system, from above
### 3. One terminal to run the main script
```bash
cd ros2_ws
colcon build
source install/setup.bash
ros2 run crsf_joystick test_violation
```
Once everything opened and connected, you can put the drone on the ground, adjust the target points and fly.
LB for arming, RB for changing the mode. You need to use STAB mode to fly correctly.

## Codes explanation
### test_violation.py
This scripts is the main one, where you set the target and handle crsf packets. It works with PID.py to get the right Throttle,Pitch, Yaw and Roll values, and at the sametime uses cbf_safeguard.py to check if drone 
is flying out of range of arena. In case it happens, the target then changes to the origin.
### pid.py
Here is very simple implementation of PID controller which still has to be tuned. Though, it works with some overshooting.
### cbf_safeguard.py
Initial goal was to implement CBF policy, however for now I only check if robot passes thorugh the x,y,z edges of the room and reutrns boolean.

## Problems
### Inconsistency of drone flights
Some times when you arm the drone, it will fly away to the right or to the left. In case that happens, you need to rerun the test_violation.py couple times. There might be a problem with flusihng the data from prev flights or something else.
### Internet connection problems
Sometimes it might happen that you wont be able to connect to mocap. In that case, restarting the PC/Mocap system/Motive software would help.




## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

Portions of the CRSF parsing logic are adapted from the [CRSF Working Group](https://github.com/crsf-wg) repository (MIT License).
