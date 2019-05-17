##### How to get ros for python3 working #####

# install dependencies for rospy
sudo apt-get install python3-pip python3-yaml
pip3 install rospkg catkin_pkg

# install dependencies to build cv_bridge
sudo apt-get install python-catkin-tools python3-dev python3-numpy
mkdir ~/catkin_build_ws && cd ~/catkin_build_ws
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install

# build cv_bridge
mkdir src
cd src
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
cd ~/catkin_build_ws
catkin build cv_bridge

# always source this stuff
source devel/setup.bash --extend