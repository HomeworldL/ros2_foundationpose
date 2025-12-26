cmake_minimum_required(VERSION 3.8)
project(ros2_foundationpose)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ROS2_REQUIRED_COMPONENTS
    ament_cmake
    ament_cmake_python
    rclpy
)

foreach(COMPONENT ${ROS2_REQUIRED_COMPONENTS})
    find_package(${COMPONENT} REQUIRED)
endforeach()

# To solve: Unknown CMake command “python3_add_library”.
find_package(Python3 REQUIRED Development)
find_package(Python3 REQUIRED Interpreter)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
 
# Install Python executables
install(PROGRAMS
  scripts/foundationpose_node.py
  scripts/pose_display.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config weights
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
