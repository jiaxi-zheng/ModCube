cmake_minimum_required(VERSION 3.0.2)
project(uuv_teleop)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)

catkin_install_python(PROGRAMS scripts/vehicle_teleop.py
                               scripts/finned_uuv_teleop.py
                               scripts/vehicle_keyboard_teleop.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
