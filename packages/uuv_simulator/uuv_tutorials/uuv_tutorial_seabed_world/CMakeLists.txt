cmake_minimum_required(VERSION 3.0.2)
project(uuv_tutorial_seabed_world)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch worlds models
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)
