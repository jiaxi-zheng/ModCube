cmake_minimum_required(VERSION 3.0.2)
project(uuv_tutorial_disturbances)
find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)
