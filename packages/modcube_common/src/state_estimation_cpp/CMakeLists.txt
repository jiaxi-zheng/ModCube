set (CMAKE_CXX_STANDARD 17)
library(state_estimator state_estimator.cpp ekf.cpp)
node(state_estimation_cpp main.cpp state_estimator modcube_alarms modcube_utils)
