#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/lib/python3/dist-packages:/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common" \
    "/usr/bin/python3" \
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/setup.py" \
     \
    build --build-base "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" --install-scripts="/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/bin"
