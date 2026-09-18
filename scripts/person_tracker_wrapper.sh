#!/bin/bash
# Runs person_tracker.py with the Python of a conda env (ultralytics + pyrealsense2), as
# alterego_object_detection does. Without hardcoded paths:
#   env:        PERSON_TRACKER_CONDA_ENV (launch arg conda_env), default ros_yolo
#   conda base: CONDA_BASE if set, else ~/miniconda3, ~/anaconda3, ~/miniforge3, /opt/conda, else `conda info --base`
# ROS is already set up by roslaunch/rosrun; /opt/ros/noetic is sourced only if it is not.

ENV_NAME="${PERSON_TRACKER_CONDA_ENV:-ros_yolo}"
HERE="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"

find_base() {
    if [ -n "${CONDA_BASE:-}" ] && [ -x "$CONDA_BASE/envs/$ENV_NAME/bin/python" ]; then echo "$CONDA_BASE"; return; fi
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        [ -x "$d/envs/$ENV_NAME/bin/python" ] && { echo "$d"; return; }
    done
    if command -v conda > /dev/null 2>&1; then
        d="$(conda info --base 2> /dev/null)"
        [ -x "$d/envs/$ENV_NAME/bin/python" ] && { echo "$d"; return; }
    fi
}

BASE="$(find_base)"
if [ -z "$BASE" ]; then
    echo "person_tracker: conda env '$ENV_NAME' not found (set CONDA_BASE or the launch arg conda_env)" >&2
    exit 1
fi

# Same activation as alterego_object_detection (library paths of the env)
source "$BASE/bin/activate" "$ENV_NAME"

if [ -z "${ROS_PACKAGE_PATH:-}" ]; then
    source /opt/ros/noetic/setup.bash
fi

exec "$BASE/envs/$ENV_NAME/bin/python" "$HERE/person_tracker.py" "$@"
