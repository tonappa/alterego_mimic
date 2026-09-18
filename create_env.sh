#!/bin/bash
# Creates the conda env for the tracker, only if it does not exist yet.
# On the robots that already run alterego_object_detection the env ros_yolo exists: nothing to do.
#   ./create_env.sh            -> env ros_yolo
#   ./create_env.sh my_env     -> another name (then: roslaunch ... conda_env:=my_env)
set -e
ENV_NAME="${1:-ros_yolo}"
cd "$(dirname "$(realpath "$0")")"
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env $ENV_NAME already exists, checking the imports"
else
    conda create --name "$ENV_NAME" --yes python=3.10
    conda run -n "$ENV_NAME" pip install -r requirements.txt
fi
conda run -n "$ENV_NAME" python -c "import ultralytics, pyrealsense2, cv2, rospkg; print('env OK, ultralytics', ultralytics.__version__)"
