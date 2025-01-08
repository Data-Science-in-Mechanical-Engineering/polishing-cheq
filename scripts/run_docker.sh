#!/bin/bash
read -p 'container_name: ' container_name
read -p 'device_id: ' device_id
export PROJECT_DIR="${PWD}"

if [ ! "$(docker ps -aqf name=$container_name)" ]; then
    # If no container is already running
    case "$(uname -s)" in
    Linux*)  # linux system
    echo "Linux system detected. Using Bash script."
    docker run -it --rm --name $container_name --gpus $device_id -v /$PROJECT_DIR:/polishing_robot -p 8889:8888 polishing_robot;;
    Darwin*)  # apple system
    echo "Mac system detected. Using Bash script."
    docker run -it --rm --name $container_name --gpus $device_id -v /$PROJECT_DIR:/polishing_robot -p 8889:8888 polishing_robot;;
    CYGWIN*|MINGW*|MSYS*)  # windows system
    echo "Windows system detected. Using Windows script."
    winpty docker run -it --rm --name $container_name --gpus $device_id -v /$PROJECT_DIR:/polishing_robot -p 8889:8888 polishing_robot;;
    *)  # something else, or not specified in case distinction:
    echo "Unsupported operating system. Please run the container manually."
    exit 1 ;;
    esac
else
    # If you need a second terminal window running on the same container
    export container_id="$(docker ps -aqf name=$container_name)"
    case "$(uname -s)" in
    Linux*)  # linux system
      echo "Linux system detected. Using Bash script."
      docker exec -it $container_id bash;;
    Darwin*)  # apple system
      echo "Mac system detected. Using Bash script."
      docker exec -it $container_id bash;;
    CYGWIN*|MINGW*|MSYS*)  # windows system
      echo "Windows system detected. Using Windows script."
      winpty docker exec -it $container_id bash;;
    *)  # something else, or not specified in case distinction:
      echo "Unsupported operating system. Please run the container manually."
    exit 1 ;;
    esac
fi