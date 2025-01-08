#!/bin/bash
# polishing_robot
# create the virtual environment and install dependencies

python3_10_cmd=false
py_cmd=false
# Check if python ot python3 command is available
if command -v python3.10 &>/dev/null; then
    echo "Found Python as command..."
    python3_10_installed=true
elif command -v python &>/dev/null; then
    echo "Found Python as command..."
elif command -v py &>/dev/null; then
    echo "Found py as command..."
    py_cmd=true
else
    echo "No Python version seems to be installed. Please install Python and try again."
    exit 1
fi

# Create a virtual environment
if [ -d "venv" ]; then
    echo "Virtual Environment already exists..."
else
    # Create virtual environment based on python version
    if [ "$python3_10_cmd" = true ]; then
        python3.10 -m venv venv
    elif [ "$py_cmd" = true ]; then
        py -m venv venv
    else
        python -m venv venv
    fi
    echo "Virtual Environment initialized."
fi

# Activate the virtual environment based on the operating system
echo "Activating virtual environment..."
case "$(uname -s)" in
    Linux*)  # linux system
    echo "Linux system detected. Using Bash activation script."    
    source "venv/bin/activate" ;;
    Darwin*)  # apple system
    echo "Mac system detected. Using Bash activation script."
    source "venv/bin/activate" ;;
    CYGWIN*|MINGW*|MSYS*)  # windows system
    echo "Windows system detected. Using Windows activation script." 
    source "venv\Scripts\activate" ||. "venv\Scripts\activate" ;;
    *)  # something else, or not specified in case distinction:
    echo "Unsupported operating system. Please activate the virtual environment manually."
    exit 1 ;;
esac

# Check if virtual environment is activated (used for decoding)
if [[ $VIRTUAL_ENV == "" ]]; then
    echo "Failed to activate the virtual environment. Please activate it manually."
    exit 1
else
    echo "Successfully activated virtual environment."
fi

# Install dependencies from setup.py using pip
echo "Installing dependencies..."
pip install .

echo "Setup complete!"