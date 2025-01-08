#!/bin/bash
# polishing_robot
# prepare and run cmd files for act.py and learn.py

# Function to find git-bash.exe
find_git_bash() {
    # Possible installation directories
    local paths=(
        "/c/Program Files/Git/git-bash.exe"
        "/c/Program Files (x86)/Git/git-bash.exe"
        "/c/Users/$USERNAME/AppData/Local/Programs/Git/git-bash.exe"
    )

    for path in "${paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

# Find the path to git-bash.exe
git_bash_path=$(find_git_bash)

if [[ -z "$git_bash_path" ]]; then
    echo "git-bash.exe not found. Please install Git Bash or add its path to the search list."
    exit 1
fi

echo "Using Git Bash at: $git_bash_path"

current_dir=$(cygpath -u "$(pwd)")

# Open the first Git Bash window for the Learner
"$git_bash_path" -c "cd \"$current_dir\" && source venv/Scripts/activate && python learn.py && exec bash" &

# Wait for a moment to allow the first window to initialize
sleep 2

# Open the second Git Bash window for the Actor
"$git_bash_path" -c "cd \"$current_dir\" && source venv/Scripts/activate && python act.py && exec bash" &