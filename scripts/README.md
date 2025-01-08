## scripts

This directory is meant to store the bash-scripts to perform multiple tasks:

- `build.sh`: Creates the virtual environment for the project locally. If a venv is already created, overwrites the current version with updated packages.
- `build_docker.sh`: This script creates the docker image with the dependencies and tools necessary for the code. It further grabs the Dockerfile in the project directory to finish the process.
- `docker_entrypoint.sh`: This script connects the docker build process with the Dockerfile.
- `run_docker.sh`: Opens a docker container based on the previously created docker image. Can be called multiple times to add new terminals for parallel running. However, will overwrite the current programs. A better practice would be to use nohup in combination.

All scripts are designed to be executed in the project directory (pwd). This means that they have to be called with:

```bash
scripts/"$(name_script)".sh
```

You can perform them in any shell since the shebang determines `bash` as interpreter. Thus you should at least have some bash interpreter installed on the applied machine.