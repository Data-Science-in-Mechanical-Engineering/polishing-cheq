## scripts

This directory is meant to store the bash-scripts to perform multiple tasks:

- `build.sh`: Creates the virtual environment for the project locally. If a venv is already created, overwrites the current version with updated packages. (Helpful tool for updating the written code to the venv)
- `build_apptainer.sh`: Creates the container using Apptainer build. This is necessary for the application in HPC. The base of the apptainer container is the docker image that is created with the `build_docker.sh` script. However, since docker is permitted on HPC we have to create the image outside of the HPC and upload the docker image to the hub in order to access it from the HPC.
- `build_docker.sh`: This script creates the docker image with the dependencies and tools necessary for the code. It further grabs the Dockerfile in the project directory to finish the process.
- `docker_entrypoint.sh`: This script connects the docker build process with the Dockerfile.
- `jobscript.sh`: This script is used to perform the code using the apptainer container that has been created previously. First it connects your WANDB_KEY to the apptainer and then it performs two tasks in the background for the parallel run. It is written using the SLURM method.
- `run_docker.sh`: Opens a docker container based on the previously created docker image. Can be called multiple times to add new terminals for parallel running. This method is meant to be performed on the DSME cluster.
- `run.sh`: Prepares two Git Bash Terminals to start the process run with Asychronous Actor and Learner.
- `test.sh`: Performs pytest.

All scripts are designed to be executed in the project directory (pwd). This means that they have to be called with:

```bash
scripts/"$(name_script)".sh
```

You can perform them in any directory since the shebang determines `bash` as interpreter. Thus you should at least have some bash interpreter installed on the applied machine.