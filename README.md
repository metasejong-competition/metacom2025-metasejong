# Guide for MetaSejong Embodied AI Competition 2025

# Release History 

| Version | Features |
|------|------|
|V1.0-R202504XX | **The first release** <br>- Launch IsaacSim Simulator <br>- Start designated simulation application <br>- Usefull *make commands* to develop and test simulation application|

# Step-by-step guide

## Pre-requirement

- Your computer must have NVIDIA GPU

    ``
    ``

- Docker and docker compose is required



## Set up the development environment 

1. clone this repository

```bash
git clone https://github.com/metasejong-competition/metacom2025-metasejong
cd metacom2025-metasejong
```

2. Setup

The following command executes the setup procedure.
This command includes logging into the NVIDIA Docker repository(nvcr.io) and configuring to connect the X server.

```bash
make setup
```

3. Download docker image file and load docker image

download link will be provided: 
loacate file to <path of metacom2025-metasejong>

The following command will load metasejong docker image to your computer.
This command includes logging into the NVIDIA Docker repository(nvcr.io) and configuring to connect the X server.

```bash
make load
```

3. Test given project

Check that the provided project works correctly.

```bash
make run
```

## Download Maps and Robot Models

To properly run the simulation, please download the following resources:

| Resource | Link |
|---------|-----|
|MetaSejong Map| [Download MetaSejong Map]([Map USD](https://drive.google.com/drive/folders/1-LUIkn_XIZH6-GLIeayIvkbB0InJSdva?usp=drive_link)) |
|MetaSejong Robots| [Download Robot Models]([Robot USD](https://drive.google.com/drive/folders/1X-uIbuVpOkLEsq5IaPro3GJ0Ft54pKzH?usp=drive_link)) |

### 📂 Placement Instructions:

1. Download and extract the files from the links above.
2. Place them into the following directory structure:

```
metacom2025-metasejong/
└── resources/
    └── models/
        ├── playground/    # Place MetaSejong Map files here
        └── robot/          # Place Robot models here
```

Example:
```
metacom2025-metasejong/resources/models/meta-sejong/    # Contains MetaSejong map files
metacom2025-metasejong/resources/models/robot/          # Contains robot models (Scout USD + Lidar + Robot Arm + Gripper)
```

Make sure the folder names are exactly:
- `playground`
- `robot`

> 🚩 **Note:** These folders will be automatically mounted inside the Docker container during simulation runtime.



## Develope Simulation Application

1. Write simulation application

write your own Isaac Sim application on **./simulation_app**. e.g. *my_simulation_app.py*. See [link](#write-simulation-app)

2. Link '**run_simulation_app.py**' file to your code
```bash
cd ./simulation_app
rm run_simulation_app.py
ln -s my_simulation_app.py run_simulation_app.py 
```

3. Test your simulation application 

If the computer has been rebooted, the "setup" command must be executed once.
```bash
make setup
```

Run the developed simulation application.
```bash
make run
```

If the Docker image needs to be rebuilt, execute the following command.

```bash
make build-n-run
```

The command to stop the running simulator is as follows.

```bash
make down
```

If you want to delete the virtual volumes along with stopping the simulator, run below command

```bash
make clean
```

## Submission of the final output.

1. Create a docker image file

Executing the following command will save the Docker image, including your developed simulation application, as a tar file.
The generated tar file is named metasejong-2025-release.tar.

```bash
make release
```

# Additional Information

## Index for 'make' command

| command | description |
|-------|---|
|setup|Setup only needs to be executed once. The setup process includes logging into the NVIDIA Docker repository and configuring the X server.|
|run|Run the simulation application in development mode. Launch IsaacSim and start the developed simulation application.|
|build-and-run|Same as 'run', but forces the Docker image to be built. Used when the Docker image needs to be rebuilt.|
|down|Stop the running Docker container.|
|clean|Stop the running Docker container and delete all temporary volumes.|
|build-for-release|This is an intermediate step for 'release'. do not need to execute this command manually.|
|release|Create a tar file for the final submission.|


## Trouble shootings

To be revised...

1. regarding DISPLAY env variable

2. ???




## Write simulation app

### simulation code

**./simulation_app** is for simulation codes. **run_simulation_app.py** will be called when the container startup time.


### user's other files

**./simulation_app** directory is binded to **/simulation_app** under docker container environment. use this directory to store user's data, configuration, etc.

