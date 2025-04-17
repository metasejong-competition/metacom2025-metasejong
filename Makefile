# /*******************************************************************************
#  * Copyright 2025 AISL Sejong University, Korea
#  *
#  *	Licence: TBD
#  *******************************************************************************/

.PHONY: help setup run down load
.SILENT: help

help:
	echo "See README.md in this folder"

# This tool now only supports compose V2, aka "docker compose" as it has replaced to old docker-compose tool.
DOCKER_COMPOSE=docker compose
DOCKER_TAR_FILE=metasejong-metacom2025-r02.tar
RELEASE_PROJECT_PATH=../metacom2025-metasejong/

# Setup only needs to be executed once.
# The setup process includes logging into the NVIDIA Docker repository and configuring the X server.
setup:
	docker login nvcr.io
	xhost +local:

# Stop the running Docker container.
down:
	${DOCKER_COMPOSE} -f docker-compose-release.yml down

run:
	${DOCKER_COMPOSE} -f docker-compose-release.yml up

load:
	docker load -i $(DOCKER_TAR_FILE)
