# /*******************************************************************************
#  * Copyright 2025 AISL Sejong University, Korea
#  *
#  *	Licence: TBD
#  *******************************************************************************/

.PHONY: help setup run down load download clean
.SILENT: help

help:
	echo "See README.md in this folder"


# playground를 포함하여 models 전체 복사하고자 하는 경우 true 
# ((주의)) Makefile.deploy 파일도 수정해줘야 함 
WITH_PG=true


# This tool now only supports compose V2, aka "docker compose" as it has replaced to old docker-compose tool.
DOCKER_COMPOSE=docker compose

DOCKER_RELEASE_REVISION=r06
DOCKER_IMAGE_FILE_ID="1fWGCelUpPuHxYX9ECWaa8SxL5NXOrL72"
# demo, dongcheon, jiphyeon, gwanggaeto 
ENV_METASEJONG_SCENARIO ?= demo

ifeq ($(WITH_PG), true)
	#	이미지 명칭과 export할 tar 파일 명칭에 추가할 옵션
	DOCKER_IMAGE_OPTION=-with-playground
else
	#	이미지 명칭과 export할 tar 파일 명칭에 추가할 옵션
	DOCKER_IMAGE_OPTION=
endif

DOCKER_IMAGE_TAR_FILE=metasejong-metacom2025$(DOCKER_IMAGE_OPTION)-$(DOCKER_RELEASE_REVISION).tar

# Setup only needs to be executed once.
# The setup process includes logging into the NVIDIA Docker repository and configuring the X server.
setup:
	docker login nvcr.io
	xhost +local:

# Start a Docker container.
run: setup
	ENV_METASEJONG_SCENARIO=$(ENV_METASEJONG_SCENARIO) $(DOCKER_COMPOSE) -f docker-compose.yml up

# Stop the running Docker container.
down:
	${DOCKER_COMPOSE} -f docker-compose.yml down

# Remove Docker image file
clean:
	rm -f $(DOCKER_IMAGE_TAR_FILE)

# Download Docker image file from google drive link
download:
	@echo "Download the Docker image file from https://drive.google.com/uc?id=$(DOCKER_IMAGE_FILE_ID) and save it to your project folder."

# Load docker image using download image tar file
load:
	@echo "Loading Docker image..."
	docker load -i $(DOCKER_IMAGE_TAR_FILE)
