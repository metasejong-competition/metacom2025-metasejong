# /*******************************************************************************
#  * Copyright 2025 AISL Sejong University, Korea
#  *
#  *	Licence: TBD
#  *******************************************************************************/

.PHONY: help setup run down load download clean
.SILENT: help

help:
	echo "See README.md in this folder"

# This tool now only supports compose V2, aka "docker compose" as it has replaced to old docker-compose tool.
DOCKER_COMPOSE=docker compose
DOCKER_IMAGE_FILE_ID=10r-tzDj0qS6OKWEle0gl4GnRtEitVKD5
DOCKER_IMAGE_TAR_FILE=metasejong-metacom2025-r02.tar



# Setup only needs to be executed once.
# The setup process includes logging into the NVIDIA Docker repository and configuring the X server.
setup:
	# docker login nvcr.io
	xhost +local:

# Stop the running Docker container.
down:
	${DOCKER_COMPOSE} -f docker-compose.yml down

run:
	${DOCKER_COMPOSE} -f docker-compose.yml up



download:
	@if [ ! -f "$(DOCKER_IMAGE_TAR_FILE)" ]; then \
		echo "Downloading file from Google Drive..."; \
		if ! command -v gdown >/dev/null 2>&1; then \
			echo "Installing gdown..."; \
			pip3 install --user gdown; \
		fi; \
		PYTHON_USER_BIN=$$(python3 -m site --user-base)/bin; \
		if [ -d "$$PYTHON_USER_BIN" ]; then \
			export PATH="$$PYTHON_USER_BIN:$$PATH"; \
		fi; \
		gdown "https://drive.google.com/uc?id=$(DOCKER_IMAGE_FILE_ID)" -O $(DOCKER_IMAGE_TAR_FILE); \
		if [ ! -f "$(DOCKER_IMAGE_TAR_FILE)" ] || [ $$(stat -f%z "$(DOCKER_IMAGE_TAR_FILE)") -lt 1000000 ]; then \
			echo "Error: Download failed or file is too small."; \
			rm -f "$(DOCKER_IMAGE_TAR_FILE)"; \
			exit 1; \
		fi; \
		echo "Download complete: $(DOCKER_IMAGE_TAR_FILE)"; \
	else \
		echo "File $(DOCKER_IMAGE_TAR_FILE) already exists."; \
	fi

clean:
	rm -f $(DOCKER_IMAGE_TAR_FILE) cookies.txt

load: $(DOCKER_IMAGE_TAR_FILE)
	@echo "Loading Docker image..."
	docker load -i $(DOCKER_IMAGE_TAR_FILE)

$(DOCKER_IMAGE_TAR_FILE):
	@make download
