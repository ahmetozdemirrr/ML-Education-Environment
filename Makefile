# Makefile for ML Education Environment

# Python interpreter
PYTHON = python3

# Let's assume there are no users in the docker group.
DOCKER_COMPOSE = sudo docker-compose

DATASETS_SCRIPT_DIR = ./project_datasets
DATASETS_TARGET_DIR = $(DATASETS_SCRIPT_DIR)/datasets
DATA_DOWNLOADER_SCRIPT = $(DATASETS_SCRIPT_DIR)/data_downloader.py

# colors
YELLOW = \033[0;33m
RED = \033[0;31m
GREEN = \033[0;32m
RESET_COLOR = \033[0m

.PHONY: all start setup download-datasets build up down logs clean-datasets clean help

all: start

start: setup up

# Setup: download datasets and build a Docker image
setup: download-datasets build

# download datasrets
download-datasets:
	@echo "$(YELLOW)>>> Running dataset download script...$(RESET_COLOR)"
	@echo "$(YELLOW)Datasets will be saved in the '$(DATASETS_TARGET_DIR)' directory.$(RESET_COLOR)"
	@mkdir -p $(DATASETS_TARGET_DIR)
	$(PYTHON) $(DATA_DOWNLOADER_SCRIPT)
	@echo "$(GREEN)>>> Dataset download completed.$(RESET_COLOR)"

build:
	@echo "$(YELLOW)>>> Creating Docker images...$(RESET_COLOR)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)>>> Docker images have been created successfully.$(RESET_COLOR)"

up:
	@echo "$(YELLOW)>>> Starting Docker containers...$(RESET_COLOR)"
	$(DOCKER_COMPOSE) up

down:
	@echo "$(YELLOW)>>> Stopping and removing Docker containers...$(RESET_COLOR)"
	$(DOCKER_COMPOSE) down --remove-orphans
	@echo "$(GREEN)>>> All services have been suspended.$(RESET_COLOR)"

logs:
	@echo "$(YELLOW)>>> Showing service logs (exit with Ctrl+C)...$(RESET_COLOR)"
	$(DOCKER_COMPOSE) logs -f

clean-datasets:
	@echo "$(YELLOW)>>> Cleaning up downloaded datasets ($(DATASETS_TARGET_DIR))...$(RESET_COLOR)"
	@printf "$(RED)WARNING: All files in '$(DATASETS_TARGET_DIR)' will be deleted. Are you sure? (y/n) $(RESET_COLOR)" && read choice; \
	if [ "$${choice}" = "y" ] || [ "$${choice}" = "Y" ]; then \
	    echo "$(YELLOW)Deleting: $(DATASETS_TARGET_DIR)/* $(RESET_COLOR)"; \
	    rm -rf $(DATASETS_TARGET_DIR)/*; \
	    echo "$(GREEN)Datasets have been cleaned.$(RESET_COLOR)"; \
	elif [ "$${choice}" = "n" ] || [ "$${choice}" = "N" ]; then\
	    echo "$(GREEN)Dataset cleanup operation was canceled.$(RESET_COLOR)"; \
	else \
		echo "$(YELLOW)Enter a valid character.$(RESET_COLOR)"; \
	fi

clean: down
	@echo "$(YELLOW)>>> System is being cleaned...$(RESET_COLOR)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)No __pycache__ directories found.$(RESET_COLOR)"
	@find . -type f -name "*.pyc" -delete 2>/dev/null || echo "$(YELLOW)No .pyc files found.$(RESET_COLOR)"
	@echo "$(GREEN)>>> Cleaning is complete.$(RESET_COLOR)"

help:
	@echo ""
	@echo "$(YELLOW)Available Makefile targets:$(RESET_COLOR)"
	@echo "  $(YELLOW)make all / start$(RESET_COLOR)       : Downloads datasets, builds Docker images, and starts services (default)."
	@echo "  $(YELLOW)make setup$(RESET_COLOR)             : It downloads datasets and builds Docker images."
	@echo "  $(YELLOW)make download-datasets$(RESET_COLOR) : It just runs the dataset download script."
	@echo "  $(YELLOW)make build$(RESET_COLOR)             : Creates/updates images for Docker services."
	@echo "  $(YELLOW)make up$(RESET_COLOR)                : Starts (builds if necessary) Docker containers."
	@echo "  $(YELLOW)make down$(RESET_COLOR)              : Stops and removes Docker containers."
	@echo "  $(YELLOW)make logs$(RESET_COLOR)              : Shows the logs of running services."
	@echo "  $(YELLOW)make clean-datasets$(RESET_COLOR)    : Deletes downloaded datasets."
	@echo "  $(YELLOW)make clean$(RESET_COLOR)             : Stops/removes all Docker containers and deletes datasets."
	@echo "  $(YELLOW)make help$(RESET_COLOR)              : This displays the help menu."
	@echo ""
