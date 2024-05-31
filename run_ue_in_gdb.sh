#!/bin/bash
docker compose -f ping_docker_compose.yml -f ping_docker_compose.gdb.yml run oai-nr-ue
