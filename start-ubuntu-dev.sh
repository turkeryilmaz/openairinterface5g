#!/bin/bash

docker run -it --rm --entrypoint bash -w /oai-ran-live -v $(pwd):/oai-ran-live ran-dev:ubuntu22-latest
