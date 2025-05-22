#!/bin/bash
mkdir -p output working
docker run --rm -p 8000:8000 --gpus all -v ./working:/mnt/working -v ./output:/mnt/output inference-job