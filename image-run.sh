#!/bin/bash
mkdir -p input output working
docker run --rm -p 8000:8000 -v ./input:/mnt/input -v ./output:/mnt/output -v ./working:/mnt/working vana-inference