#!/bin/bash
mkdir -p input output working
docker run --rm -p 8000:8000 -v ./working:/mnt/working -v ./output:/mnt/output vana-inference