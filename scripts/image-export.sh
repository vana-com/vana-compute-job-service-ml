#!/bin/bash
docker save -o inference-job.tar inference-job
gzip -k inference-job.tar
SHA256=$(shasum -a 256 inference-job.tar.gz | awk '{ print $1 }')
echo "Image exported to 'inference-job.tar.gz' with SHA256: $SHA256"