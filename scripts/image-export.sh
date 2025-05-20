#!/bin/bash
docker save -o vana-inference.tar vana-inference
gzip -k vana-inference.tar
SHA256=$(shasum -a 256 vana-inference.tar.gz | awk '{ print $1 }')
echo "Image exported to 'vana-inference.tar.gz' with SHA256: $SHA256"