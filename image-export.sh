#!/bin/bash
docker save -o vana-inference.tar vana-inference
echo "Image exported to vana-inference.tar"
echo "To compress: gzip vana-inference.tar"
echo "To calculate checksum: sha256sum vana-inference.tar.gz | cut -d' ' -f1"