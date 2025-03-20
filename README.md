# Compute Engine Job PoC

This project serves as PoC simulation of a Compute Engine job made by DLP owners.

## Overview

The worker leverages the test sqlite DB mounted to `/mnt/input/data.db` (overridable via `INPUT_PATH` env variable) following an unwrapped schema used for demoing the query engine in the hope of an eventual, consistent E2E workflow.

It processes the input data and outputs a `stats.json` under `/mnt/output/stats.json` (overridable via `OUTPUT_PATH` env variable).

## Utility scripts

These are sugar scripts for docker commands to build, export, and run the worker image consistently for simpler dev cycles / iteration.

The `image-export.sh` script builds an exportable `.tar` for uploading in remote services for registering with the compute engine / image registry contracts.