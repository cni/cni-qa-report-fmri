#!/bin/bash
# Builds the container.
# The container can be exported using the export.sh script
GEAR=scitran/qa-report-fmri:v0.1.7
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

docker build --no-cache --tag $GEAR $DIR
