#!/bin/bash

set -x
set -e

# Clone TensorFlow project in proto directory
ls src/main/proto/tensorflow/**/*[a-n] | xargs rm
ls src/main/proto/tensorflow/**/*[p-z] | xargs rm
