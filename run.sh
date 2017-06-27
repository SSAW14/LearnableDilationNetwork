#!/bin/bash

./caffe-dynamic-dilation/.build_release/tools/caffe.bin train --solver=solver.prototxt --weights=init.caffemodel --gpu=0  2>&1 | tee log.txt

echo "Done."
