#!/bin/bash

for i in {1..10}; do
    echo "Executing optimize for trial: $i"
    python optimize_cnn.py
done