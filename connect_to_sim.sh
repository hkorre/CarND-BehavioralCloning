#!/bin/bash

pwd=$(echo $PWD)

docker run --rm \
           -p 4567:4567 \
           -v $pwd:/home/behavior_cloning \
           -w /home/behavior_cloning  \
           -it hkorre_udacity:anaconda \
           python drive.py model.json
         
