#!/bin/bash

curl -LOk https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
apt-get install unzip
unzip data.zip
