#!/bin/sh

data_dir="./data"

style_name=vincent-van-gogh

python test.py \
   --alpha 1 \
   --content_size 256 \
   --style_size 256 \
   --crop \
   --content_dir ${data_dir}/content \
   --style_dir ${data_dir}/style/${style_name} \
   --output ${data_dir}/output/${style_name} \
   --decoder models/decoder_iter_160000.pth \
   --normal_vector  data/boundary/${style_name}/boundary.npy \
   --constant data/boundary/${style_name}/constant.npy \
   --artist ${style_name} > logs/${style_name}.log 

