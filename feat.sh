#!/bin/sh

data_dir='./data'

#style_name="real_image"
style_name="vincent-van-gogh"

rm -rf ${data_dir}/boundary/${style_name}/images/list.txt
rm -rf ${data_dir}/boundary/${style_name}/images/feat_map

python feat.py \
  --style_dir ${data_dir}/boundary/${style_name}/images/ \
  --output_dir ${data_dir}/boundary/${style_name}/images/feat_map/

