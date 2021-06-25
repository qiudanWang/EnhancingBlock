# pytorch-LMLR
This is an official pytorch implementation of a paper, LMLR: Manipulating Image StyleTransformation via Latent-Space SVM. 

# Requirements
Please install requirements by pip install -r requirements.txt

Python 3.8.5
PyTorch 1.4.0
tqdm

# Usage
## Download data
Hide the link for paper review

## Download models
Hide the link for paper review

## Test
Use --content and --style to provide the respective path to the content and style image directory.

```
python test.py \
   --lamda 1.0 \
   --crop true \
   --content_size 256 \
   --style_size 256 \
   --content data/content \
   --style data/style/vincent-van-gogh \
   --output data/output/vincent-van-gogh \
   --normal_vector data/boundary/vincent-van-gogh/boundary.npy \
   --constant data/boundary/vincent-van-gogh/constant.npy \
   --artist vincent-van-gogh
```

### Some Other Options:

--crop: If crop the input images.

--content_size: New (minimum) size for the content image. Keeping the original size if set to 0.

--style_size: New (minimum) size for the content image. Keeping the original size if set to 0.

--lamda: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).

--preserve_color: Preserve the color of the content image.

--artist: Name of the artist.

--normal_vector: File path to the normal vector.

--constant: File path to the constant.


## Train

### Step one: extract images' latent representations
Use --image_dir and --output_dir to provide the respective directory to the images and outputs.

```
rm -rf data/boundary/vincent-van-gogh/images/list.txt
rm -rf data/boundary/vincent-van-gogh/images/feat_map

python feat.py \
  --style_dir data/boundary/vincent-van-gogh/images \
  --output_dir data/boundary/vincent-van-gogh/images/feat_map
```

### Step two: feed latent representations into SVM classifiers
Use --style_dir and --content_dir to provide the respective directory to the positive and negative samples.
```
python feat_boundary.py \
  --artist vincent-van-gogh \
  --style_dir data/boundary \
  --content_dir data/boundary/real_images
```

## Evaluation
In stylization assessment, we use the [deception rate](https://github.com/CompVis/adaptive-style-transfer/tree/master/evaluation) to categorize styles and use its output score as the probability of the stylized output being classified into the target style; In content assessment, we use [content retention](https://github.com/tensorflow/models/tree/v1.12.0/research/slim) to categorize contents and use its content retention to indicate the retention of the content. 

