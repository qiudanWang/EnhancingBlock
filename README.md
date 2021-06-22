# pytorch-LMLP
This is an official pytorch implementation of a paper, LMLP: Manipulating Image StyleTransformation via Latent-Space SVM. 

![image](https://github.com/qiudanWang/LMLP/blob/main/img/Figure0.png)

# Requirements
Please install requirements by pip install -r requirements.txt

Python 3.5+
PyTorch 0.4+
TorchVision
tqdm

# Usage
## Download data
Download xxxxx and put them under data/.

## Download models
Download xxxxx and put them under models/.

## Test
Use --content and --style to provide the respective path to the content and style image.

```
python test.py \
   --lamda 1 \
   --content_size 256 \
   --style_size 256 \
   --crop \
   --content data/content \
   --style data/style/vincent-van-gogh \
   --output data/output/vincent-van-gogh \
   --decoder models/decoder_iter_160000.pth \
   --normal_vector data/boundary/vincent-van-gogh/boundary.npy \
   --constant data/boundary/vincent-van-gogh/constant.npy \
   --artist vincent-van-gogh > logs/vincent-van-gogh.log 
```

This is an example of modifying the value of lamda:


### Some Other Options:

--content_size: New (minimum) size for the content image. Keeping the original size if set to 0.

--style_size: New (minimum) size for the content image. Keeping the original size if set to 0.

--lamda: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).

--preserve_color: Preserve the color of the content image.

## Train
