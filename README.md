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
Use --content and --style to provide the respective path to the content and style image directory.

```
python test.py 
   --lamda 1.0 
   --crop true
   --content_size 256 
   --style_size 256 
   --content data/content 
   --style data/style/vincent-van-gogh 
   --output data/output/vincent-van-gogh 
   --normal_vector data/boundary/vincent-van-gogh/boundary.npy 
   --constant data/boundary/vincent-van-gogh/constant.npy 
   --artist vincent-van-gogh
```

This is an example of modifying the value of lamda:


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
