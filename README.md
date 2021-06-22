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
## Download models
Download xxxxx and put them under models/.
Download xxxxx and put them under data/.

## Test
Use --content and --style to provide the respective path to the content and style image.

This is an example:


### Some Other Options:

--content_size: New (minimum) size for the content image. Keeping the original size if set to 0.
--style_size: New (minimum) size for the content image. Keeping the original size if set to 0.
--alpha: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).
--preserve_color: Preserve the color of the content image.

## Train
