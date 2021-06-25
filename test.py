import argparse
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils import net
from utils.function import adaptive_instance_normalization, coral


def img_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, normal_vector, constant, lamda=1.0):
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    distance = torch.mm(feat.reshape(1, 524288), torch.transpose(normal_vector, 1, 0)) + constant
    if (distance > 0):
      feat = feat + lamda * distance  * normal_vector.reshape(1, 512, 32, 32)
    res = decoder(feat)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='./models/decoder_iter_160000.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--lamda', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument('--normal_vector', type=str,
                    help='File path to the normal vector')
parser.add_argument('--constant', type=str,
                    help='File path to the constant')
parser.add_argument('--artist', type=str,
                    help='artist name')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --contentDir should be given.
assert (args.content_dir)
content_dir = Path(args.content_dir)
content_paths = [f for f in content_dir.glob('*')]

# Either --styleDir should be given.
assert (args.style_dir)
style_dir = Path(args.style_dir)
style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = img_transform(args.content_size, args.crop)
style_tf = img_transform(args.style_size, args.crop)

normal_vector = torch.from_numpy(np.load(args.normal_vector)).to(device)
constant = torch.from_numpy(np.load(args.constant)).to(device)

for content_path in content_paths:
    for style_path in style_paths:
        content = content_tf(Image.open(str(content_path)))
        style = style_tf(Image.open(str(style_path)))
        if args.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, normal_vector, constant,
                                    args.lamda)
        output = output.cpu()

        output_name = output_dir / '{:s}_{:s}_stylized_{:s}{:s}'.format(
            content_path.stem, style_path.stem.replace(" ", ""), args.artist, args.save_ext)
        save_image(output, str(output_name))     
