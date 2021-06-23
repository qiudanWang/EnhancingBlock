import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import net

def style_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

parser = argparse.ArgumentParser()
parser.add_argument("--content_dir", type=str, required=True,
                    help="directory of the style images")
parser.add_argument("--output_dir", type=str, required=True,
                    help="directory of the output latent code")
parser.add_argument('--vgg', type=str, required=False,
                    default='models/vgg_normalised.pth')
args = parser.parse_args()

device = torch.device("cpu")

vgg = net.vgg
decoder = net.decoder
vgg.load_state_dict(torch.load("./models/vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)

style_dir = Path(args.style_dir)
style_paths = [f for f in style_dir.glob('*')]
style_tf = style_transform()
print("Number of style images: %d" % len(style_paths))

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

for style_path in style_paths:
    print("handle image: %s" % style_path.name)
    style = style_tf(Image.open(str(style_path)))
    style = style.to(device).unsqueeze(0)
    style_feats = network.encode_with_intermediate(style)

    output_name = style_path.name.replace(".jpg", ".pth")
    output_file = output_dir / output_name
    torch.save(style_feats[-1], output_file)

