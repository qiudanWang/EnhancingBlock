import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from utils import net

def image_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True,
                    help="directory of the image images")
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

image_dir = Path(args.image_dir)
image_paths = [f for f in image_dir.glob('*')]
image_tf = image_transform()
print("Number of image images: %d" % len(image_paths))

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

for image_path in image_paths:
    print("handle image: %s" % image_path.name)
    image = image_tf(Image.open(str(image_path)))
    image = image.to(device).unsqueeze(0)
    image_feats = network.encode_with_intermediate(image)

    output_name = image_path.name.replace(".jpg", ".pth")
    output_file = output_dir / output_name
    torch.save(image_feats[-1], output_file)
