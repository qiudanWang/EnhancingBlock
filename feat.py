import argparse                                                                                                                                                                                                                               
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import net

def img_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, required=True,
                    help="directory of the img images")
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

img_dir = Path(args.img_dir)
img_paths = [f for f in img_dir.glob('*')]
img_tf = img_transform()
print("Number of img images: %d" % len(img_paths))

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

for img_path in img_paths:
    print("handle image: %s" % img_path.name)
    img = img_tf(Image.open(str(img_path)))
    img = img.to(device).unsqueeze(0)
    img_feats = network.encode_with_intermediate(img)

    output_name = img_path.name.replace(".jpg", ".pth")
    output_file = output_dir / output_name
    torch.save(img_feats[-1], output_file)
