import argparse
from pathlib import Path

import numpy as np
import torch

from utils.boundary import train_boundary


def load_latent_code(file, index, dim):
    file_path = file.parent / file.name.replace("-0.pth", "-{}.pth".format(index))
    return torch.load(file_path).reshape(1, dim)


parser = argparse.ArgumentParser()
parser.add_argument("--style_dir", type=str, required=False, default="data/boundary",
                    help="directory of the style images which includes feature map directory named feat_map")
parser.add_argument("--style_name", type=str, required=False, default="monet_water-lilies-1914",
                    help="name of the style to train boundary with svm")
args = parser.parse_args()

style_dir = Path(args.style_dir)
style_name = args.style_name

latent_code_arr = []
scores = None
positive_num = 0

for style_path in [ style_dir / style_name, "data/boundary/real_images/" ]:
    print("Process %s" % style_path)
    style_path = Path(style_path)
    latent_code_dir = style_path / 'images' / 'feat_map'
    latent_code_files = [f for f in latent_code_dir.glob('*.pth')]

    for latent_code_path in latent_code_files:
        latent_code = torch.load(latent_code_path).reshape(1, 512 * 32 * 32).data.cpu().numpy()[0]
        latent_code_arr.append(latent_code)

    latent_code_num = len(latent_code_files)
    if style_path.name != "real_images":
        style_score = np.ones(shape=(latent_code_num,1))
        positive_num = latent_code_num
    else:
        style_score = np.zeros(shape=(latent_code_num,1)) - np.ones(shape=(latent_code_num,1))

    scores = style_score if (scores is None) else np.append(scores, style_score, axis=0)

latent_code_arr = np.array(latent_code_arr)

chosen_num_or_ratio = positive_num / len(latent_code_arr)

boundary, constant, a, b = train_boundary(
    latent_codes=latent_code_arr,
    chosen_num_or_ratio=chosen_num_or_ratio,
    scores=scores)

np.save(style_dir / style_name / 'boundary.npy', boundary)
np.save(style_dir / style_name / 'constant.npy', constant)

print(boundary)
print(constant)

