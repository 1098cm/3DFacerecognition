import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data import TestFace
from data.TestFace import prepare_dataloader, DIR_FAR
from model import Backbone, MobileFaceNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def infer(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, images_v1, idx in tqdm(dataloader):
            images = images.to(device)
            images_v1 = images_v1.to(device)
            embeddings = model(images, images_v1)
            features.append(embeddings.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features

if __name__ == '__main__':
    test_datasets = TestFace.FaceTest(root='data/datasets/Texas3Dface/ori_images', alignment_dir_name='all_images')
    test_datasets_v1 = TestFace.FaceTest(root='data/datasets/Texas3Dface/shape', alignment_dir_name='all_images')
    img_paths = test_datasets.image_paths
    img_paths_v1 = test_datasets_v1.image_paths

    print('total images : {}'.format(len(img_paths)))  # 1149  mate probes: 1023, non mate probes: 21
    print('probe images : {}'.format(len(test_datasets.probe_paths)))  # 1044
    print('gallery images : {}'.format(len(test_datasets.gallery_paths)))  # 105
    dataloader = prepare_dataloader(img_paths, img_paths_v1, batch_size=20, num_workers=0)

    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir').to(device)
    model_path = "models/model_2025-01-02-21-38.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print('Loading weights into state dict...')
    model = model.eval()

    features = infer(model, dataloader)
    DIRs, FARs, thresholds = test_datasets.test_identification(features, ranks=range(1, 51, 1))
    # DIRs, FARs, thresholds = test_datasets.test_identification(features, ranks=[1])
    print(DIRs)

    save_path = 'result'
    os.makedirs(save_path, exist_ok=True)
    pd.DataFrame({'rank': range(1, 51, 1), 'values': DIRs}).to_csv(os.path.join(save_path, 'res50_result.csv'))
