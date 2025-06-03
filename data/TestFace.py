import os

import cv2
import numpy as np
import torch
from PIL import Image
from cv2 import imread
from skimage.transform import warp, estimate_transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_all_files(root):
    all_files = []
    for filenames in os.listdir(root):
        file_path = os.path.join(root, filenames)
        all_files.append(file_path)
    return all_files


class FaceTest:
    def __init__(self, root='CASIA-3D_test', alignment_dir_name='aligned_pad_0.1_pad_high'):

        self.image_paths = get_all_files(os.path.join(root, alignment_dir_name))
        self.image_paths = np.array(self.image_paths).astype(object).flatten()

        self.probe_paths = get_all_files(os.path.join(root, 'probe'))
        self.probe_paths = np.array(self.probe_paths).astype(object).flatten()

        self.gallery_paths = get_all_files(os.path.join(root, 'gallery'))
        self.gallery_paths = np.array(self.gallery_paths).astype(object).flatten()

        self.init_proto(self.probe_paths, self.gallery_paths)

    def get_key(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    def get_label(self, image_path):
        # return int(os.path.basename(image_path).split('_')[2])  # Tex3d

        # return int(os.path.basename(image_path).split('_')[0])  # Tinyface（作废）

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # # if base_name.split('_')[-1] == 'shape':
        # #     name_parts = base_name.split('_')[:-3]
        # #     name = '_'.join(name_parts)
        # #     return name  # Facescrub
        # # name_parts = base_name.split('_')[:-2]
        # # name = '_'.join(name_parts)
        # # return name  # Facescrub
        #
        if base_name.split('_')[-1] == 'shape':
            name_parts = base_name.split('_')[:-2]
            name = '_'.join(name_parts)
            return name  # Facescrub
        name_parts = base_name.split('_')[:-1]
        name = '_'.join(name_parts)
        return name  # Facescrub （部分）

        # return int(os.path.basename(image_path).split('-')[0])  # CASIA-3D

    def init_proto(self, probe_paths, match_paths):
        index_dict = {}
        for i, image_path in enumerate(self.image_paths):
            index_dict[self.get_key(image_path)] = i

        self.indices_probe = np.array([index_dict[self.get_key(img)] for img in probe_paths])
        self.indices_gallery = np.array([index_dict[self.get_key(img)] for img in match_paths])

        self.labels_probe = np.array([self.get_label(img) for img in probe_paths])
        self.labels_gallery = np.array([self.get_label(img) for img in match_paths])

    def test_identification_v1(self, features, ranks=None):
        if ranks is None:
            ranks = [1, 5, 20]
        feat_probe = features[self.indices_probe]
        feat_gallery = features[self.indices_gallery]
        # compare_func = inner_product
        compare_func = euclidean_distance
        score_mat = compare_func(feat_probe, feat_gallery)

        label_mat = self.labels_probe[:, None] == self.labels_gallery[None, :]

        # results, _, _ = DIR_FAR(score_mat, label_mat, ranks)
        DIRs, FARs, thresholds = DIR_FAR(score_mat, label_mat, ranks, FARs=[0.7])

        return DIRs, FARs, thresholds

    def test_identification(self, features, ranks=None):
        if ranks is None:
            ranks = [1, 5, 20]
        feat_probe = features[self.indices_probe]
        feat_gallery = features[self.indices_gallery]
        # compare_func = inner_product
        compare_func = euclidean_distance
        score_mat = compare_func(feat_probe, feat_gallery)

        label_mat = self.labels_probe[:, None] == self.labels_gallery[None, :]

        # results, _, _ = DIR_FAR(score_mat, label_mat, ranks)
        DIRs, FARs, thresholds = DIR_FAR(score_mat, label_mat, ranks)

        return DIRs, FARs, thresholds


def euclidean_distance(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)

    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("输入数组必须是二维的")

    diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]

    # 计算欧式距离
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return 1 - dist_matrix


def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_false_indices=False):
    '''
    Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC)
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_false_indices:    not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks,
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    assert score_mat.shape == label_mat.shape
    # assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    match_indices = label_mat.astype(bool).any(axis=1)
    score_mat_m = score_mat[match_indices, :]
    label_mat_m = label_mat[match_indices, :]
    score_mat_nm = score_mat[np.logical_not(match_indices), :]
    label_mat_nm = label_mat[np.logical_not(match_indices), :]

    print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as threshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
        openset = False
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[
                   0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)
        openset = True

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=bool)
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])[::-1]
        sorted_label_mat_m[row, :] = label_mat_m[row, sort_idx]

    # Calculate DIRs for different FARs and ranks
    if openset:
        gt_score_m = score_mat_m[label_mat_m]
        assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    if get_false_indices:
        false_retrieval = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=bool)
        false_reject = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=bool)
        false_accept = np.zeros([len(FARs), len(ranks), score_mat_nm.shape[0]], dtype=bool)
    for i, threshold in enumerate(thresholds):
        for j, rank in enumerate(ranks):
            success_retrieval = sorted_label_mat_m[:, 0:rank].any(axis=1)
            if openset:
                success_threshold = gt_score_m >= threshold
                DIRs[i, j] = (success_threshold & success_retrieval).astype(np.float32).mean()
            else:
                DIRs[i, j] = success_retrieval.astype(np.float32).mean()
            if get_false_indices:
                false_retrieval[i, j] = ~success_retrieval
                false_accept[i, j] = score_mat_nm.max(1) >= threshold
                if openset:
                    false_reject[i, j] = ~success_threshold
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()

    if get_false_indices:
        return DIRs, FARs, thresholds, match_indices, false_retrieval, false_reject, false_accept, sort_idx_mat_m
    else:
        return DIRs, FARs, thresholds


def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-5):
    assert len(score_vec.shape) == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    # score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0] + epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1] - epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm == 0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm - 1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    else:
        raise NotImplementedError
    return old_size, center


class MTCNN(object):
    def __init__(self, device='cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)

    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None, ...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, 'bbox'


class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            bbox = [left, top, right, bottom]
            return bbox, 'kpt68'


class ListDatasetv1(Dataset):
    def __init__(self, img_list, iscrop=True, image_is_saved_with_swapped_B_and_R=True):
        super(ListDatasetv1, self).__init__()
        self.img_list = img_list
        self.transform = transforms.Compose(
            [transforms.Resize((112, 112)), transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.face_detector = MTCNN()
        self.iscrop = iscrop
        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        img = imread(image_path)
        if self.image_is_saved_with_swapped_B_and_R:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = np.array(img)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        if self.iscrop:
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
            else:
                left = bbox[0]
                right = bbox[2]
                top = bbox[1]
                bottom = bbox[3]
            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size * 1.25)
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        DST_PTS = np.array([[0, 0], [0, 112 - 1], [112 - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.

        dst_image = warp(image, tform.inverse, output_shape=(112, 112))
        dst_image = dst_image.transpose(2, 0, 1)
        return torch.tensor(dst_image).float(), idx


class ListDataset_v2(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=True):
        super(ListDataset_v2, self).__init__()
        self.img_list = img_list
        self.transform = transforms.Compose(
            [transforms.Resize((112, 112)), transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image not found or failed to load at {image_path}")
            if self.image_is_saved_with_swapped_B_and_R:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return None

        img = Image.fromarray(img)
        img = self.transform(img)

        return img, idx


class ListDataset(Dataset):
# /mnt/volume3/home/cz/msx/test_data/Texas3Dface/ori_images/all_images
# /mnt/volume3/home/cz/msx/test_data/Texas3Dface/shape/all_images
# Clean_0001_001_20050913115022_Portrait_shape.jpg
# Clean_0001_001_20050913115022_Portrait.png
    def __init__(self, img_list, img_list_v1, image_is_saved_with_swapped_B_and_R=True):
        super(ListDataset, self).__init__()
        self.img_list = img_list
        self.img_list_v1 = img_list_v1
        self.transform = transforms.Compose(
            [transforms.Resize((112, 112)), transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        assert len(self.img_list) == len(self.img_list_v1)

        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        # img = cv2.imread(image_path)
        # img = img[:, :, :3]
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image not found or failed to load at {image_path}")
            if self.image_is_saved_with_swapped_B_and_R:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return None

        img = Image.fromarray(img)
        img = self.transform(img)
        
        # image_path_v1 = image_path.replace("ori_images", "shape").replace(".png", "_shape.jpg")   # Texas/Facescrub（部分）
        # image_path_v1 = image_path.replace("ori_images", "shape").replace(".jpg", "_shape.jpg")  # tinyface
        # image_path_v1 = image_path.replace("ori_images", "shape").replace(".bmp", "_shape.jpg")  # CASIA3D
        image_path_v1 = image_path.replace("ori_images", "shape").replace(".jpeg", "_shape.jpg").replace(".png", "_shape.jpg").replace(".bmp", "_shape.jpg")  # Facescrub

        # image_path_v1 = self.img_list_v1[idx]
        # img = cv2.imread(image_path)
        # img = img[:, :, :3]
        # print(os.path.basename(image_path))
        # print(os.path.basename(image_path_v1))
        # assert os.path.basename(image_path).replace('.png', '_shape.jpg') == os.path.basename(image_path_v1)  # Texas/Facescrub（部分）
        # assert os.path.basename(image_path).replace('.jpg', '_shape.jpg') == os.path.basename(image_path_v1)  # tinyface
        # assert os.path.basename(image_path).replace('.bmp', '_shape.jpg') == os.path.basename(image_path_v1)  # CASIA3D
        assert os.path.basename(image_path).replace('.jpeg', '_shape.jpg').replace(".png", "_shape.jpg").replace(".bmp", "_shape.jpg") == os.path.basename(image_path_v1)  # Facescrub

        try:
            img_v1 = cv2.imread(image_path_v1)
            if img_v1 is None:
                raise ValueError(f"Image not found or failed to load at {image_path_v1}")
            if self.image_is_saved_with_swapped_B_and_R:
                img_v1 = cv2.cvtColor(img_v1, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image at {image_path_v1}: {e}")
            return None

        img_v1 = Image.fromarray(img_v1)
        img_v1 = self.transform(img_v1)
        return img, img_v1, idx  # imgs,shape


def prepare_dataloader(img_list, img_list_v1, batch_size, num_workers=0, image_is_saved_with_swapped_B_and_R=True):
    image_dataset = ListDataset(img_list, img_list_v1,
                                image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R)  # Tex3D
    # image_dataset = ListDatasetv1(img_list, iscrop=True)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader


def prepare_dataloader_v1(img_list, batch_size, num_workers=0, image_is_saved_with_swapped_B_and_R=True):
    # image_dataset = ListDataset(img_list, img_list_v1, image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R)  # Tex3D
    # image_dataset = ListDatasetv1(img_list, iscrop=False, image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R)
    image_dataset = ListDataset_v2(img_list, image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader
