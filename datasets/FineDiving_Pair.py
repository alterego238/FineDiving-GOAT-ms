import mindspore as ms
import mindspore.ops as ops
import numpy as np
import os
import pickle
import random
import glob
#from torchvideotransforms import video_transforms, volume_transforms
from msvideo.data import transforms
from os.path import join
from PIL import Image


class FineDiving_Pair_Dataset:
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        if args.use_i3d_bb:
            args.feature_root = args.i3d_feature_path
        elif args.use_swin_bb:
            args.feature_root = args.swin_feature_path
        else:
            args.feature_root = args.bpbb_feature_path
        self.args = args
        self.subset = subset
        self.transforms = transform
        self.random_choosing = args.random_choosing
        self.action_number_choosing = args.action_number_choosing
        self.length = args.length
        self.img_size = args.img_size
        self.num_boxes = args.num_boxes
        self.out_size = args.out_size
        self.num_selected_frames = args.num_selected_frames
        self.voter_number = args.voter_number
        print(args.feature_root)

        # file path
        self.data_root = args.data_root
        self.data_anno = self.read_pickle(args.label_path)
        with open(args.train_split, 'rb') as f:
            self.train_dataset_list = pickle.load(f)
        with open(args.test_split, 'rb') as f:
            self.test_dataset_list = pickle.load(f)
        with open(args.feature_root, 'rb') as f:
            self.feature_dict = pickle.load(f)
        with open(args.feamap_root, 'rb') as f:
            self.feamap_dict = pickle.load(f)
        self.boxes_dict = pickle.load(open(args.boxes_path, 'rb'))
        self.cnn_feature_dict = self.read_pickle(args.cnn_feature_path)
        self.formation_features_dict = pickle.load(open(args.formation_feature_path, 'rb'))
        self.bp_feature_path = args.bp_feature_path

        # transforms
        self.transforms = transforms.Compose([
            transforms.VideoResize(self.img_size),
            transforms.VideoToTensor(),
            transforms.VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.action_number_dict = {}
        self.difficulties_dict = {}
        if self.subset == 'train':
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.action_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.action_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def preprocess(self):
        for item in self.train_dataset_list:
            dive_number = self.data_anno.get(item)[0]
            if self.action_number_dict.get(dive_number) is None:
                self.action_number_dict[dive_number] = []
            self.action_number_dict[dive_number].append(item)
        if self.subset == 'test':
            for item in self.test_dataset_list:
                dive_number = self.data_anno.get(item)[0]
                if self.action_number_dict_test.get(dive_number) is None:
                    self.action_number_dict_test[dive_number] = []
                self.action_number_dict_test[dive_number].append(item)

    def check_exemplar_dict(self):
        if self.subset == 'train':
            for key in sorted(list(self.action_number_dict.keys())):
                file_list = self.action_number_dict[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key
        if self.subset == 'test':
            for key in sorted(list(self.action_number_dict_test.keys())):
                file_list = self.action_number_dict_test[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key

    def load_video(self, frames_path):
        length = self.length
        transforms = self.transforms
        image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
        if len(image_list) >= length:
            start_frame = int(image_list[0].split("/")[-1][:-4])
            end_frame = int(image_list[-1].split("/")[-1][:-4])
            frame_list = np.linspace(start_frame, end_frame, length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(length)]
            video = [Image.open(image_list[image_frame_idx[i]]) for i in range(length)]
            return transforms(video).transpose(0, 1), image_frame_idx
        else:
            T = len(image_list)
            img_idx_list = np.arange(T)
            img_idx_list = img_idx_list.repeat(2)
            idx_list = np.linspace(0, T * 2 - 1, length).astype(np.int)
            image_frame_idx = [img_idx_list[idx_list[i]] for i in range(length)]

            video = [Image.open(image_list[image_frame_idx[i]]) for i in range(length)]
            return transforms(video).transpose(0, 1), image_frame_idx

    def load_transits(self, video_file_name):
        image_list = sorted((glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))
        # longer than length
        if len(image_list) >= self.length:
            start_frame = int(image_list[0].split("/")[-1][:-4])
            end_frame = int(image_list[-1].split("/")[-1][:-4])
            frame_list = np.linspace(start_frame, end_frame, self.length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

            # video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
            frames_labels = [self.data_anno.get(video_file_name)[4][i] for i in image_frame_idx]
            frames_catogeries = list(set(frames_labels))
            frames_catogeries.sort(key=frames_labels.index)
            transitions = [frames_labels.index(c) for c in frames_catogeries]
            # return self.transforms(video), np.array([transitions[1]-1,transitions[-1]-1]), np.array(frames_labels)
            return np.array([transitions[1]-1,transitions[-1]-1]), np.array(frames_labels)
        # shorter than length
        else:
            frame_labels_1 = self.data_anno.get(video_file_name)[4]
            TT = frame_labels_1.shape[0]
            frame_labels_1 = ms.Tensor.from_numpy(frame_labels_1).unsqueeze(-1).broadcast_to((len(image_list), 2))
            frame_labels_1 = frame_labels_1.reshape(-1)
            TT1 = frame_labels_1.shape[0]
            frame_labels_1 = frame_labels_1.asnumpy().tolist()

            # assert T1 == TT1 and T == TT
            select_space = np.linspace(0, TT1 - 1, self.length).astype(np.int)
            select_frame_list = [select_space[i] for i in range(self.length)]

            # video_1 = torch.cat([video_1[:, ii, :, :].unsqueeze(1) for ii in select_frame_list], 1)
            frame_labels_1 = [frame_labels_1[ii] for ii in select_frame_list]
            frames_catogeries = list(set(frame_labels_1))
            frames_catogeries.sort(key=frame_labels_1.index)
            transitions = [frame_labels_1.index(c) for c in frames_catogeries]
            # return video_1, np.array([transitions[1]-1,transitions[-1]-1]), np.array(frame_labels_1)
            return np.array([transitions[1]-1,transitions[-1]-1]), np.array(frame_labels_1)

    def load_idx(self, frames_path):
        length = 5406
        image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
        if len(image_list) >= length:
            start_frame = int(image_list[0].split("/")[-1][:-4])
            end_frame = int(image_list[-1].split("/")[-1][:-4])
            frame_list = np.linspace(start_frame, end_frame, length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(length)]
            return image_frame_idx
        else:
            T = len(image_list)
            img_idx_list = np.arange(T)
            img_idx_list = img_idx_list.repeat(2)
            idx_list = np.linspace(0, T * 2 - 1, length).astype(np.int)
            image_frame_idx = [img_idx_list[idx_list[i]] for i in range(length)]
            return image_frame_idx

    def load_boxes(self, key, image_frame_idx, out_size):  # T,N,4
        key_bbox_list = [(key[0], str(key[1]), str(i).zfill(4)) for i in image_frame_idx]
        N = self.num_boxes
        H, W = out_size
        boxes = []
        for key_bbox in key_bbox_list:
            person_idx_list = []
            for i, item in enumerate(self.boxes_dict[key_bbox]['box_label']):
                if item == 'person':
                    person_idx_list.append(i)
            tmp_bbox = []
            tmp_x1, tmp_y1, tmp_x2, tmp_y2 = 0, 0, 0, 0
            for idx, person_idx in enumerate(person_idx_list):
                if idx < N:
                    box = self.boxes_dict[key_bbox]['boxes'][person_idx]
                    box[:2] -= box[2:] / 2
                    x, y, w, h = box.tolist()
                    x = x * W
                    y = y * H
                    w = w * W
                    h = h * H
                    tmp_x1, tmp_y1, tmp_x2, tmp_y2 = x, y, x + w, y + h
                    tmp_bbox.append(ms.Tensor([x, y, x + w, y + h]).unsqueeze(0))  # 1,4 x1,y1,x2,y2
            if len(person_idx_list) < N:
                step = len(person_idx_list)
                while step < N:
                    tmp_bbox.append(ms.Tensor([tmp_x1, tmp_y1, tmp_x2, tmp_y2]).unsqueeze(0))  # 1,4
                    step += 1
            boxes.append(ops.concat(tmp_bbox).unsqueeze(0))  # 1,N,4
        boxes_tensor = ops.concat(boxes)
        return boxes_tensor

    def random_select_frames(self, video, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        select_list_per_clip = [i for i in range(16)]
        selected_frames_list = []
        selected_frames_idx = []
        for i in range(length // 10):
            random_sample_list = random.sample(select_list_per_clip, num_selected_frames)
            selected_frames_list.extend([video[10 * i + j].unsqueeze(0) for j in random_sample_list])
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in random_sample_list])
        selected_frames = ops.concat(selected_frames_list, dim=0)  # 540*t,C,H,W; t=num_selected_frames
        return selected_frames, selected_frames_idx

    def random_select_idx(self, image_frame_idx):
        length = 5406
        num_selected_frames = self.num_selected_frames
        select_list_per_clip = [i for i in range(16)]
        selected_frames_idx = []
        for i in range(length // 10):
            random_sample_list = random.sample(select_list_per_clip, num_selected_frames)
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in random_sample_list])
        return selected_frames_idx

    def select_middle_idx(self, image_frame_idx):
        length = 5406
        num_selected_frames = self.num_selected_frames
        selected_frames_idx = []
        for i in range(length // 10):
            sample_list = [16 // (num_selected_frames + 1) * (j + 1) - 1 for j in range(num_selected_frames)]
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in sample_list])
        return selected_frames_idx

    def load_goat_data(self, data: dict, key: tuple):
        if self.args.use_goat:
            if self.args.use_formation:
                # use formation features
                data['formation_features'] = self.formation_features_dict[key]  # 540,1024 [Middle]
            elif self.args.use_bp:
                # use bp features
                file_name = key[0] + '_' + str(key[1]) + '.npy'
                bp_features_ori = ms.Tensor(np.load(os.path.join(self.bp_feature_path, file_name)))  # T_ori,768
                if bp_features_ori.shape[0] == 768:
                    bp_features_ori = bp_features_ori.reshape(-1, 768)
                frames_path = os.path.join(self.data_root, key[0], str(key[1]))
                image_frame_idx = self.load_idx(frames_path)  # T,C,H,W
                if self.args.random_select_frames:
                    selected_frames_idx = self.random_select_idx(image_frame_idx)
                else:
                    selected_frames_idx = self.select_middle_idx(image_frame_idx)
                bp_features_list = [bp_features_ori[i].unsqueeze(0) for i in selected_frames_idx]  # [1,768]
                data['bp_features'] = ops.concat(bp_features_list, dim=0).to(ms.float32)  # 540,768
            elif self.args.use_self:
                data = data
            else:
                if self.args.use_cnn_features:
                    frames_path = os.path.join(self.data_root, key[0], str(key[1]))
                    image_frame_idx = self.load_idx(frames_path)  # T,C,H,W
                    if self.args.random_select_frames:
                        selected_frames_idx = self.random_select_idx(image_frame_idx)
                    else:
                        selected_frames_idx = self.select_middle_idx(image_frame_idx)
                    data['boxes'] = self.load_boxes(key, selected_frames_idx, self.out_size)  # 540*t,N,4
                    data['cnn_features'] = self.cnn_feature_dict[key].squeeze(0)
                else:
                    frames_path = os.path.join(self.data_root, key[0], str(key[1]))
                    video, image_frame_idx = self.load_video(frames_path)  # T,C,H,W
                    if self.args.random_select_frames:
                        selected_frames, selected_frames_idx = self.random_select_frames(video, image_frame_idx)
                    else:
                        selected_frames, selected_frames_idx = self.select_middle_frames(video, image_frame_idx)
                    data['boxes'] = self.load_boxes(key, selected_frames_idx, self.out_size)  # 540*t,N,4
                    data['video'] = selected_frames  # 540*t,C,H,W
        return data

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        data = {}
        data['feature'] = self.feature_dict[sample_1]
        data['feamap'] = self.feamap_dict[sample_1]
        frames_path = os.path.join(self.data_root, sample_1[0], str(sample_1[1]))
        data['transits'], data['frame_labels'] = self.load_transits(sample_1)
        data['number'] = self.data_anno.get(sample_1)[0]
        data['final_score'] = self.data_anno.get(sample_1)[1]
        data['difficulty'] = self.data_anno.get(sample_1)[2]
        data['completeness'] = (data['final_score'] / data['difficulty'])

        # goat
        data = self.load_goat_data(data, sample_1)

        # choose an exemplar for video
        if self.subset == 'train':
            # train phrase
            if self.action_number_choosing == True:
                file_list = self.action_number_dict[self.data_anno[sample_1][0]].copy()
            elif self.DD_choosing == True:
                file_list = self.difficulties_dict[self.data_anno[sample_1][2]].copy()
            else:
                # randomly
                file_list = self.train_dataset_list.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            # target['video'], target['transits'], target['frame_labels'] = self.load_video(sample_2)
            target['feature'] = self.feature_dict[sample_2]
            target['feamap'] = self.feamap_dict[sample_2]
            frames_path = os.path.join(self.data_root, sample_2[0], str(sample_2[1]))
            target['transits'], target['frame_labels'] = self.load_transits(sample_2)
            target['number'] = self.data_anno.get(sample_2)[0]
            target['final_score'] = self.data_anno.get(sample_2)[1]
            target['difficulty'] = self.data_anno.get(sample_2)[2]
            target['completeness'] = (target['final_score'] / target['difficulty'])

            # goat
            target = self.load_goat_data(target, sample_2)

            return data, target
        else:
            # test phrase
            if self.action_number_choosing:
                train_file_list = self.action_number_dict[self.data_anno[sample_1][0]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            elif self.DD_choosing:
                train_file_list = self.difficulties_dict[self.data_anno[sample_1][2]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                # tmp['video'], tmp['transits'], tmp['frame_labels'] = self.load_video(item)
                tmp['feature'] = self.feature_dict[item]
                tmp['feamap'] = self.feamap_dict[item]
                frames_path = os.path.join(self.data_root, item[0], str(item[1]))
                tmp['transits'], tmp['frame_labels'] = self.load_transits(item)
                tmp['number'] = self.data_anno.get(item)[0]
                tmp['final_score'] = self.data_anno.get(item)[1]
                tmp['difficulty'] = self.data_anno.get(item)[2]
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])

                # goat
                tmp = self.load_goat_data(tmp, item)

                target_list.append(tmp)
            return data, target_list

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    import traceback
    from mindspore.dataset import GeneratorDataset
    import os, sys
    sys.path.append(os.getcwd())
    from utils.misc import import_class
    
    def get_video_trans():
        train_trans = transforms.Compose([
            transforms.VideoRandomHorizontalFlip(),
            transforms.VideoResize((112,112)),
            transforms.VideoRandomCrop(112),
            transforms.VideoToTensor(),
            transforms.VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_trans = transforms.Compose([
            transforms.VideoResize((112,112)),
            transforms.VideoCenterCrop(112),
            transforms.VideoToTensor(),
            transforms.VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_trans, test_trans

    def dataset_builder(args):
        train_trans, test_trans = get_video_trans()
        DatasetGenerator = import_class("datasets." + args.benchmark)
        train_dataset = DatasetGenerator(args, transform=train_trans, subset='train')
        #train_dataset = GeneratorDataset(train_dataset_generator, ["data", "target"], num_parallel_workers=args.workers)
        test_dataset = DatasetGenerator(args, transform=test_trans, subset='test')
        #test_dataset = GeneratorDataset(test_dataset_generator, ["data", "target"], shuffle=False, num_workers=args.workers)
        return train_dataset, test_dataset
    
    from mindspore.common.initializer import One, Normal
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--benchmark', type=str, choices=['FineDiving'], default='FineDiving', help='dataset')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer name')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training (interrupted by accident)')
    parser.add_argument('--voter_number', type=int, help='voter_number', default=10)

    # basic
    parser.add_argument('--seed', type=int, help='manual seed', default=42)
    parser.add_argument('--workers', type=int, help='number of subprocesses for dataloader', default=24)

    # goat setting below
    # cnn
    parser.add_argument('--length', type=int, help='length of videos', default=96)
    parser.add_argument('--img_size', type=tuple, help='input image size', default=(224, 224))
    parser.add_argument('--out_size', type=tuple, help='output image size', default=(25, 25))
    parser.add_argument('--crop_size', type=tuple, help='RoiAlign image size', default=(5, 5))

    # gcn
    parser.add_argument('--num_boxes', type=int, help='boxes number of each frames', default=8)
    parser.add_argument('--num_selected_frames', type=int, help='number of selected frames per 16 frames', default=1)

    # path
    parser.add_argument('--data_root', type=str, help='root of dataset', default='/mnt/petrelfs/daiwenxun/AS-AQA/Video_result')
    parser.add_argument('--label_path', type=str, help='path of annotation file', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/Anno_result/anno_dict.pkl')
    parser.add_argument('--boxes_path', type=str, help='path of boxes annotation file', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/DINO/ob_result_new.pkl')
    # backbone features path
    parser.add_argument('--i3d_feature_path', type=str, help='path of i3d feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/video_feature_dict.pkl')
    parser.add_argument('--swin_feature_path', type=str, help='path of swin feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/video-swin-features/swin_features_dict_new.pkl')
    parser.add_argument('--bpbb_feature_path', type=str, help='path of bridge-prompt feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/bpbb_features_540.pkl')
    # attention features path
    parser.add_argument('--feamap_root', type=str, help='path of feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/video_feamap_dict.pkl')
    parser.add_argument('--train_split', type=str, help='', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/Anno_result/train_split3.pkl')
    parser.add_argument('--test_split', type=str, help='', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/Anno_result/test_split3.pkl')
    parser.add_argument('--cnn_feature_path', type=str, help='path of cnn feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Inceptionv3/inception_feature_dict.pkl')
    parser.add_argument('--stage1_model_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Group-AQA-Distributed/ckpts/STAGE1_256frames_rho0.3257707338254451_(224, 224)_(25, 25)_loss82.48323059082031.pth', help='stage1_model_path')
    parser.add_argument('--bp_feature_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/bp_features', help='bridge prompt feature path')
    parser.add_argument('--formation_feature_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/formation_features_middle_1.pkl', help='formation feature path')

    # [BOOL]
    # bool for attention mode[GOAT / BP / FORMATION / SELF]
    parser.add_argument('--use_goat', type=int, help='whether to use group-aware-attention', default=1)
    parser.add_argument('--use_bp', type=int, help='whether to use bridge prompt features', default=0)
    parser.add_argument('--use_formation', type=int, help='whether to use formation features', default=0)
    parser.add_argument('--use_self', type=int, help='whether to use self attention', default=0)
    # bool for backbone[I3D / SWIN / BP]
    parser.add_argument('--use_i3d_bb', type=int, help='whether to use i3d as backbone', default=1)
    parser.add_argument('--use_swin_bb', type=int, help='whether to use swin as backbone', default=0)
    parser.add_argument('--use_bp_bb', type=int, help='whether to use bridge-prompt as backbone', default=0)
    # others
    parser.add_argument('--random_select_frames', type=int, help='whether to select frames randomly', default=0)
    parser.add_argument('--use_cnn_features', type=int, help='whether to use pretrained cnn features', default=1)

    # fixed parameters
    parser.add_argument('--random_choosing', type=int, help=' ', default=0)
    parser.add_argument('--action_number_choosing', type=int, help=' ', default=1)

    args = parser.parse_args()

    from mindspore.dataset import GeneratorDataset
    train_dataset_generator, test_dataset_generator = dataset_builder(args)

    test_dataset = GeneratorDataset(test_dataset_generator, ["data", "target"], shuffle=False, num_parallel_workers=args.workers)
    test_dataset = test_dataset.batch(batch_size=args.bs_test)
    test_dataloader = test_dataset.create_tuple_iterator()
    data = next(test_dataset.create_dict_iterator())
    print(data["data"].keys(), data["target"].keys())
    print(next(test_dataloader))