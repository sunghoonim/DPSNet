import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random

def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, ttype='train.txt', sequence_length=2, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/ttype
        scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.ttype = ttype
        self.scenes = sorted(scenes)
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = sequence_length//2

        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            poses = np.genfromtxt(scene/'poses.txt').astype(np.float32)
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(len(imgs)):
                if i < demi_length:
                    shifts = list(range(0,sequence_length))
                    shifts.pop(i)
                elif i >= len(imgs)-demi_length:
                    shifts = list(range(len(imgs)-sequence_length,len(imgs)))
                    shifts.pop(i-len(imgs))
                else:
                    shifts = list(range(i-demi_length, i+(sequence_length+1)//2))
                    shifts.pop(demi_length)

                img = imgs[i]
                depth = img.dirname()/img.name[:-4] + '.npy'
                pose_tgt = np.concatenate((poses[i,:].reshape((3,4)), np.array([[0,0,0,1]])), axis=0)
                sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth, 'ref_imgs': [], 'ref_poses': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[j])
                    pose_src = np.concatenate((poses[j,:].reshape((3,4)), np.array([[0,0,0,1]])), axis=0)
                    pose_rel = pose_src @ np.linalg.inv(pose_tgt)
                    pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
                    sample['ref_poses'].append(pose)
                sequence_set.append(sample)
        if self.ttype == 'train.txt':
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        tgt_depth = np.load(sample['tgt_depth'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        ref_poses = sample['ref_poses']
        if self.transform is not None:
            imgs, tgt_depth, intrinsics = self.transform([tgt_img] + ref_imgs, tgt_depth, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth

    def __len__(self):
        return len(self.samples)
