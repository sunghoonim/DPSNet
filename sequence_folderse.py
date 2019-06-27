import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import os

def load_as_float(path):
    img = np.zeros((544,832,3)).astype(np.float32)
    img[4:, 22:, :] = imread(path).astype(np.float32)
    return img

def quat2mat(q):
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < np.finfo(np.float).eps:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                     [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                     [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])



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

    def __init__(self, root, seed=None, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = sorted([name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))])
        self.scenes = [self.root/folder for folder in scene_list_path]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        for scene in self.scenes:

            # intrinsics
            f_int = open(scene/'gt_cam/cameras.txt', 'r')
            lines_int = f_int.readlines()
            linelist = lines_int[3].split(' ')
            intrinsics = np.array([[float(linelist[4]), 0., float(linelist[6])], [0., float(linelist[5]), float(linelist[7])], [0., 0., 1.]]).astype(np.float32)
            intrinsics[0,:] = intrinsics[0,:] * (810/float(linelist[2]))
            intrinsics[1,:] = intrinsics[1,:] * (540/float(linelist[3]))
            f_int.close()

            # camera order
            f_order = open(scene/'gt_cam/order.txt', 'r')
            lines_order = f_order.readlines()
            orders = []
            for il, line in enumerate(lines_order):
                linelist = line.split(' ')
                orders.append(linelist)

            # camera poses
            f_pose = open(scene/'gt_cam/images.txt', 'r')
            lines_pose = f_pose.readlines()
            linelist_pose = lines_pose[3].split(' ')
            ncam = int(linelist_pose[4].split(',')[0])
            #poses = [None]*ncam
            poses = []
            imgidx = [None]*ncam
            for il, line in enumerate(lines_pose):
                if il >= 4:
                    if il%2 == 0:
                        linelist = line.split(' ')
                        linelist_ = linelist[1:8]
                        imgidx[int(linelist[0])-1] = int((il-4)/2)
                        poses.append([float(qt) for qt in linelist_])

            imgs = sorted((scene/'reference_rgb').files('*.png'))
            gt_depths = sorted((scene/'gt_depth').files('*.npy'))
            gt_demonb = sorted((scene/'DeMoN_best').files('*.npy'))
            gt_demonm = sorted((scene/'DeMoN_median').files('*.npy'))
            gt_deepmvs = sorted((scene/'DeepMVS').files('*.npy'))
            gt_COLMAP = sorted((scene/'COLMAP_unfiltered').files('*.npy'))


            depths = gt_depths[0::2]
            demonb = gt_demonb[0::2]
            demonm = gt_demonm[0::2]
            deepmvs = gt_deepmvs[0::2]
            COLMAP = gt_COLMAP[0::2]

            depths1 = gt_depths[1::2]
            demonb1 = gt_demonb[1::2]
            demonm1 = gt_demonm[1::2]
            deepmvs1 = gt_deepmvs[1::2]
            COLMAP1 = gt_COLMAP[1::2]

            for i in range(len(imgs)):
                img = imgs[i]
                depth = depths[i]
                pose_tgt = np.concatenate((np.concatenate((quat2mat(poses[i][:4]), np.asarray(poses[i][4:]).reshape(3,1)), axis = 1), np.array([[0,0,0,1]])), axis=0)
                sample = {'demonb1':demonb1[i], 'demonm1':demonm1[i], 'deepmvs1':deepmvs1[i], 'colmap1':COLMAP1[i], 'demonb':demonb[i], 'demonm':demonm[i], 'deepmvs':deepmvs[i], 'colmap':COLMAP[i], 'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth, 'ref_imgs': [], 'ref_poses': []}
                order = orders[i]
                for jj in range(1,sequence_length+1):
                    j = int(order[jj])
                    sample['ref_imgs'].append(imgs[j])
                    pose_src = np.concatenate((np.concatenate((quat2mat(poses[j][:4]), np.asarray(poses[j][4:]).reshape(3,1)), axis = 1), np.array([[0,0,0,1]])), axis=0)
                    pose_rel = pose_src @ np.linalg.inv(pose_tgt)
                    pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
                    sample['ref_poses'].append(pose)
                sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        tgt_depth_ = 1/np.load(sample['tgt_depth'])
        tgt_depth = np.zeros((544,832)).astype(np.float32)
        tgt_depth[4:, 22:] = tgt_depth_.astype(np.float32)
        scale = 1/np.amin(tgt_depth[tgt_depth>0])
        tgt_depth = tgt_depth*scale
        #demonb = scale/np.load(sample['demonb'])
        #demonm = scale/np.load(sample['demonm'])
        #deepmvs = scale/np.load(sample['deepmvs'])
        #colmap = scale/np.load(sample['colmap'])
        #demonb1 = np.load(sample['demonb1']).astype(np.float32)
        #demonm1 = np.load(sample['demonm1']).astype(np.float32)
        #deepmvs1 = np.load(sample['deepmvs1']).astype(np.float32)
        #colmap1 = np.load(sample['colmap1']).astype(np.float32)
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        ref_poses = [np.concatenate((ref_pose[:,:,:3], ref_pose[:,:,3:]*scale), axis=2) for ref_pose in sample['ref_poses']]
        if self.transform is not None:
            imgs, tgt_depth, intrinsics = self.transform([tgt_img] + ref_imgs, tgt_depth, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        intrinsics[0,2] = intrinsics[0,2] + 22
        intrinsics[1,2] = intrinsics[1,2] + 4

        return tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth, scale#, demonb, demonm, deepmvs, colmap, demonb1, demonm1, deepmvs1, colmap1

    def __len__(self):
        return len(self.samples)
