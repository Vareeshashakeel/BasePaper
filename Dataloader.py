import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np

from utility import RandomIdentitySampler, RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

__factory = {
    'Mars': Mars,
    'iLIDSVID': iLIDSVID,
    'PRID': PRID
}


def train_collate_fn(batch):
    imgs, pids, camids, a = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(a, dim=0)


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids_batch, img_paths


def dataloader(Dataset_name, batch_size=16, num_workers=4, seq_len=4, num_instances=4):
    train_transforms = T.Compose([
        T.Resize([256, 128], interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = __factory[Dataset_name]()

    train_set = VideoDataset_inderase(
        dataset.train,
        seq_len=seq_len,
        sample='intelligent',
        transform=train_transforms
    )

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=RandomIdentitySampler(dataset.train, batch_size, num_instances),
        num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    q_val_set = VideoDataset(dataset.query, seq_len=seq_len, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=seq_len, sample='dense', transform=val_transforms)

    return train_loader, len(dataset.query), num_classes, cam_num, view_num, q_val_set, g_val_set


def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            if len(indices) < self.seq_len:
                indices = np.array(indices)
                indices = np.append(indices, [indices[-1] for _ in range(self.seq_len - len(indices))])
            else:
                indices = np.array(indices)

            imgs = []
            targt_cam = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                targt_cam.append(camid)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            cur_index = 0
            frame_indices = [i for i in range(num)]
            indices_list = []

            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            last_seq = frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)

            indices_list.append(last_seq)
            imgs_list = []
            targt_cam = []

            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    targt_cam.append(camid)
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)

            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, targt_cam, img_paths

        elif self.sample == 'dense_subset':
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)

            cur_index = begin_index
            frame_indices = [i for i in range(num)]
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            last_seq = frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)

            indices_list.append(last_seq)
            imgs_list = []

            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)

            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        elif self.sample == 'intelligent_random':
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
                else:
                    indices.append(random.randint(min(i * each, num - 1), num - 1))

            imgs = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid

        else:
            raise KeyError("Unknown sample method: {}".format(self.sample))


class VideoDataset_inderase(Dataset):
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))
            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
        else:
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
                else:
                    indices.append(random.randint(min(i * each, num - 1), num - 1))

        imgs = []
        labels = []
        targt_cam = []

        for index in indices:
            index = int(index)
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img, temp = self.erase(img)
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
            targt_cam.append(camid)

        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, targt_cam, labels