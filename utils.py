import os
import torch
from torchvision import datasets, transforms
import config as c
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import glob
import torch.nn.functional as F

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m))
    return torch.cat(flat_maps, dim=1)[..., None]


def get_loss(z, jac):
    z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
    jac = sum(jac)
    x = z ** 2
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]

def gen_anomaly_map(z, jac):
    flow_maps: List[Tensor] = []
    flow_maps2: List[Tensor] = []  
    log_like: List[Tensor] = []   
        
    for (hidden_variable, jacobian) in zip(z, jac):
        
        #log_prob = -torch.sum(z ** 2, dim=(1,))*0.5
        log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5

        prob = torch.exp(log_prob)
        flow_map = F.interpolate(
            input=-prob,
            size=512,
            mode="bilinear",
            align_corners=False,
        )
        flow_maps.append(flow_map)

    flow_maps = torch.stack(flow_maps, dim=-1) #torch.Size([1, 1, 256, 256, 3])
    anomaly_score = torch.max(flow_maps)
    
    return anomaly_score


def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)

def load_datasets2(dataset_path, class_name):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_path>:

    train data:

            dataset_path/class_name/train/good/any_filename.png
            dataset_path/class_name/train/good/another_filename.tif
            dataset_path/class_name/train/good/xyz.png
            [...]

    test data:

        'normal data' = non-anomalies

            dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
            dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
            dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
            dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
            dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
            [...]

        anomalies - assume there are anomaly classes 'crack' and 'curved'

            dataset_path/class_name/test/crack/dat_crack_damn.png
            dataset_path/class_name/test/crack/let_it_crack.png
            dataset_path/class_name/test/crack/writing_docs_is_fun.png
            [...]

            dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
            dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
            [...]
    '''

    def target_transform(target):
        return class_perm[target]

    if c.pre_extracted:
        trainset = FeatureDataset(train=True)
        testset = FeatureDataset(train=False)
    else:
        data_dir_train = os.path.join(dataset_path, class_name, 'train')
        data_dir_test = os.path.join(dataset_path, class_name, 'test')
        
        print(data_dir_train, data_dir_test)

        classes = os.listdir(data_dir_train)
        if '.ipynb_checkpoints' in classes:
            os.rmdir(data_dir_train+'/.ipynb_checkpoints')
            
        classes = os.listdir(data_dir_test)
        print(classes)
        if 'OK' not in classes:
            print('There should exist a subdirectory "good". Read the doc of this function for further information.')
            exit()
        classes.sort()
        class_perm = list()
        class_idx = 1
        for cl in classes:
            if cl == 'OK':
                class_perm.append(0)
            else:
                class_perm.append(class_idx)
                class_idx += 1

        tfs = [transforms.Resize(c.img_size), transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)]
        transform_train = transforms.Compose(tfs)

        trainset = ImageFolder(data_dir_train, transform=transform_train)
        testset = ImageFolder(data_dir_test, transform=transform_train, target_transform=target_transform)
    return trainset, testset


def load_datasets(dataset_path, class_name):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_path>:

    train data:

            dataset_path/class_name/train/good/any_filename.png
            dataset_path/class_name/train/good/another_filename.tif
            dataset_path/class_name/train/good/xyz.png
            [...]

    test data:

        'normal data' = non-anomalies

            dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
            dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
            dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
            dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
            dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
            [...]

        anomalies - assume there are anomaly classes 'crack' and 'curved'

            dataset_path/class_name/test/crack/dat_crack_damn.png
            dataset_path/class_name/test/crack/let_it_crack.png
            dataset_path/class_name/test/crack/writing_docs_is_fun.png
            [...]

            dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
            dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
            [...]
    '''

    def target_transform(target):
        return class_perm[target]

    if c.pre_extracted:
        trainset = FeatureDataset(train=True)
        testset = FeatureDataset(train=False)
    else:
        data_dir_train = os.path.join(dataset_path, class_name, 'train')
        data_dir_test = os.path.join(dataset_path, class_name, 'test') 
        
        print(data_dir_train, data_dir_test)

        classes = os.listdir(data_dir_train)
        if '.ipynb_checkpoints' in classes:
            os.rmdir(data_dir_train+'/.ipynb_checkpoints')
        print(os.listdir(data_dir_train))
        if ('good' not in classes) and ('OK' not in classes):
            print('There should exist a subdirectory "good". Read the doc of this function for further information.')
            exit()
        classes.sort()
        class_perm = list()
        class_idx = 1
        for cl in classes:
            if (cl == 'good') or (cl == 'OK'):
                class_perm.append(0)
            else:
                class_perm.append(class_idx)
                class_idx += 1
        trainset = SynapseData(c, data_dir_train, train=True)
        testset = SynapseData(c, data_dir_test, train=False)

    return trainset, testset

class SynapseData(Dataset):
    def __init__(self, config, data_dir, train=True, ret_path=True):
        super().__init__()
        self.config = config
        self.dir = data_dir
        self.train = train
        self.size = config.img_size[0]
        self.ret_path = ret_path
        self.mag = 0.3
        
        if self.train:
            self.files = glob.glob(self.dir+'/*/*')
            self.transforms = A.Compose(
                [
                    #A.augmentations.transforms.ColorJitter(brightness=self.mag, contrast=self.mag, saturation=self.mag, hue=self.mag, p=0.5),
                    #A.transforms.Flip(p=0.5),
                    #A.ShiftScaleRotate(shift_limit=self.mag, scale_limit=0.0, rotate_limit=self.mag, border_mode=0, value=(0,0,0),p=0.5),
                    A.Resize(height=self.size, width=self.size, always_apply=True),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
)
        else:
            self.files = glob.glob(self.dir+'/*/*')
            self.transforms = A.Compose(
                [
                    A.Resize(height=self.size, width=self.size, always_apply=True),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
)
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        self.file = self.files[idx]
        if ('OK' in self.file) or ('good' in self.file):
            self.label = 0
        else:
            self.label = 1

        self.img=Image.open(self.file)
        self.img = self.transforms(image = np.array(self.img))
        self.img = self.img['image']
        
        if not self.ret_path:
            return self.img, self.label
        else:
            return self.img, self.label, self.file
    
class FeatureDataset(Dataset):
    def __init__(self, root="data/features/" + c.class_name + '/', n_scales=c.n_scales, train=False):

        super(FeatureDataset, self).__init__()
        self.data = list()
        self.n_scales = n_scales
        self.train = train
        suffix = 'train' if train else 'test'

        for s in range(c.n_scales):
            self.data.append(np.load(root + c.class_name + '_scale_' + str(s) + '_' + suffix + '.npy'))

        self.labels = np.load(os.path.join(root, c.class_name + '_labels.npy')) if not train else np.zeros(
            [len(self.data[0])])
        self.paths = np.load(os.path.join(root, c.class_name + '_image_paths.npy'))
        #print(self.paths)
        self.class_names = 0#[img_path.split('/')[-2] for img_path in self.paths] 

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        out = list()
        for d in self.data:
            sample = d[index]
            sample = torch.FloatTensor(sample)
            out.append(sample)
        return out, self.labels[index]


def make_dataloaders(trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                              drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=1, shuffle=False,
                                             drop_last=False)
    return trainloader, testloader


def preprocess_batch(data):
    '''move data to device and reshape image'''
    flag = 0
    if c.pre_extracted:
        try:
            inputs, labels, path = data
            flag = 1
        except:
            inputs, labels = data
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(c.device)
        labels = labels.to(c.device)
    else:
        try:
            inputs, labels, path = data
            flag = 1
        except:
            inputs, labels = data
        inputs, labels = inputs.to(c.device), labels.to(c.device)
        inputs = inputs.view(-1, *inputs.shape[-3:])
    if flag == 1:
        return inputs, labels
    return inputs, labels


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.min_loss_epoch = 0
        self.min_loss_score = 0
        self.min_loss = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if self.max_score == None or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d} \t epoch_loss: {:d}'.format(self.name, self.last,
                                                                                                   self.max_score,
                                                                                                   self.max_epoch,
                                                                                                   self.min_loss_epoch))
