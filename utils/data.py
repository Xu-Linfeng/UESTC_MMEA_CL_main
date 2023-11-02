import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.mydataset import MyDataSet
from transforms import ArrayToTensor, DataStack, GroupNormalize, IdentityTransform, ImgStack, ToTorchFormatTensor, GroupScale, GroupCenterCrop

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iMydataset(iData):
    use_path = False
    class_order = [
        26,
        14,
        23,
        4,
        11,
        25,
        31,
        10,
        29,
        5,
        6,
        9,
        17,
        22,
        2,
        19,
        13,
        1,
        21,
        16,
        8,
        3,
        27,
        28,
        15,
        30,
        0,
        7,
        12,
        18,
        20,
        24
    ]

    def __init__(self, model, modality, arch, train_list, test_list):
        self.modality = modality
        self.arch = arch
        self.train_list = train_list
        self.test_list = test_list
        
        self.crop_size = model.feature_extract_network.crop_size
        self.scale_size = model.feature_extract_network.scale_size
        self.input_mean = model.feature_extract_network.input_mean
        self.input_std = model.feature_extract_network.input_std
        self.data_length = model.feature_extract_network.new_length
        self.train_augmentation = model.feature_extract_network.get_augmentation()

        self.train_trsf = {}
        self.test_trsf = {}
        self.normalize = {}

    def download_data(self):
        
        for m in self.modality:
            if (m != 'RGBDiff'):
                self.normalize[m] = GroupNormalize(self.input_mean[m], self.input_std[m])
            else:
                self.normalize[m] = IdentityTransform()

        for m in self.modality:
            if (m != 'Gyro' and m != 'Acce'):
                # Prepare train/val dictionaries containing the transformations
                # (augmentation+normalization)
                # for each modality
                self.train_trsf[m] = transforms.Compose([
                self.train_augmentation[m],
                ImgStack(roll=self.arch == 'BNInception'),
                ToTorchFormatTensor(div=self.arch != 'BNInception'),
                self.normalize[m],
                ])

                self.test_trsf[m] = transforms.Compose([
                    GroupScale(int(self.scale_size[m])),
                    GroupCenterCrop(self.crop_size[m]),
                    ImgStack(roll=self.arch == 'BNInception'),
                    ToTorchFormatTensor(div=self.arch != 'BNInception'),
                    self.normalize[m],
                ])
            else:
                self.train_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                    self.normalize[m],
                ])

                self.test_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                    self.normalize[m],
                ])
            

        train_set = MyDataSet(self.train_list)
        test_set = MyDataSet(self.test_list)

        self.train_data, self.test_data = np.array(train_set.video_list), np.array(test_set.video_list)
        self.train_targets, self.test_targets = np.array(self._get_targets(train_set)), np.array(self._get_targets(test_set))

    def _get_targets(self, dataset):
        """
        get target list from MyDataset
        """
        targets = []
        for i in range(len(dataset)):
            targets.append(dataset.video_list[i].label)

        return targets