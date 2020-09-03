import os
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, dataset_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_dataset in sorted(dataset_to_idx.keys()):
        dataset_index = dataset_to_idx[target_dataset]
        imgs_target_dir = os.path.join(directory, target_dataset, 'imgs')
        gts_target_dir = os.path.join(directory, target_dataset, 'gts')
        if not os.path.isdir(imgs_target_dir) or not os.path.isdir(gts_target_dir):
            continue
        (imgs_root, _imgs_fnames) = [(i, k) for i, _, k in os.walk(imgs_target_dir, followlinks=True)][0]
        (gts_root, _gts_fnames) = [(i, k) for i, _, k in os.walk(gts_target_dir, followlinks=True)][0]
        imgs_fnames = sorted([x for x in _imgs_fnames if is_valid_file(x)])
        gts_fnames = sorted([x for x in _gts_fnames if is_valid_file(x)])
        if len(imgs_fnames) != len(gts_fnames):
            raise Exception('Number of images and ground truths are not equal. Number of; images: {}, ground turths: {}'.format(len(imgs_fnames), len(gts_fnames)))
        for fnames in zip(imgs_fnames, gts_fnames):
            img, gt = fnames
            img_name = img[:img.find('.')]
            gt_name = gt[:gt.find('.')]
            if img_name != gt_name:
                raise Exception('Image and ground truth filenames do not match. Filename; image: {}, ground turth: {}'.format(img, gt))

            img_path = os.path.join(imgs_root, img)
            gt_path = os.path.join(gts_root, gt)
            if is_valid_file(img_path) and is_valid_file(gt_path):
                item = img_path, gt_path, target_dataset, img_name
                instances.append(item)

    return instances

def make_dataset_eval(directory, preds_directory, dataset_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_dataset in sorted(dataset_to_idx.keys()):
        dataset_index = dataset_to_idx[target_dataset]
        gts_target_dir = os.path.join(directory, target_dataset, 'gts')
        preds_target_dir = os.path.join(preds_directory, target_dataset)

        if not os.path.isdir(preds_target_dir) or not os.path.isdir(gts_target_dir):
            continue
        (preds_root, _preds_fnames) = [(i, k) for i, _, k in os.walk(preds_target_dir, followlinks=True)][0]
        (gts_root, _gts_fnames) = [(i, k) for i, _, k in os.walk(gts_target_dir, followlinks=True)][0]
        preds_fnames = sorted([x for x in _preds_fnames if is_valid_file(x)])
        gts_fnames = sorted([x for x in _gts_fnames if is_valid_file(x)])
        if len(preds_fnames) != len(gts_fnames):
            raise Exception('Number of images and ground truths are not equal. Number of; images: {}, ground turths: {}'.format(len(preds_fnames), len(gts_fnames)))
        for fnames in zip(preds_fnames, gts_fnames):
            pred, gt = fnames
            pred_name = pred[:pred.find('.')]
            gt_name = gt[:gt.find('.')]
            if pred_name != gt_name:
                raise Exception('Image and ground truth filenames do not match. Filename; image: {}, ground turth: {}'.format(pred, gt))

            pred_path = os.path.join(preds_root, pred)
            gt_path = os.path.join(gts_root, gt)
            if is_valid_file(pred_path) and is_valid_file(gt_path):
                item = pred_path, gt_path, target_dataset, gt_name
                instances.append(item)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, pred_root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, crop=False):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        datasets, dataset_to_idx = self._find_datasets(self.root)
        if pred_root is None:
            samples = make_dataset(self.root, dataset_to_idx, extensions, is_valid_file)
        else:
            samples = make_dataset_eval(self.root, pred_root, dataset_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.crop = crop
        self.loader = loader
        self.extensions = extensions

        self.datasets = datasets
        self.dataset_to_idx = dataset_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_datasets(self, dir):
        """
        Finds the individual dataset in a of folders of datasets.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (datasets, dataset_to_idx) where datasets are relative to (dir), and dataset_to_idx is a dictionary.
        Ensures:
            No dataset is a subdirectory of another.
        """
        datasets = [d.name for d in os.scandir(dir) if d.is_dir()]
        datasets.sort()
        dataset_to_idx = {ds_name: i for i, ds_name in enumerate(datasets)}
        return datasets, dataset_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is dataset_index of the target dataset.
        """
        img_path, gt_path, dataset, img_name = self.samples[index]
        sample = self.loader(img_path)
        orig_img = sample
        target = self.loader(gt_path)
        img_res = tuple(sample.size)

        if self.crop:
            res = 402
            sample = transforms.Resize((res, res))(sample)
            orig_img = transforms.Resize((res, res))(orig_img)
            target = transforms.Resize((res, res))(target)

            w, h = (res, res)
            th, tw = (352, 352)
            i = torch.randint(0, h - th + 1, size=(1,)).item()
            j = torch.randint(0, w - tw + 1, size=(1,)).item()

            sample = F.crop(sample, i, j, th, tw)
            orig_img = F.crop(orig_img, i, j, th, tw)
            target = F.crop(target, i, j, th, tw)

            if torch.rand(1) < 0.5:
                sample = F.hflip(sample)
                orig_img = F.hflip(orig_img)
                target = F.hflip(target)

            sample = transforms.ToTensor()(sample)
            orig_img = transforms.ToTensor()(orig_img)
            target = transforms.ToTensor()(target)

            sample = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(sample)

        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
                orig_img = self.target_transform(orig_img)

        return sample, target, dataset, img_name, img_res, orig_img

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if 'imgs' in path:
            return img.convert('RGB')
        else:
            return img.convert('L')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageGroundTruthFolder(DatasetFolder):
    """A data loader for images and their ground truths from multiple datasets where the images are arranged in this way: ::
        root/<dataset>/imgs/xxx.png
        root/<dataset>/gts/xxx.png
        root/DUTS/imgs/abc.png
        root/DUTS/gtss/abc.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        datasets (list): List of the dataset names sorted alphabetically.
        dataset_to_idx (dict): Dict with items (dataset_name, dataset_index).
        imgs (list): List of (image path, dataset_index) tuples
    """

    def __init__(self, root, pred_root=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, crop=False):
        super(ImageGroundTruthFolder, self).__init__(root, pred_root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          crop=crop)
        self.imgs = self.samples

class EvalImageGroundTruthFolder(DatasetFolder):
    """A data loader for images and their ground truths from multiple datasets where the images are arranged in this way: ::
        root/<dataset>/imgs/xxx.png
        root/<dataset>/gts/xxx.png
        root/DUTS/imgs/abc.png
        root/DUTS/gtss/abc.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        datasets (list): List of the dataset names sorted alphabetically.
        dataset_to_idx (dict): Dict with items (dataset_name, dataset_index).
        imgs (list): List of (image path, dataset_index) tuples
    """

    def __init__(self, root, pred_root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(EvalImageGroundTruthFolder, self).__init__(root, pred_root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
