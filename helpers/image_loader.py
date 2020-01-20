from torchvision import datasets
import torchvision.transforms as transforms
import torch

def image_transforms(img_size):
    """
    Parmerters:
    img_size

    returns transforms and normalization
    """

    img_transforms = transforms.Compose(
                                [transforms.Resize(size=(img_size,img_size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225] )])

    return img_transforms


def images_from_dir(data_dir, img_transforms):
    """
    Parmerters:
    data_dir, img_transforms

    returns data transformed from ImageFolder(data_dir)
    """

    img_data = datasets.ImageFolder(root=data_dir,
                                  transform=img_transforms)
    return img_data


def images_from_csv():
    pass


def image_data_loader(data,batch_size,num_workers,shuffle=False):
    """
    Parmerters:
    data, batch_size, num_workers, shuffle=False

    returns data from torch DataLoader
    """
    img_loader = torch.utils.data.DataLoader(data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=shuffle)

    return img_loader

def dir_loader_stack(data_dir,img_size,batch_size,num_workers,shuffle=False):
    """
    Parmerters:
    data_dir, img_size, batch_size, num_workers, shuffle=False

    Return:
    DataLoader object of dara from data_dir path 
    """
    data = image_data_loader(
        images_from_dir(data_dir,
                        image_transforms(img_size)),
                        batch_size,
                        num_workers,
                        shuffle)
    return data
