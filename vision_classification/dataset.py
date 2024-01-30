import zipfile
import io
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

class ResData(torch.utils.data.Dataset):
    def __init__(self, df, indices, phase): # df = rst
        self.df = df
        self.indices = indices
        self.phase = phase

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # index 접근 후 image name
        info = self.df[self.df.index == idx]

        # 해당 이미지의 path, boxes 정보 가져오기
        img = info.ImageID.iloc[0]
        boxes = info.iloc[:,-4:]

        # 이미지, 라벨 정보
        img = self.load_img(img, self.phase)
        label = info.values[:, -5].astype('int64')
        label = torch.tensor(label, dtype = torch.int64)

        # 이미지 crop/resize
        cropped_img = self.crop_image(img, boxes)
        preprocess = self.assign_transform()
        img = preprocess(cropped_img)

        return (img, label)

    def assign_transform(self): 
        # Augmentation : Flip/Color/RandomResize/Traslation etc..
        # preprocess = T.Compose([T.RandomHorizontalFlip(p = 0.5),
        #                         T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        #                         T.ToTensor(),
        #                         T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        preprocess = T.Compose([T.Resize((224,224)),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        return preprocess

    def crop_image(self, img, boxes):
        '''
        img : PIL.Image.open
        boxes : box coordinate pandas DataFrame
        '''
        bbox = boxes.values
        left, top, right, bottom = bbox.squeeze(0).astype('int32')
        height = bottom - top
        width = right - left
        cropped_image = T.functional.crop(img, int(top), int(left), int(height),int(width))

        return cropped_image

    def load_img(self, img, phase):
        with zipfile.ZipFile(f'{phase}2017.zip', 'r') as zp:
            ioimg = zp.read(f'{phase}2017/{img}')

        img = Image.open(io.BytesIO(ioimg)).convert('RGB')

        return img

def collate_fn(batch):
    imgs = np.array(list(map(lambda x : x[0].numpy(), batch)))
    labels = np.array(list(map(lambda x : x[1].numpy(), batch))).squeeze(1)
    imgs, labels = torch.tensor(imgs), torch.tensor(labels)
    
    return imgs, labels