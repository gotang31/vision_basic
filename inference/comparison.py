import torchvision
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
import pandas as pd
import faiss
from PIL import Image, ImageFile
from feature_extraction import res_fe, detr_fe

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Compare:
    def __init__(self, model1, model2, process1, process2, fe1, fe2, vectorDB1, vectorDB2):
        self.model1 = model1
        self.model2 = model2
        self.process1 = process1
        self.process2 = process2
        self.fe1 = fe1
        self.fe2 = fe2
        self.vectorDB1 = vectorDB1
        self.vectorDB2 = vectorDB2

        self.device = self.set_device()
        self.initiate_model(self.model1)
        self.initiate_model(self.model2)
    
    def set_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return device

    def initiate_model(self, model):

        model.to(self.device)

        return model.eval()

    def feature_extractor(self, img_dir):
        feature1 = self.fe1(img_dir, self.model1, self.process1)
        feature2 = self.fe2(img_dir, self.model2, self.process2)

        return feature1, feature2
    
    def retrieval(self, img_dir, n): # n == the number of retrieval images
        feature1, feature2 = self.feature_extractor(img_dir)
        feature1, feature2 = feature1.cpu().detach().numpy() , feature2.cpu().detach().numpy()

        distance1, indices1 = self.vectorDB1.search(feature1.reshape(1, -1), n)
        distance2, indices2 = self.vectorDB2.search(feature2.reshape(1, -1), n)

        return indices1, indices2
    
    def visualization(self, img_dir, n, df):
        indices1, indices2 = self.retrieval(img_dir, n)

        plt.figure(figsize=(20, 8))
        self.display_img(img_dir, 'Query', 1)
        
        i = 6                                                    # i == subplot index
        i = self.display_plt(indices1, 'DETR', i, df)
        i = self.display_plt(indices2, 'ResNet50', i, df)
        
        plt.show()
        return
    
    def display_plt(self, indices, mode, i, df):
        for index, itemid in enumerate(indices.squeeze(0)):
            img_dir_ret = '../crawling/laptop/' + df.iloc[itemid, 0]
            self.display_img(img_dir_ret, f'{mode} - {index}', i)
            i += 1
        return i

    def display_img(self, img_dir, title, i):
        plt.subplot(3,5,i)
        plt.imshow(Image.open(img_dir))
        plt.title(title, fontsize = 10)
        plt.axis("off")

        return

if __name__ == "__main__":

    detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                    revision="no_timm",
                                                    num_labels = 91,
                                                    ignore_mismatched_sizes=True)
    detr_process = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    resnet50 = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
    resnet50_process = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
    detr_vectorDB = faiss.read_index('vectorDB/laptop_detr.index')
    resnet50_vectorDB = faiss.read_index('vectorDB/laptop_resnet50.index')
    df = pd.read_csv('laptop.csv')

    comparision = Compare(detr, resnet50, detr_process, 
                          resnet50_process, detr_fe, res_fe,
                          detr_vectorDB, resnet50_vectorDB)
    
    img_dir = '../crawling/laptop/' + df.sample(n=1).iloc[0, 0] # a directory of query image ; if you want to input an image directly, you must modify feature extraction function code partially.

    comparision.visualization(img_dir, 5, df)