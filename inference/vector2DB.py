import os
import torchvision
import torch
import pandas as pd
from transformers import DetrForObjectDetection, DetrImageProcessor
import faiss
from feature_extraction import res_fe, detr_fe

class SimBck: 
    def __init__(self,
                model,
                process,
                dir = False):
        '''
        dir : file directory of fine-tuned model state_dict. Default is pre-trained model
        model : pretrained model
        '''
        self.model = model
        self.process = process
        self.device = self.set_device()

        if dir:
            self.model.load_state_dict(torch.load(dir))
        else:
            pass

    def set_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return device

    def initiate_model(self): 
        self.model.to(self.device)

        return self.model.eval()

class VecDB(SimBck):
    def __init__(self, model, process, fe, dir, file_name):
        '''
        model : inference model
        process : image transform process for input model
        fe : feacture extractor function
        dir : a directory to save a vector DB
        '''    
        super(VecDB, self).__init__(model, process)
        self.dir = dir
        self.fe = fe
        self.file_name = file_name
        self.initiate_model()

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
    def vector2_faiss_L2(self, df):
        print(f'{self.file_name} : start the feature extraction and build vectorDB')
        index = faiss.IndexFlatL2(2048)
        index = faiss.IndexIDMap2(index)

        i = 0
        for img_dir in df.dir.values:
            itemid = df[df.dir == img_dir].index.values
            img_dir = '../crawling/laptop/' + img_dir
            
            if i % 500 == 0:
                faiss.write_index(index, f'{self.dir}/laptop_{self.file_name}.index')
            
            i += 1

            try:
                img_embedding= self.fe(img_dir, self.model, self.process)
                img_embedding = img_embedding.cpu().detach().numpy()
                index.add_with_ids(img_embedding, itemid) # index의 vetor space에 vector projection with itemid
                print(itemid, '성공')

            except:
                print(itemid, '실패')
                continue
          
        faiss.write_index(index, f'{self.dir}/laptop_{self.file_name}.index')
        print(f'contruction of {self.file_name} vectorDB is completed')

if __name__== "__main__":
    detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                    revision="no_timm",
                                                    num_labels = 91,
                                                    ignore_mismatched_sizes=True)
    detr_process = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    resnet50 = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
    resnet50_process = torchvision.models.ResNet50_Weights.DEFAULT.transforms()

    DETR = VecDB(detr, detr_process, detr_fe, 'vectorDB', 'detr')
    RESNET50 = VecDB(resnet50, resnet50_process, res_fe, 'vectorDB', 'resnet50')

    df = pd.read_csv('laptop.csv')

    DETR.vector2_faiss_L2(df)
    RESNET50.vector2_faiss_L2(df)


