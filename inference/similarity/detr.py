from transformers import DetrForObjectDetection, DetrImageProcessor
import sys
from PIL import Image
import torch

# In instance-level image retrieval, there are two stage mode like single and mulit-stage.
# We use detection model like DETR and extract a feature at the specific layer.
# DETR model is composed of Resnet backbone and vision transformer model.
# So, I think we can extract the proper feature at the layers which are at the end of backbone, and in the middle of encoder or decoder repeating block.
# In my code, I extract the feature vector (1, 100, 2048) at the end of last decoder layer because that dimension vector is used to classify thr classes and predict the bboxes.

def feature_extractor(model, input):
    embed_feature = []
    DetrModel = list(model.named_children())[-3][1]
    DetrDecoder = list(DetrModel.children())[-1]
    LastDecoder = list(DetrDecoder.children())[-2][-1]
    hook = LastDecoder.fc1.register_forward_hook(
        lambda self, input, output: embed_feature.append(output))
    res = model(input)
    hook.remove()

    return embed_feature[0], res

def detr_fe(img_dir, model, process):
    query_img = Image.open(img_dir).convert('RGB')
    input_img = torch.tensor(process(query_img)['pixel_values'][0])
    features, outputs = feature_extractor(model, input_img.unsqueeze(0))
    
    width, height = query_img.size
    postprocessed_outputs = process.post_process_object_detection(outputs, target_sizes=[(height, width)], threshold=0.0)
    idx = torch.argmax(postprocessed_outputs[0]['scores'], dim = 0)

    return features[:, idx] # (1, 2048)

if __name__ == "__main__":
 
    img_dir = sys.argv[1]   

    process = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                    revision="no_timm",
                                                    num_labels = 91,
                                                    ignore_mismatched_sizes=True)
    detr_fe(img_dir, model, process)