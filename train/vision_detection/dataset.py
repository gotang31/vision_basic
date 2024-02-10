import torchvision

class DetrData(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        annotation_file_path: str,
        image_processor):

        super(DetrData, self).__init__(image_directory_path, annotation_file_path) # image directory와 annotation json file directory
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(DetrData, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target # pixel_values: image_processor에서 처리한 이미지 결과, target: image_processor에서 처리한 라벨 
