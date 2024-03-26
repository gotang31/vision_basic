# What is this folder.

- I want to test whether extracted feature from DETR is well performed in image retrieval.
- So, I do crawl the products' images of laptop in Coupang website and extract the feature of each image from DETR and Resnet50.
- And, I rank the image and compare DETR with Resnet50 by eye-evaluation.

# File Directory

```
- crawling  
     |- laptop (you have to run coupang.py if you run comparison.py)
          |- ...jpg
          |- ...jpg
          |- ...jpg
          ...

- inference  
     |- laptop.csv
     |- vectorDB
          |- laptop_detr.index
          |- laptop_resnet50.index
```
# Results
![다운로드 (1)](https://github.com/gotang31/vision_basic/assets/147139248/502d18a2-2b0b-4542-894b-f10464b4a58a)
![다운로드 (2)](https://github.com/gotang31/vision_basic/assets/147139248/9b900520-a0c0-40bf-89be-3ebeb5d74f97)

