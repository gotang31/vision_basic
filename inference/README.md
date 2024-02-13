# What is this folder.

- I want to test whether extracted feature from DETR is well performed in image retrieval.
- So, I do crawling the products image of laptop in Coupang site and extract the feature of each image from DETR and Resnet50.
- And, I rank the image and compare DETR with Resnet50 by eye-evaluation.

# File Directory

```
- crawling  
     |- laptop (you have to run coupang.py if you run compare.py)
          |- ...jpg
          |- ...jpg
          |- ...jpg
          ...

- inference  
     |- laptop.csv
```
  
