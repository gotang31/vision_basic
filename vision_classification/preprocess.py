import json
import pandas as pd

# coco detection의 annotations을 json 파일
# detection 학습의 경우 Cocodetection 데이터셋을 상속받은 커스텀 데이터 클래스로 학습을 진행할 수 있지만, 이는 분류기 학습이므로 따로 데이터 전처리 진행
# ms-coco에서 다운받는 annotations 파일이 아닌 직접 라벨링하여 만든 coco 형식의 annotations의 경우(roboflow 등의 툴 이용), 아래의 코드를 조금 수정해야한다.

def train_coco_dataset():
    with open('annotations/instances_train2017.json', 'r') as fp:
        d = json.load(fp)

    images = pd.DataFrame(d['images']).drop(labels = ['license', 'coco_url',  'date_captured', 'flickr_url'], axis = 1)
    images.rename(columns = {'file_name': 'ImageID'}, inplace = True)
    annotations = pd.DataFrame(d['annotations']).drop(labels = ['id', 'area','iscrowd', 'segmentation'], axis = 1)

    for i in annotations['bbox']:
        i[2] = i[0] + i[2]
        i[3] = i[1] + i[3]

    x = [list(map(lambda x: x[i], annotations.iloc[:, 1])) for i in range(4)]
    annotations = pd.DataFrame({'id': annotations['image_id'], 'label' : annotations['category_id'],'x1': x[0], 'y1': x[1], 'x2': x[2], 'y2': x[3]})

    res = images.set_index('id').join(annotations.set_index('id'), how = 'inner').reset_index()

    # coco 클래스 별 개수 조정: 중위값인 6000개보다 큰 클래스의 경우 6000개로 맞추고, 아닌 경우는 그대로 개수 유지
    value = res.label.value_counts().sort_index() 
    index_arr = value[value > 6000].index

    rst = pd.DataFrame()
    for i in index_arr:
        temp = res[res.label == i].sample(n = 6000)
        rst = pd.concat([rst, temp])

    res_data = pd.concat([rst, res[~res.label.isin(index_arr)]])
    res_data.index = list(range(len(res_data)))
    res_data.to_csv('train2017.csv', index = False)
    
    return res_data

def val_coco_dataset():
    with open('annotations/instances_val2017.json', 'r') as fp:
        d = json.load(fp)

    images = pd.DataFrame(d['images']).drop(labels = ['license', 'coco_url',  'date_captured', 'flickr_url'], axis = 1)
    images.rename(columns = {'file_name': 'ImageID'}, inplace = True)
    annotations = pd.DataFrame(d['annotations']).drop(labels = ['id', 'area','iscrowd', 'segmentation'], axis = 1)

    for i in annotations['bbox']:
        i[2] = i[0] + i[2]
        i[3] = i[1] + i[3]

    x = [list(map(lambda x: x[i], annotations.iloc[:, 1])) for i in range(4)]
    annotations = pd.DataFrame({'id': annotations['image_id'], 'label' : annotations['category_id'],'x1': x[0], 'y1': x[1], 'x2': x[2], 'y2': x[3]})

    res = images.set_index('id').join(annotations.set_index('id'), how = 'inner').reset_index()
    res.to_csv('val2017.csv', index = False)

    return res

if __name__ == "__main__":
    train_coco_dataset()
    val_coco_dataset()