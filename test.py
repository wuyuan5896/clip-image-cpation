# import torch
# print(torch.cuda.is_available())
# device = torch.device("cuda:{}".format('3') if torch.cuda.is_available() else "cpu")
# print("my device is {}".format(torch.cuda.current_device()))
import json

path = "/server24/rsh/clip-image-cpation/data/coco/annotations/train_caption.json"

with open(path,'r',encoding='utf-8') as f:
    data = json.load(f)

sample_data = data[:5]

for key,value in enumerate(sample_data):
    print(key,value)