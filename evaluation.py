from predict import Predictor 
import os
import json
from tqdm import tqdm
def get_image_id(image_path):
    """从图像路径提取 image_id，假设图像文件名格式为 'COCO_val2014_000000000000.jpg'."""
    filename = os.path.basename(image_path)  # 获取文件名
    image_id = int(filename.split('_')[-1].split('.')[0])  # 提取 image_id
    return image_id

def generate_val_json(val_dir, model_name="coco", use_beam_search=False, output_file="val.json"):

    predictor = Predictor()
    predictor.setup()
    print('>>>>>>Start evaluating<<<<<')
    results = []
    progress = tqdm(total=len(os.listdir(val_dir)), desc="Generating val:",position=0)
    step = 0
    for image_filename in os.listdir(val_dir):
        image_path = os.path.join(val_dir, image_filename)
        
        image_id = get_image_id(image_path)
    
        caption = predictor.predict(image=image_path, model=model_name, use_beam_search=use_beam_search)
        
        result = {
            "image_id": image_id,
            "caption": caption
        }
        results.append(result)
        step += 1
        progress.set_postfix_str('num:{}/{}'.format(step, len(os.listdir(val_dir))))
        progress.update()
        
    # 将结果保存到json文件
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"结果已保存到 {output_file}")

# 使用示例
val_dir = "./data/coco/val2014"  # 你验证集图片的目录
generate_val_json(val_dir, model_name="coco", use_beam_search=False, output_file="val.json")
