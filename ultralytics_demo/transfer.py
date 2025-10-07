import json
import os
import shutil

# 类别映射
category_map = {
    "bus": 0,
    "car": 1,
    "motorbike": 2,
    "threewheel": 3,
    "truck": 4,
    "van": 5,
}

def convert_json_to_yolo(json_file, img_dir, output_img_dir, output_label_dir,name):
    with open(json_file, 'r') as f:
        data = json.load(f)

    img_name = name[:-5]
    print(img_name)
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        print(f"图像文件不存在：{img_path}")
        return

    # 复制图像到输出目录
    shutil.copy(img_path, os.path.join(output_img_dir, img_name + '.jpg'))

    # 获取图像尺寸
    img_width = data['size']['height']
    img_height = data['size']['width']

    # 创建标签文件
    label_file_path = os.path.join(output_label_dir, img_name + '.txt')
    with open(label_file_path, 'w') as label_file:
        for obj in data['objects']:
            if obj['geometryType'] != 'rectangle':
                continue

            # 获取标注信息
            points = obj['points']['exterior']
            x_min, y_min = points[0]
            x_max, y_max = points[1]

            # 获取类别信息
            category_name = obj['classTitle']
            category_id = category_map.get(category_name)

            if category_id is None:
                print(f"未知类别: {category_name}")
                continue

            # 计算 YOLO 格式的标注
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 写入标签文件
            label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

def process_directory(json_dir, img_dir, output_img_dir, output_label_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            convert_json_to_yolo(
                os.path.join(json_dir, json_file),
                img_dir,
                output_img_dir,
                output_label_dir,
                json_file
            )

if __name__ == '__main__':
    json_dir = '/home/kevinfreshman/Downloads/valid/ann'
    img_dir = '/home/kevinfreshman/Downloads/valid/img'
    output_img_dir = '/home/kevinfreshman/desktop/ultralytics_demo/dataset/images/val'
    output_label_dir = '/home/kevinfreshman/desktop/ultralytics_demo/dataset/labels/val'

    process_directory(json_dir, img_dir, output_img_dir, output_label_dir)
