import cv2
import numpy as np
from openvino.runtime import Core

# ========== 1. 路径设置 ==========
model_xml = "yolov8_model/yolov8n_openvino_model/yolov8n.xml"
image_path = "test1.jpg"
output_path = "output.jpg"

# ========== 2. 初始化 OpenVINO 推理引擎 ==========
ie = Core()
compiled_model = ie.compile_model(model=model_xml, device_name="CPU")
input_layer = compiled_model.inputs[0]
output_layer = compiled_model.outputs[0]

# ========== 3. 加载并预处理图像 ==========
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

original_image = image.copy()
input_image = cv2.resize(image, (640, 640))
input_image = input_image.transpose((2, 0, 1))[None, ...]  # BCHW
input_image = input_image.astype(np.float32) / 255.0

# ========== 4. 推理 ==========
results = compiled_model([input_image])[output_layer]
print(f"Inference result shape: {results.shape}")

# ========== 5. 后处理 ==========
conf_threshold = 0.25
num_classes = results.shape[1] - 4
boxes = results[0, :4, :].T
scores = results[0, 4:, :].T

class_ids = np.argmax(scores, axis=1)
confidences = np.max(scores, axis=1)
mask = confidences > conf_threshold

boxes = boxes[mask]
class_ids = class_ids[mask]
confidences = confidences[mask]

# 将框从640x640映射回原图
h, w = original_image.shape[:2]
scale_x, scale_y = w / 640, h / 640
boxes[:, [0, 2]] *= scale_x
boxes[:, [1, 3]] *= scale_y

# ========== 6. 绘制检测框 ==========
for box, cls_id, conf in zip(boxes, class_ids, confidences):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"ID {cls_id} ({conf:.2f})"
    cv2.putText(original_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# ========== 7. 保存结果 ==========
cv2.imwrite(output_path, original_image)
print(f"✅ Detection result saved to: {output_path}")