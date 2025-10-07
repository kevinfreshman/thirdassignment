import cv2
import numpy as np
from openvino.runtime import Core

# ========== 1. 模型路径 ==========
model_xml = "yolov8_model/yolov8n_openvino_model/yolov8n.xml"

# ========== 2. 初始化 OpenVINO 推理引擎 ==========
ie = Core()
compiled_model = ie.compile_model(model=model_xml, device_name="CPU")
input_layer = compiled_model.inputs[0]
output_layer = compiled_model.outputs[0]

# ========== 3. 初始化摄像头 ==========
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开摄像头，请检查设备或权限设置。")

print("✅ 摄像头已启动，按 'q' 键退出。")

# ========== 4. 检测参数 ==========
conf_threshold = 0.25
input_size = (640, 640)

# ========== 5. 实时检测循环 ==========
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 无法读取摄像头帧。")
        break

    original_frame = frame.copy()
    h, w = frame.shape[:2]

    # ========== 图像预处理 ==========
    resized = cv2.resize(frame, input_size)
    input_image = resized.transpose((2, 0, 1))[None, ...]  # BCHW
    input_image = input_image.astype(np.float32) / 255.0

    # ========== 推理 ==========
    results = compiled_model([input_image])[output_layer]

    # ========== 后处理 ==========
    num_classes = results.shape[1] - 4
    boxes = results[0, :4, :].T
    scores = results[0, 4:, :].T
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    mask = confidences > conf_threshold

    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # 坐标映射回原图
    scale_x, scale_y = w / input_size[0], h / input_size[1]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # ========== 绘制检测框 ==========
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {cls_id} ({conf:.2f})"
        cv2.putText(original_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ========== 在窗口中显示 ==========
    cv2.imshow("YOLOv8 OpenVINO Realtime Detection", original_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== 6. 清理 ==========
cap.release()
cv2.destroyAllWindows()
print("🛑 检测结束，摄像头已关闭。")
