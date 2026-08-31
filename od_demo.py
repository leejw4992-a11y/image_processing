"""TF Object Detection API 설치 확인용 데모.

models/research 의 object_detection 패키지 + 컴파일된 protos 가 정상인지
사전학습 SSD MobileNet V2 로 dog.png 를 추론해서 확인한다.

실행: python od_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

LABEL_MAP = "models/research/object_detection/data/mscoco_label_map.pbtxt"
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
IMAGE = "dog.png"

category_index = label_map_util.create_category_index_from_labelmap(
    LABEL_MAP, use_display_name=True)

detector = hub.load(MODEL_URL)

image_np = np.array(Image.open(IMAGE).convert("RGB"))
results = detector(tf.convert_to_tensor(image_np[tf.newaxis, ...], dtype=tf.uint8))
results = {k: v.numpy() for k, v in results.items()}

image_with_boxes = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_with_boxes,
    results["detection_boxes"][0],
    results["detection_classes"][0].astype(int),
    results["detection_scores"][0],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    min_score_thresh=0.3,
    line_thickness=3)

top = results["detection_scores"][0] > 0.3
for cls, score in zip(results["detection_classes"][0][top].astype(int),
                      results["detection_scores"][0][top]):
    print(f"{category_index[cls]['name']}: {score:.2f}")

plt.figure(figsize=(10, 8))
plt.imshow(image_with_boxes)
plt.axis("off")
plt.savefig("od_demo_result.png", bbox_inches="tight")
print("saved -> od_demo_result.png")
