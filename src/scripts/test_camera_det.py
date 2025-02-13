import sys
sys.path.append('./src')

import json
import numpy as np
import cv2
from PIL import ImageDraw, Image

import torch
from transformers import RTDetrImageProcessor
from inference.tensorrt_inference import TensorRTInference
from models.detr_output import DetrOutput


image_processor = RTDetrImageProcessor.from_pretrained("jadechoghari/RT-DETRv2")
image_processor.size = {'height': 480, 'width': 640}
trt_inference = TensorRTInference("model.trt")

with open('./assets/labels.txt', 'r') as f:
    labels = json.loads(f.read())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30.0)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if ret:
        input_tensor = image_processor(frame, return_tensors="pt")
        output_data = trt_inference([input_tensor['pixel_values']])
        detr_output = DetrOutput(logits = torch.from_numpy(output_data[0].host.reshape((1, -1, 80))),
                                 pred_boxes = torch.from_numpy(output_data[1].host.reshape((1, -1, 4))))
        results = image_processor.post_process_object_detection(detr_output, target_sizes=torch.tensor([frame.shape[:-1]]), threshold=0.3)

        draw_img = Image.fromarray(frame[..., ::-1])
        draw = ImageDraw.Draw(draw_img)

        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                score, label = score.item(), label.item()
                box = [round(i, 2) for i in box.tolist()]
                if score > 0.5:
                    draw.rectangle(box, fill=None, outline='red')
                    draw.text((box[0], box[1]), f"{labels[str(label)]}: {score:.2f}")

        # Display the resulting frame
        cv2.imshow('frame', np.array(draw_img)[..., ::-1])
        if cv2.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
