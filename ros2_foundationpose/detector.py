import torch
# import torchvision
import numpy as np
import cv2
import PIL
import os
import sys
import supervision as sv
import torchvision
from ros2_foundationpose.model_config import GroundingDINOConfig
from ros2_foundationpose.utils_vision import overlay_davis

class GroundingDINOModel():
    def __init__(self,
                 model_type="swint", # swint swinb 
                 device="cuda:0",
                 box_threshold=0.3,
                 text_threshold=0.25):
        
        self.gd_config = GroundingDINOConfig.from_args(device=device)
        self.gd_predictor = self.gd_config.get_predictor()
        self.box_threshold= box_threshold
        self.text_threshold = text_threshold
        
        self.nms_threshold = 0.5
        
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def set_model(self,
                  box_threshold=0.3,
                  text_threshold=0.25):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    @torch.inference_mode()
    def run_grounding_caption(self,
                              image,
                              caption):
        
        detections, labels = self.gd_predictor.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.nms_threshold,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        # detections.class_id = detections.class_id[nms_idx]
        labels = [labels[idx] for idx in nms_idx]

        # 创建一个标签到ID的映射，ID是根据标签出现的顺序生成的
        label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}

        # 根据labels更新class_id
        class_ids = [label_to_id[label] for label in labels]
        detections.class_id = np.array(class_ids)
        return detections, labels
        
    @torch.inference_mode()
    def run_grounding_classes(self,
                              image,
                              classes):
        
        detections = self.gd_predictor.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # nms_idx = (
        #     torchvision.ops.nms(
        #         torch.from_numpy(detections.xyxy),
        #         torch.from_numpy(detections.confidence),
        #         self.nms_threshold,
        #     )
        #     .numpy()
        #     .tolist()
        # )

        # detections.xyxy = detections.xyxy[nms_idx]
        # detections.confidence = detections.confidence[nms_idx]
        # detections.class_id = detections.class_id[nms_idx]
        
        return detections
    
    def get_detection_image_classes(self,
                            image,
                            detections,
                            classes):
        
        labels = [classes[cls_id] for cls_id in detections.class_id]
        scores = detections.confidence.tolist()
        labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]

        visualization = self.box_annotator.annotate(scene=image, detections=detections)
        visualization = self.label_annotator.annotate(
            scene=visualization, detections=detections, labels=labels_with_scores
        )
        return visualization
    
    def get_detection_image_labels(self,
                            image,
                            detections,
                            labels):
        
        scores = detections.confidence.tolist()
        labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]

        visualization = self.box_annotator.annotate(scene=image, detections=detections)
        visualization = self.label_annotator.annotate(
            scene=visualization, detections=detections, labels=labels_with_scores
        )
        return visualization
        
if __name__ == "__main__":
    import time
    import supervision as sv
    
    ts = time.time()
    detector = Detector("cuda:0")
    print(f'time for loading groundingdino: {time.time()-ts}')
    
    box_annotator = sv.BoxAnnotator()
    
    image = cv2.imread('./data/des.png')
    # origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    grounding_caption = "Mustard Bottle. Bowl"
    
    detections, labels = detector.run_grounding_caption(
        image=image,
        caption=grounding_caption,
    )
    
    annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
    cv2.imshow("annotated_image", annotated_image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()