"""
Face Mask Detector pipeline combining OpenCV Caffe SSD Face Detection + Mask Classifier Model.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.dataset import preprocess_face


class MaskDetectorPipeline:
    def __init__(
        self,
        mask_model_path: str,
        face_net_prototxt: str,
        face_net_weights: str,
        confidence_threshold: float = 0.5,
        target_size: tuple = (150, 150),
    ):
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        
        # Load Caffe face detector
        self.face_net = cv2.dnn.readNetFromCaffe(face_net_prototxt, face_net_weights)
        
        # Load mask classifier
        self.mask_model = load_model(mask_model_path)
        
        self.labels = {0: "mask", 1: "without mask"}
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green = mask (0), Red = without mask (1)

    def detect_and_predict(self, frame: np.ndarray):
        """Phát hiện khuôn mặt và phân loại khẩu trang trên từng frame.
        
        Returns:
            processed_frame, detections_list (bounding_box, label, confidence)
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.face_net.setInput(blob)
        raw_detections = self.face_net.forward()

        results = []

        for i in range(raw_detections.shape[2]):
            confidence = raw_detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue

            box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            batch = preprocess_face(face_img, target_size=self.target_size)
            predictions = self.mask_model(batch, training=False).numpy()
            
            label_idx = int(np.argmax(predictions, axis=1)[0])
            score = float(np.max(predictions)) * 100

            results.append({
                "box": (x1, y1, x2, y2),
                "label": self.labels[label_idx],
                "class_id": label_idx,
                "confidence": score
            })

            # Draw box & text
            color = self.colors[label_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), color, -1)
            cv2.putText(
                frame,
                f"{self.labels[label_idx]} ({score:.1f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        return frame, results
