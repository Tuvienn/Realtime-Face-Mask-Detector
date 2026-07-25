import argparse
import time
import cv2

from src.detector import MaskDetectorPipeline


def main():
    parser = argparse.ArgumentParser(description="Real-time Face Mask Detection via Webcam")
    parser.add_argument("--model", type=str, default="./best_model.keras", help="Path to trained mask classifier model")
    parser.add_argument("--prototxt", type=str, default="./models/deploy.prototxt", help="Path to face detector prototxt")
    parser.add_argument("--weights", type=str, default="./models/res10_300x300_ssd_iter_140000.caffemodel", help="Path to face detector weights")
    parser.add_argument("--confidence", type=float, default=0.5, help="Face detection confidence threshold")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()

    print("[INFO] Initializing Face Mask Detector Pipeline...")
    pipeline = MaskDetectorPipeline(
        mask_model_path=args.model,
        face_net_prototxt=args.prototxt,
        face_net_weights=args.weights,
        confidence_threshold=args.confidence,
    )

    print(f"[INFO] Opening webcam (device index: {args.webcam})...")
    webcam = cv2.VideoCapture(args.webcam, cv2.CAP_AVFOUNDATION)
    if not webcam.isOpened():
        webcam = cv2.VideoCapture(args.webcam)

    if not webcam.isOpened():
        raise RuntimeError(
            f"[ERROR] Cannot open webcam at index {args.webcam}.\n"
            "📌 Hướng dẫn sửa trên macOS:\n"
            " 1. Cấp quyền Camera: Mở 'System Settings' -> 'Privacy & Security' -> 'Camera' -> Bật quyền cho Terminal / VS Code / Python.\n"
            " 2. Đảm bảo không có ứng dụng khác (FaceTime, Zoom, Teams...) đang chiếm camera.\n"
            " 3. Nếu dùng camera ngoài hoặc iPhone Continuity Camera, thử tham số: python test.py --webcam 1"
        )

    print("[INFO] Starting real-time detection. Press 'q' or ESC to quit.\n")
    prev_time = time.time()

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("[WARNING] Failed to read frame from webcam — stopping.")
            break

        frame = cv2.flip(frame, 1)

        # High-level pipeline call
        frame, _ = pipeline.detect_and_predict(frame)

        # FPS Counter
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Face Mask Detector — Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    webcam.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped real-time detection.")


if __name__ == "__main__":
    main()

