
import argparse
import cv2
from pynput.keyboard import Controller

from hand_tracking import HandDetector
from keyboard_engine import KeyboardEngine


def _parse_camera_source(raw: str):
    """Return an ``int`` if *raw* looks like a camera index, else the raw URL."""
    return int(raw) if raw.isdigit() else raw


def main():
    parser = argparse.ArgumentParser(
        description="Virtual Keyboard — type using hand gestures in mid-air."
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index (0, 1, …) or an IP-camera URL.",
    )
    args = parser.parse_args()

    cam_source = _parse_camera_source(args.camera)
    print(f"[INFO] Opening camera: {cam_source}")

    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = HandDetector(detection_confidence=0.8)
    kb_ctrl = Controller()
    engine = KeyboardEngine(kb_ctrl)

    print("[INFO] Virtual Keyboard started.  Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame — is the camera connected?")
            break

        # 1. Detect hands and landmarks
        frame = detector.find_hands(frame)
        lm_list, _ = detector.find_position(frame)

        # 2. Render on-screen keyboard
        frame = engine.draw(frame)

        # 3. Check for pinch-over-key events
        frame = engine.check_touches(lm_list, frame)

        # 4. Display
        cv2.imshow("Virtual Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
