"""Simple webcam test. Press Q or Esc in the camera window to quit."""
import cv2


def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Close other camera apps and check Windows camera permissions.")
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break
            cv2.imshow("Webcam test | Q to quit", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
