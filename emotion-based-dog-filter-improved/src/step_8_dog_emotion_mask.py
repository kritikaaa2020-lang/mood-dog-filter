
"""Real-time JAFFE emotion-based animal filter. Press Q or Esc to quit."""

from collections import Counter, deque
from pathlib import Path
import sys
import cv2

from jaffe_predictor import get_jaffe_predictor

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
MODEL_PATH = HERE / "models" / "jaffe_emotion_model.pth"

EMOTION_NAMES = ["Happy", "Sad", "Surprise"]


def apply_mask(face, mask):
    """Fit a white-background mask into the detected face region."""

    if face is None or face.size == 0 or mask is None:
        return face

    fh, fw = face.shape[:2]
    mh, mw = mask.shape[:2]

    if min(fh, fw, mh, mw) <= 0:
        return face

    scale = min(fh / mh, fw / mw)

    nw = max(1, int(mw * scale))
    nh = max(1, int(mh * scale))

    resized = cv2.resize(
        mask,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    result = face.copy()

    ox = (fw - nw) // 2
    oy = (fh - nh) // 2

    roi = result[oy:oy + nh, ox:ox + nw]

    visible = (resized < 245).any(axis=2)
    roi[visible] = resized[visible]

    return result


def open_camera():
    """Open webcam using Windows DirectShow if available."""

    if sys.platform == "win32":
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if camera.isOpened():
            return camera

        camera.release()

    return cv2.VideoCapture(0)


def main():

    camera = open_camera()

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Happy = Brown dog
    # Sad = Dalmatian
    # Surprise = Sheepdog

    mask_files = [
        "dog.png",
        "dalmation.png",
        "sheepdog.png"
    ]

    masks = [
        cv2.imread(str(ASSETS / name))
        for name in mask_files
    ]

    if any(mask is None for mask in masks):

        missing = [
            name
            for name, mask in zip(mask_files, masks)
            if mask is None
        ]

        print("ERROR: Missing mask images:", missing)
        camera.release()
        return

    cascade = cv2.CascadeClassifier(
        str(ASSETS / "haarcascade_frontalface_default.xml")
    )

    if cascade.empty():
        print("ERROR: Could not load face detection XML.")
        camera.release()
        return

    if not MODEL_PATH.exists():
        print("ERROR: JAFFE model not found:", MODEL_PATH)
        camera.release()
        return

    print("Loading JAFFE emotion model...")

    try:
        predictor = get_jaffe_predictor(str(MODEL_PATH))

    except Exception as exc:
        print("ERROR while loading emotion model:", exc)
        camera.release()
        return

    print("JAFFE model loaded successfully!")
    print("Filter is running. Press Q or Esc to quit.")

    frame_number = 0
    face_box = None
    previous_box = None
    missed_detections = 0

    emotion_votes = deque(maxlen=7)
    emotion_index = 0

    detect_every = 3
    predict_every = 9
    smoothing = 0.65

    last_printed_emotion = None

    try:

        while True:

            ok, frame = camera.read()

            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break

            frame_number += 1
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # Detect face every 3 frames.
            if frame_number % detect_every == 0:

                small = cv2.resize(
                    frame,
                    None,
                    fx=0.5,
                    fy=0.5,
                    interpolation=cv2.INTER_AREA
                )

                gray = cv2.cvtColor(
                    small,
                    cv2.COLOR_BGR2GRAY
                )

                gray = cv2.equalizeHist(gray)

                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=5,
                    minSize=(35, 35),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                if len(faces):

                    x, y, w, h = max(
                        faces,
                        key=lambda box: box[2] * box[3]
                    )

                    detected = (
                        int(x * 2),
                        int(y * 2),
                        int(w * 2),
                        int(h * 2)
                    )

                    if previous_box is None:
                        face_box = detected

                    else:
                        face_box = tuple(
                            int(
                                smoothing * old
                                + (1 - smoothing) * new
                            )
                            for old, new in zip(
                                previous_box,
                                detected
                            )
                        )

                    previous_box = face_box
                    missed_detections = 0

                else:

                    missed_detections += 1

                    if missed_detections > 5:
                        face_box = None
                        previous_box = None
                        emotion_votes.clear()
                        last_printed_emotion = None

            if face_box is not None:

                x, y, w, h = face_box

                x0 = max(0, x)
                y0 = max(0, y - int(0.25 * h))

                x1 = min(width, x + w)
                y1 = min(height, y + int(0.75 * h))

                fx0 = max(0, x)
                fy0 = max(0, y)

                fx1 = min(width, x + w)
                fy1 = min(height, y + h)

                # Predict emotion every 9 frames.
                if (
                    frame_number % predict_every == 0
                    and fx1 > fx0
                    and fy1 > fy0
                ):

                    crop = frame[fy0:fy1, fx0:fx1]

                    try:

                        prediction = predictor(crop)

                        emotion_votes.append(prediction)

                        emotion_index = Counter(
                            emotion_votes
                        ).most_common(1)[0][0]

                        # Print emotion when prediction changes.
                        if emotion_index != last_printed_emotion:

                            print(
                                "Predicted emotion:",
                                EMOTION_NAMES[emotion_index]
                            )

                            last_printed_emotion = emotion_index

                    except Exception as exc:

                        print(
                            "Emotion prediction warning:",
                            exc
                        )

                if x1 > x0 and y1 > y0:

                    emotion_index = max(
                        0,
                        min(2, int(emotion_index))
                    )

                    # Apply the corresponding animal mask.
                    frame[y0:y1, x0:x1] = apply_mask(
                        frame[y0:y1, x0:x1],
                        masks[emotion_index]
                    )

                    # Display emotion above the face.
                    cv2.putText(
                        frame,
                        EMOTION_NAMES[emotion_index],
                        (x0, max(30, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow(
                "Emotion-Based Dog Filter | Q to quit",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
