
import cv2
import mediapipe as mp
import math
from typing import List, Tuple, Optional


class HandDetector:
    """Detect and track hands via MediaPipe, exposing landmark positions."""

    # Landmark indices for the five fingertips
    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ):
        """
        Parameters
        ----------
        mode : bool
            If ``True`` every frame is treated as a static image (slower but
            more robust when hands move erratically).
        max_hands : int
            Maximum number of hands to detect simultaneously.
        detection_confidence : float
            Minimum confidence for the palm detector (0-1).
        tracking_confidence : float
            Minimum confidence for the landmark tracker (0-1).
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self._mp_draw = mp.solutions.drawing_utils

        # Populated after each call to ``find_hands``
        self.results = None
        self.landmark_list: List[List[int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_hands(self, img, draw: bool = True):
        """Run hand detection on *img* and optionally draw skeleton overlay.

        Returns the (possibly annotated) image.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self._hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self._mp_draw.draw_landmarks(
                        img, hand_lms, self._mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(
        self, img, hand_index: int = 0, draw: bool = True
    ) -> Tuple[List[List], Optional[Tuple[int, int, int, int]]]:
        """Return landmark list and bounding-box for a single detected hand.

        Parameters
        ----------
        hand_index : int
            Which hand to query when multiple are detected (0-based).
        draw : bool
            Draw circles at each landmark on *img*.

        Returns
        -------
        lm_list : list of [id, x, y, z]
            21 hand landmarks.  ``z`` is the *relative* depth from MediaPipe
            (not metric depth – see MediaPipe docs).
        bbox : tuple (xmin, ymin, xmax, ymax) or None
        """
        x_vals: List[int] = []
        y_vals: List[int] = []
        self.landmark_list = []
        bbox = None

        if (
            self.results
            and self.results.multi_hand_landmarks
            and hand_index < len(self.results.multi_hand_landmarks)
        ):
            hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = img.shape
            for idx, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_vals.append(cx)
                y_vals.append(cy)
                self.landmark_list.append([idx, cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            bbox = (min(x_vals), min(y_vals), max(x_vals), max(y_vals))
            if draw:
                cv2.rectangle(
                    img,
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[2] + 20, bbox[3] + 20),
                    (0, 255, 0),
                    2,
                )

        return self.landmark_list, bbox

    def fingers_up(self) -> List[int]:
        """Return a list of five 0/1 values indicating which fingers are raised.

        Index order: [Thumb, Index, Middle, Ring, Pinky].
        Requires ``find_position`` to have been called first.
        """
        if not self.landmark_list:
            return []

        fingers = []

        # Thumb – compare x of tip vs IP joint (works for right-hand view)
        if (
            self.landmark_list[self.TIP_IDS[0]][1]
            > self.landmark_list[self.TIP_IDS[0] - 1][1]
        ):
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers – tip y above PIP y means extended
        for i in range(1, 5):
            if (
                self.landmark_list[self.TIP_IDS[i]][2]
                < self.landmark_list[self.TIP_IDS[i] - 2][2]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_handedness(self) -> List[str]:
        """Return ``['Left']``, ``['Right']``, or both, for detected hands."""
        labels: List[str] = []
        if self.results and self.results.multi_handedness:
            for hand_h in self.results.multi_handedness:
                labels.append(hand_h.classification[0].label)
        return labels

    @staticmethod
    def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance between two 2-D points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# ------------------------------------------------------------------
# Quick standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = detector.find_hands(frame)
        lm_list, _ = detector.find_position(frame)
        if lm_list:
            print("Thumb tip:", lm_list[4])
        cv2.imshow("Hand Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
