
import cv2
import numpy as np
import time
from typing import List, Tuple


class Key:
    """Represents a single key on the virtual keyboard."""

    def __init__(self, pos: List[int], text: str, size: Tuple[int, int] = (60, 60)):
        self.pos = list(pos)      # [x, y] top-left corner
        self.size = list(size)    # [width, height]
        self.text = text


class KeyboardEngine:
    """Draw and manage a virtual keyboard overlaid on a camera frame.

    Parameters
    ----------
    controller
        A ``pynput.keyboard.Controller`` (or compatible) instance used to
        emit real key-press events to the OS.
    key_size : tuple of int
        ``(width, height)`` of each key in pixels.
    key_spacing : int
        Horizontal/vertical spacing between keys.
    origin : tuple of int
        ``(x, y)`` pixel offset for the top-left corner of the keyboard grid.
    pinch_threshold : int
        Maximum pixel distance between index tip and thumb tip to count
        as a "pinch" (i.e. a key press).
    debounce_seconds : float
        Minimum seconds between consecutive presses of the *same* key.
    alpha : float
        Transparency of the keyboard overlay (0 = invisible, 1 = opaque).
    """

    # Default QWERTY layout – easily swappable
    DEFAULT_LAYOUT = [
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ]

    def __init__(
        self,
        controller,
        layout: List[List[str]] = None,
        key_size: Tuple[int, int] = (60, 60),
        key_spacing: int = 70,
        origin: Tuple[int, int] = (30, 30),
        pinch_threshold: int = 30,
        debounce_seconds: float = 0.20,
        alpha: float = 0.5,
    ):
        self.controller = controller
        self.pinch_threshold = pinch_threshold
        self.debounce_seconds = debounce_seconds
        self.alpha = alpha

        layout = layout or self.DEFAULT_LAYOUT

        self.keys: List[Key] = []
        for row_idx, row in enumerate(layout):
            for col_idx, char in enumerate(row):
                x = key_spacing * col_idx + origin[0]
                y = key_spacing * row_idx + origin[1]
                self.keys.append(Key([x, y], char, size=key_size))

        self._last_press_time: dict[str, float] = {}
        self.typed_text: str = ""

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, img) -> np.ndarray:
        """Render the keyboard overlay on *img* and return the composited frame."""
        overlay = np.zeros_like(img, dtype=np.uint8)

        for key in self.keys:
            x, y = key.pos
            w, h = key.size
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
            cv2.putText(
                overlay,
                key.text,
                (x + 15, y + 45),
                cv2.FONT_HERSHEY_PLAIN,
                2.5,
                (255, 255, 255),
                3,
            )

        out = img.copy()
        mask = overlay.astype(bool)
        blended = cv2.addWeighted(img, 1 - self.alpha, overlay, self.alpha, 0)
        out[mask] = blended[mask]
        return out

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def check_touches(self, lm_list: List[List], img) -> np.ndarray:
        """Detect pinch-over-key events and emit key presses.

        Parameters
        ----------
        lm_list : list
            Landmark list from ``HandDetector.find_position``.
            Each entry is ``[id, x, y, z]``.
        img : ndarray
            Current frame (modified in-place with visual feedback).

        Returns
        -------
        img : ndarray
            Annotated frame.
        """
        if not lm_list:
            return img

        # Locate index tip (8) and thumb tip (4)
        index_tip = thumb_tip = None
        for pt in lm_list:
            if pt[0] == 8:
                index_tip = (pt[1], pt[2])
            elif pt[0] == 4:
                thumb_tip = (pt[1], pt[2])

        if index_tip is None or thumb_tip is None:
            return img

        pinch_dist = ((index_tip[0] - thumb_tip[0]) ** 2
                      + (index_tip[1] - thumb_tip[1]) ** 2) ** 0.5

        now = time.time()

        for key in self.keys:
            x, y = key.pos
            w, h = key.size

            # Is index tip hovering over this key?
            if x < index_tip[0] < x + w and y < index_tip[1] < y + h:
                # Highlight hovered key
                cv2.rectangle(img, (x, y), (x + w, y + h), (175, 0, 175), 2)

                if pinch_dist < self.pinch_threshold:
                    last = self._last_press_time.get(key.text, 0.0)
                    if now - last > self.debounce_seconds:
                        # Visual feedback – fill key green
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(
                            img,
                            key.text,
                            (x + 15, y + 45),
                            cv2.FONT_HERSHEY_PLAIN,
                            2.5,
                            (255, 255, 255),
                            3,
                        )
                        self.controller.press(key.text)
                        self.controller.release(key.text)
                        self.typed_text += key.text
                        self._last_press_time[key.text] = now

        # Show running text
        cv2.putText(
            img,
            self.typed_text[-40:],  # last 40 chars
            (30, 450),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            2,
        )

        return img
