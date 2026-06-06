

import cv2


def list_ports(max_ports: int = 10):
    """Probe camera indices 0 … *max_ports*-1 and report status."""
    print(f"Scanning camera ports 0-{max_ports - 1} …")
    working = []
    for port in range(max_ports):
        cam = cv2.VideoCapture(port)
        if not cam.isOpened():
            print(f"  Port {port}: not available")
            continue
        ok, _ = cam.read()
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam.release()
        if ok:
            print(f"  Port {port}: WORKING ({w}x{h})")
            working.append(port)
        else:
            print(f"  Port {port}: opens but returns no frames")
    if not working:
        print("\nNo working cameras found.")
    else:
        print(f"\nWorking ports: {working}")
    return working


if __name__ == "__main__":
    list_ports()
