"""
Detect face landmarks on a rendered image with MediaPipe Face Landmarker (478 points, FaceMesh topology).

    lipsync/.venv/bin/python lipsync/face_landmarks.py face.png landmarks.json [--debug debug.png]

Writes {"width", "height", "points": [[x_px, y_px, z], ...]} (478 points, pixels from the top-left) or exits 2 when
no face is found. Runs in its own venv (mediapipe does not install into Blender's Python).
"""
import json
import os
import sys

import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'face_landmarker.task')

argv = sys.argv[1:]
src, out = argv[0], argv[1]
debug = argv[argv.index('--debug') + 1] if '--debug' in argv else None

opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MODEL), num_faces=1,
                                    min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
det = vision.FaceLandmarker.create_from_options(opts)
img = mp.Image.create_from_file(src)
res = det.detect(img)
det.close()  # explicit: closing it during interpreter shutdown prints a harmless traceback
if not res.face_landmarks:
    print('no face found in', src, file=sys.stderr)
    sys.exit(2)
w, h = img.width, img.height
pts = [[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]]
json.dump({'width': w, 'height': h, 'points': pts}, open(out, 'w'))
print('landmarks', len(pts), '->', out, file=sys.stderr)

if debug:
    import numpy as np
    import cv2
    im = cv2.imread(src)
    for i, (x, y, _) in enumerate(pts):
        cv2.circle(im, (int(x), int(y)), 1, (0, 200, 0), -1)
    lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    cv2.polylines(im, [np.array([[int(pts[i][0]), int(pts[i][1])] for i in lips])], True, (0, 0, 255), 1)
    cv2.imwrite(debug, im)
