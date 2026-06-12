from mediapipe.tasks import python
from mediapipe.tasks.python import vision

my_base_options = python.BaseOptions(
    model_asset_path="../Assets/face_landmarker.task",
    delegate=python.BaseOptions.Delegate.CPU
)

options = vision.FaceLandmarkerOptions(
    base_options=my_base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)