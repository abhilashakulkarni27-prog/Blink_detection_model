# BlinkVision

BlinkVision is a real-time blink detection system built using a custom CNN model and MediaPipe Face Landmarks.

The system first detects facial landmarks using MediaPipe, extracts and normalizes the eye regions, and then classifies the eye state into four categories:

* Eyes Open
* Left Eye Closed
* Right Eye Closed
* Both Eyes Closed

Unlike many traditional blink detection approaches that rely on eye-aspect-ratio (EAR) thresholds, BlinkVision uses a lightweight convolutional neural network trained directly on eye images. This allows the model to learn visual blink patterns rather than relying solely on geometric heuristics.

## Features

* Real-time webcam inference
* Lightweight CNN architecture
* Detection of left-eye, right-eye, and full blinks
* MediaPipe-based eye localization
* GPU acceleration support through PyTorch
* Modular codebase for experimentation and extension

## Project Structure

```text
BlinkVision/
├── LICENSE
├── requirements.txt
├── notebooks/
├── models/
├── Assets/
└── src/
```

## Installation

```bash
git clone <repository-url>
cd BlinkVision

pip install -r requirements.txt
```

## Running

```bash
python src/webcam.py
```

Press `q` to quit.

## Method

1. Detect facial landmarks using MediaPipe.
2. Extract left and right eye regions.
3. Resize and combine eye images.
4. Pass the processed image through the CNN classifier.
5. Apply temporal filtering to reduce false blink detections.

## Performance

The model is designed for low-latency real-time inference and runs comfortably on consumer hardware.

While the model is not perfect and can still produce occasional misclassifications, it performed noticeably better in practical testing than several simple threshold-based blink detection implementations while maintaining high inference speed.

## Future Improvements

* Larger training dataset
* Better data augmentation
* Temporal sequence modeling
* Head-pose robustness
* Quantitative benchmarking

## Acknowledgements

* MediaPipe for facial landmark detection
* PyTorch for model training and inference
* Axon – a custom deep learning framework developed as a separate project and used to build the CNN architecture

## Current Limitations

This project is currently a proof-of-concept and is still under active development.

The current model was trained on a custom dataset collected and labeled specifically for this project. While it performs well in many real-time scenarios and demonstrates promising results, the dataset is relatively limited in size and diversity. As a result, the model may not generalize reliably across all users, lighting conditions, camera qualities, head poses, or occlusions.

The primary goal of this version was to build a complete real-time blink detection pipeline and validate the effectiveness of a CNN-based approach compared to simple threshold-based methods.

## Future Work

The next major milestone is training a significantly larger and more diverse model using an expanded dataset. Planned improvements include:

* Increased dataset size and subject diversity
* Better generalization across lighting conditions and camera setups
* Improved robustness to head movement and partial occlusion
* Quantitative benchmarking and evaluation
* Model optimization for deployment on low-power devices

This repository will continue to be updated as more data is collected and improved models are trained.

## Demo Note

The webcam feed shown in the demo is mirrored, similar to the preview displayed by most webcam applications.

As a result, left and right blink detections may appear visually reversed to the viewer. For example, a detected **left-eye blink** may appear as a **right-eye blink** in the displayed video feed.

The model predictions themselves are not affected; only the displayed webcam preview is mirrored.
## Demo

![BlinkVision Demo](assets/demo.gif)
