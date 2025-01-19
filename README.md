# Facial Expression Recognition

## Overview
This repository provides an implementation of a YOLOv8-based solution for facial expression detection. It includes steps to set up the environment, train a custom model, and run predictions on test data. The dataset is managed using Roboflow, and the model leverages the Ultralytics YOLOv8 framework.

## Prerequisites

- Python 3.8 or later
- pip (Python package manager)
- [Ultralytics](https://github.com/ultralytics/ultralytics) library
- [Roboflow](https://roboflow.com/) account and API key
- NVIDIA GPU (recommended for training)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/facial-expression-ai-solution.git
   cd facial-expression-ai-solution
   ```

2. Install required Python libraries:
   ```bash
   pip install ultralytics roboflow pandas
   ```

## Setup

1. Download the dataset from Roboflow:
   - Use the following snippet to download the dataset:
     ```python
     from roboflow import Roboflow
     rf = Roboflow(api_key="<YOUR_API_KEY>")
     project = rf.workspace("ai-solutions-nbzxp").project("facial-expression-ai-solution")
     version = project.version(1)
     dataset = version.download("yolov8")
     ```
   - Replace `<YOUR_API_KEY>` with your Roboflow API key.

2. Verify the dataset is correctly downloaded and locate the `data.yaml` file.

## Training the Model

1. Import YOLOv8 and load the model:
   ```python
   from ultralytics import YOLO

   # Load the YOLOv8 model
   model = YOLO('yolov8n.pt')
   ```

2. Train the model:
   ```python
   model.train(
       data="path/to/data.yaml",  # Replace with the path to your data.yaml
       epochs=8,                  # Number of training epochs
       imgsz=440,                 # Image size
       batch=16                   # Batch size
   )
   ```
   - Ensure you update the `data` parameter with the correct path to `data.yaml`.

3. Monitor training results:
   - After training, navigate to the output folder (e.g., `runs/detect/trainX`) to review the confusion matrix and results plots.

## Inference

1. Prepare your test image and set paths:
   ```python
   import os

   source_path = r"path/to/test/image.png"  # Path to the test image
   results_folder = r"path/to/output/folder"
   name_folder = "output"

   wtpath = r"path/to/weights/folder"

   # Construct the YOLO command
   command = f'YOLO task=detect mode=predict model={wtpath}\best.pt conf=0.25 source="{source_path}" save=True project="{results_folder}" name="{name_folder}"'

   # Run the command
   os.system(command)
   ```

2. Check the results:
   - Output will be saved in the specified results folder under the `output` subdirectory.

## Evaluation

1. Analyze training metrics:
   ```python
   import pandas as pd

   df = pd.read_csv(f'{resdt}/results.csv')
   print(df.tail(2))
   ```

2. Visualize the confusion matrix and results:
   ```python
   from IPython.display import Image, display

   display(Image(filename=f'{resdt}/confusion_matrix.png'))
   display(Image(filename=f'{resdt}/results.png'))
   ```

## Directory Structure
```
facial-expression-ai-solution/
|
|-- data.yaml                         # Dataset configuration
|-- train.py                          # Script for training the model
|-- inference.py                      # Script for running predictions
|-- requirements.txt                  # Python dependencies
|-- README.md                         # Project documentation (this file)
|-- runs/                             # Directory for training results
```

## Notes
- Replace paths and API keys in the provided code snippets with your own configurations.
- Ensure your system meets the GPU requirements for efficient model training.
- If in future Roboflow deletes or lock the dataset for facial expression recognition, You can contact me via email or on Linkedin for the dataset.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.


---
Thank You😄!

