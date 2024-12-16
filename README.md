# Optical Flow Environment Setup  

This guide provides step-by-step instructions to set up a Python environment for working with PyTorch, MMFlow, and MMCV.  

## Installation  

### 1. Create a Python Environment  
Run the following command to create a virtual environment:  
```bash  
python3 -m venv opticalFlow
```

### 2. Activate the Environment
Activate your newly created environment:
```bash 
source opticalFlow/bin/activate
```

### 3. Install pytorch
```bash 
pip3 install torch torchvision
```

### 4. Install MMVC and MIM

```bash 
pip install -U openmim
mim install mmcv-full  
```

### 5. Install MMFLOW
```bash 
pip install mmflow
```

### 6. ffmpeg is required for this code:
```bash 
sudo apt install ffmpeg
```

## Code Execution
On the project folder execute:
```bash 
python movementdetection.py video_file.mp4
```
Replace video_file.mp4 with the path to the video file you want to process.

## Output

The output will be:
A folder with the same name as the video file, containing:
Frames extracted from the video, saved as PNG files in a folder.

Two CSV files:
```bash 
1. motion_data.csv:
    - frame_index: Index of each frame.
    - total_motion_per_frame: Total motion detected in each frame.
    - average_motion_per_frame: Average motion detected in each frame.

2. normalized_data.csv:
    - frame_index: Index of each frame.
    - total_motion: Total motion per frame, normalized using min-max normalization.
    - average_motion: Average motion per frame, normalized using min-max normalization.
```


