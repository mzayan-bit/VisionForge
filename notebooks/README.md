# VisionForge Research Notebooks

This directory contains interactive Jupyter notebooks and Google Colab templates for computer vision research workflows.

## Colab GPU Training Workflow

To execute compute-intensive training on Google Colab free T4 GPUs:

1. Clone the repository into your Colab session:
   ```bash
   !git clone https://github.com/mzayan-bit/VisionForge.git
   %cd VisionForge
   !pip install -e ./backend
   ```
2. Run the remote training runner:
   ```bash
   !python scripts/train_colab.py --preparation-id prep_coco8_v1_0_0 --model yolo11n.pt --epochs 25 --device 0
   ```
3. Export the resulting `runs/weights/best.pt` checkpoint back to your VisionForge local or containerized model registry.
