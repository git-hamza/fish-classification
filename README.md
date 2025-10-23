# fish-classification

A simple pipeline to train and test a ResNet-18 model for fish image classification.

## Setup
- Use Python 3.10+ and pip.
- (Recommended) Create and activate a virtual environment.
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Run

- Train:
  ```
  python main.py --mode train
  ```

- Test (specify an image path):
  ```
  python main.py --mode test --img_path path/to/image.png
  ```

Notes:
- Training and evaluation settings (batch size, learning rate, epochs) are defined in `configs/config.py`.
- Classes are read from `constants.CLASS_FILE`.
- The trained checkpoint is expected at `constants.CKPT_PATH/ckpt.pt` when running in test mode.

## Project overview
**Task:** Create an image classification pipeline

Core requirements
- Data pipeline (load, data augmentation)
- Finetune a pre-trained model (simple training loop)

**Data:**
- Images of different fish species
- 9 classes × ~1000 images per class

### Solution:

Steps taken to solve the problem:

- Understand the data (visualize samples; inspect shapes, sizes, and class distribution)
- Experiment with data/model/evaluation in `playground.ipynb`
- Implement production-ready code (with planned improvements due to time constraint)

Some details about the code:
- **configs**: centralized configuration in `configs/config.py`
- **dataloader**: custom Dataset applying transformations and augmentations. Note: the current split logic needs improvement to avoid duplicating data.
- **model**: fine-tuned a small ResNet‑18.
- **evaluation**: currently uses accuracy due to time constraints. We plan to add additional metrics (confusion matrix, precision, recall, F1).



### References:
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://medium.com/@imabhi1216/fine-tuning-a-pre-trained-resnet-18-model-for-image-classification-on-custom-dataset-with-pytorch-02df12e83c2c

### Roadmap / TODO
- REST API for inference
- Containerization (Docker)
- Better logging and experiment tracking (e.g., WandB)
- Additional evaluation metrics (confusion matrix, precision, recall, F1)
- Improve dataset splitting to prevent duplication