# fish-classification

A simple pipeline to train and test a ResNet-18 model for fish image classification.

## Setup
- Ensure Python and pip are installed.
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

### Helperdocs:
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://medium.com/@imabhi1216/fine-tuning-a-pre-trained-resnet-18-model-for-image-classification-on-custom-dataset-with-pytorch-02df12e83c2c

### TODO:
- rest api to for inference
- containerize
- monitoring