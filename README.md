# LPSRGAN: License Plate Super-Resolution GAN

Unofficial implementation of the SR model proposed by Pan et al. in the paper "LPSRGAN: Generative Adversarial Networks for Super-Resolution of License Plate Image". This repository follows the methodology described by Nascimento et al. in "Toward Advancing License Plate Super-Resolution in Real-World Scenarios".

## Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU acceleration)
- Required packages: `numpy`, `opencv-python`, `tqdm`, `yaml`, `torch`
  Install via:  
  ```bash
  pip install -r requirements.txt
  ```
# Installation

```bash
git clone https://github.com/your-username/LPSRGAN.git
cd LPSRGAN
```

# Usage

This section provides instructions on testing the model, training it from scratch, and fine-tuning it on a custom dataset.

# Testing

To test the model, ensure that the config file specifies the path to each of the .pth files (e.g., config file), as shown in the example below:

```yaml
model:
  name: lpsrgan
  load: ./save/testing/model_lpsrgan.pth
  args:
    num_blocks: 3
    dropout_prob: 0.5

model_ocr:
  name: GPLPR
  load: ./save/testing/model_ocr_GPLPR.pth
  args:
    alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nc: 3
    K: 7
    isSeqModel: True
    head: 2
    inner: 256
    isl2Norm: True
```

Once the configuration is set, execute the following command to start the test:

```
python3 test_mv.py --config ./config/testing.yaml --save ./results --tag example
```

The pre-tained model of GP_LPR can be found in [GP_LPR Unofficial Implementation](https://github.com/valfride/gplpr/tree/main).
The pre-trained model for LPSRGAN can be found in the release section of this repository.

## Training from Scratch

To train the model from scratch, update the following variables in the [config file](config/training.yaml):

```yaml
resume: null
```

Optionally, you can add the --tag argument for versioning:

```python
python3 train.py --config ./config/training.yaml --save True
```

## Training from Scratch

configure the [training.yaml](dummy.com)

```yaml
resume: null  # Ensure no pretrained model is loaded
train_dataset:
  dataset:
    args:
      path_split: ./custom_split.txt
```

Start training:

```bash
python3 train.py --config ./config/training.yaml --save True --tag experiment_1
```

## Training on a Custom Dataset

To train or fine-tune the model on a custom dataset, you need to create a .txt file that lists the image paths along with their corresponding data split (training, validation, or testing). Each line in the file should follow this format:

```txt
<license_plate_text>;<image_path>;<split>
```
Example:

```txt
ABC1234;path/to/LP_image1.jpg;training
XYZ5678;path/to/LP_image2.jpg;validation
```

Update training.yaml to point to your split file:

```yaml
train_dataset:
  dataset:
    args:
      path_split: ./custom_split.txt

val_dataset:
  dataset:
    name: multi_image
    args:
      path_split: ./custom_split.txt
      phase: validation
```

For reference, you can check example files, such as [train_dir_split.txt](train_dir_split.txt) (with its images and .txt files annotations located under [train_dir](train_dir) directory), [split_all_pku.txt](split_all_pku.txt) and [split_all_rodosol.txt](split_all_rodosol.txt), which demonstrate this format.
