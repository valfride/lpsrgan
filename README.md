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
python3 test_mv.py --config ./config/testing.yaml --save ./results
```

* Pretrained models are available at the following locations:
  - Download GP_LPR from [GP_LPR Unofficial Implementation](https://github.com/valfride/gplpr/tree/main).
  - The pre-trained model for LPSRGAN is available in the releases section of this repository.

## Training from Scratch

To train the model from scratch, update the following variables in the [training config file](config/LPSRGAN/train_lpsrgan.yaml):

```yaml
resume: null  # Ensure no pretrained model is loaded
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

Optionally, you can add the --tag argument for versioning:

```bash
python3 train.py --config ./config/training.yaml --save path/to/save/dest --tag experiment_1
```

## Training on a Custom Dataset

To train or fine-tune the model on a custom dataset, you need to create a .txt file that lists the image folder paths along with their corresponding data split (training, validation, or testing). Each line in the file should follow this format:

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
resume: path/to/lpsrgan.pth  # Ensure the pretrained model is loaded

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

For reference, you can check example files, such as [split_all_example.txt](split_all_example.txt) (with its images and .txt files annotations located under [dataset_intelbras_1280X960/mercosur](dataset_intelbras_1280X960/mercosur) and [dataset_intelbras_1920x1080](dataset_intelbras_1920x1080) directories), which demonstrate this format.

## Configuration Files

* Training: config/training.yaml
  - Adjust hyperparameters, dataset paths, and optimizer settings here.

* Testing: config/testing.yaml
  - Specify model checkpoints and test dataset details.

## Troubleshooting
* CUDA Out of Memory: Reduce batch size in training.yaml.
* Missing Pretrained Models: Ensure paths in testing.yaml or training.yaml are correct.
* Dataset Errors: Validate split file formatting and image paths.

## Citation

If you use this work, please cite the following works:

```bibtex
@article{pan2024lpsrgan,
  title={LPSRGAN: Generative adversarial networks for super-resolution of license plate image},
  author={Pan, Yuecheng and Tang, Jin and Tjahjadi, Tardi},
  journal={Neurocomputing},
  volume={580},
  pages={127426},
  year={2024},
  publisher={Elsevier}
}

@article{nascimento2025toward,
  title = {Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark},
  author = {V. {Nascimento} and G. E. {Lima} and R. O. {Ribeiro} and W. R. {Schwartz} and R. {Laroca} and D. {Menotti}},
  year = {2025},
  journal = {Journal of the Brazilian Computer Society},
  volume = {},
  number = {},
  pages = {1-14},
  doi = {},
  issn = {},
}
```

Additionally, consider showing your support by **starring** :star: this repository.

## Related Publications
Explore our other works on license plate recognition and super-resolution:
- [Combining Attention Module and Pixel Shuffle for License Plate Super-resolution (SIBGRAPI 2022)](https://ieeexplore.ieee.org/document/9991753)
- [Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers (Computers & Graphics 2023)](https://www.sciencedirect.com/science/article/pii/S0097849323000602?via%3Dihub)
- [Full list of publications on vehicle identification](https://scholar.google.com/scholar?hl=pt-BR&as_sdt=0%2C5&as_ylo=2018&q=allintitle%3A+plate+OR+license+OR+vehicle+author%3A%22David+Menotti%22&btnG=)

## Contact
For questions or feedback, contact:

**Valfride Wallace do Nascimento** [[Webpage](https://www.inf.ufpr.br/vwnascimento/)]

[vwnascimento@inf.ufpr.br](mailto:email@example.com)
