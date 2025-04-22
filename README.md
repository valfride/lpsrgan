# LPSRGan
Unofficial implementation of the SR model proposed by Pan et al. in the paper "LPSRGAN: Generative adversarial networks for super-resolution of license plate image."

# Usage

This section provides instructions on testing the model, training it from scratch, and fine-tuning it on a custom dataset. Follow the steps below to set up and run the model. Additionally,

# Testing

To test the model, ensure that the config file specifies the path to the .pth file (e.g., config file), as shown in the example below:

```yaml
test
```

Once the configuration is set, execute the following command to start the test:

```
python3 test.py
```

## Training from Scratch

To train the model from scratch, update the following variables in the [config file](config/training.yaml):

```yaml
resume: null
```

## Training on a Custom Dataset

To train or fine-tune the model on a custom dataset, you need to create a .txt file that lists the image paths along with their corresponding data split (training, validation, or testing). Each line in the file should follow this format:

```txt
ABC1234;path/to/LP_image1.jpg;training
ABC1A34;path/to/LP_image2.jpg;validation
ABC2B35;path/to/LP_image3.jpg;testing

```
For reference, you can check example files, such as [train_dir_split.txt](train_dir_split.txt) (with its images and .txt files annotations located under [train_dir](train_dir) directory), [split_all_pku.txt](split_all_pku.txt) and [split_all_rodosol.txt](split_all_rodosol.txt), which demonstrate this format.
