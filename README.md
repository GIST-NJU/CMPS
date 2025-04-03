# CMPS: Cluster-Based Multi-Objective Metamorphic Test Case Pair Selection for Deep Neural Networks

This repository contains the replication package for the paper "*Cluster-Based Multi-Objective Metamorphic Test Case Pair Selection for Deep Neural Networks*". It provides experimental code, data, and related documentation to support the reproduction of the CMPS approach proposed in the paper and further research.

---

## Repository Structure

The directory is structured as follows:

```

```

### Experimental Subjects

First, we briefly introduce the datasets, DNN models, and metamorphic relations (MRs) used in our experiment.

#### Datasets & DNN Models

In our experiments, we selected three datasets and assigned two DNN models to each dataset, i.e., a total of six experimental subjects, as follows:

| Dataset       | Test Set | DNN Model |
|:-------------:|:--------:|:---------:|
| Fashion-MNIST | 10,000   | LeNet1    |
|               |          | LeNet5    |
| CIFAR-10      | 10,000   | VGG19     |
|               |          | ResNet50  |
| ImageNet      | 10,000   | GoogleNet |
|               |          | ResNet50  |

The `datasets` subfolder under the `subjects` directory contains these three datasets, while the `models` subfolder stores the pretrained models corresponding to these six experimental subjects.

#### Metamorphic Relations (MRs)

We selected five different MRs:

* **MR1 - Flip Left-right** : The DNN model’s outputs should remain consistent when the image is flipped from left to right.
* **MR2 - Gaussian Blur** : The DNN model’s outputs should remain consistent when the image undergoes Gaussian blurring.
* **MR3 - Rotate 5°** : The DNN model’s outputs should remain consistent when the image is rotated by 5&deg;.
* **MR4 - Change Chromaticity** : The DNN model’s outputs should remain consistent when the chromaticity of the image is increased.
* **MR5 - Adjust Brightness** : The DNN model’s outputs should remain consistent when the brightness of the image is increased.

Here's an example demonstrating how MRs transform the source test case into its follow-up test cases.

![Examples of the Selected MRs Applied to an Image of a Cat](mrs_example.png)

The implementations of these five MRs are in the `mr_5.p` file under the `src` folder. You can also add new MRs based on your own needs.

#### Experimental Data


For easy replication, each subfolder in the `experimental_data` directory contains data for the corresponding experimental subjects, including the model's predicted labels for the source test cases, output probabilities of source test cases, and predicted labels for the follow-up test cases. This way, you can directly read the corresponding `.mat` files of each subject for processing without the need to load the models. Taking the `cifar10_vgg19` folder as an example, the results of each MR are stored separately in their respective `.mat` files, totaling five files.

---

## Getting Started

Follow these steps to set up the environment and run the experiments.

### Requirements
Run the following command to set up the same environment as the experiment based on the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

