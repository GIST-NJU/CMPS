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

```
![Examples of the Selected MRs Applied to an Image of a Cat](https://github.com/GIST-NJU/CMPS/blob/main/mrs_example.png)
```
