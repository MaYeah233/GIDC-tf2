# GIDC - Modified for supporting TensorFlow 2.10
> Forked from [GIDC](https://github.com/FeiWang0824/GIDC) by [FeiWang0824](https://github.com/FeiWang0824)
## Installation guide
1.  **Download files from github**
      * Using the git bash (recommended)

    ```bash
    git clone https://github.com/MaYeah233/GIDC-tf2.git
    ```
      * Or you can just download the `.zip` file and unzip it.
2.  **Open the Anaconda terminal**：
      * For Windows, open **Anaconda Prompt**。
      * For Linux, open **Terminal**。
3.  **Navigate to the file directory**：
    Use the `cd` command to switch to the folder where you have the project. For example:

    ```bash
    cd path\to\your\project\GIDC-tf2
    ```
4.  **Execute the create command**：
    Run the following command to create a Conda environment called `gidc`.

    ```bash
    conda env create -f environment.yml
    ```
    This process may take several minutes, depending on your network and PC specs. Afterwards, run the following command to activate the environment you just created.
    ```bash
    conda activate gidc
    ```
5.  **Run the GIDC python file**
    Run the following command to run the code with the test data.
    ```bash
    python GIDC_main.py
    ```








---
Here is the content of the original readme file.
# GIDC

Tensorflow implementation of paper: [Far-field super-resolution ghost imaging with a deep neural network constraint.](https://www.nature.com/articles/s41377-021-00680-w). One of the experiment data was provided.

## Citation
If you find this project useful, we would be grateful if you cite the **GIDC paper：**

Fei Wang, Chenglong Wang, Mingliang Chen, Wenlin Gong, Yu Zhang, Shensheng Han and Guohai Situ. Far-field super-resolution ghost imaging with a deep neural network constraint. *Light Sci Appl* **11**, 1 (2022).

## Abstract
Ghost imaging (GI) facilitates image acquisition under low-light conditions by single-pixel measurements and thus has great potential in applications in various fields ranging from biomedical imaging to remote sensing. However, GI usually requires a large amount of single-pixel samplings in order to reconstruct a high-resolution image, imposing a practical limit for its applications. Here we propose a far-field super-resolution GI technique that incorporates the physical model for GI image formation into a deep neural network. The resulting hybrid neural network does not need to pre-train on any dataset, and allows the reconstruction of a far-field image with the resolution beyond the diffraction limit. Furthermore, the physical model imposes a constraint to the network output, making it effectively interpretable. We experimentally demonstrate the proposed GI technique by imaging a flying drone, and show that it outperforms some other widespread GI techniques in terms of both spatial resolution and sampling ratio. We believe that this study provides a new framework for GI, and paves a way for its practical applications.

## Overview
![avatar](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41377-021-00680-w/MediaObjects/41377_2021_680_Fig1_HTML.png?as=webp)

## How to use
**Step 1: Configuring required packages**

python 3.6

tensorflow 1.9

matplotlib 3.1.3

numpy 1.18.1

pillow 7.1.2

**Step 2: Run GIDC_main.py after download and extract the ZIP file.**

## License
For academic and non-commercial use only.
