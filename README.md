[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC
Quality Assessment for AI-Generated Content - Track 1: Image
</b>
</h1>
<b>
Beijing University of Posts and Telecommunications. </br>
Beijing Xiaomi Mobile Software Co., Ltd.
</b>
 
</div>

![network](https://github.com/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC/assets/15050507/70edc6f2-2740-47d2-b896-0977afda723a)

# Environment Installation
* torch 1.8+
* torchvision
* Python 3
* pip install ftfy regex tqdm
* pip install git+https://github.com/openai/CLIP.git

# Data Preparation
Download the competition test dataset from the specified website and unzip it into the "<u>./data/AIGCQA-30K-Image/test</u>" directory.

# Trained Weights
 Download weights from: [google drive](https://drive.google.com/drive/folders/11c92mV5wuDuiPyaJzlVVKi2V0Sy1-TCq?usp=drive_link).

# Evaluation on AIGCQA-30k-Image test-set
After preparing the code environment and downloading the data and model weights, run the following code to obtain the output.txt file.
```bash
python AIGC_DB_prompt_final.py
```

# If you find our work is useful, pleaes cite our paper:
Paper will be released.

# Our other works:
+ "Thinking Image Color Aesthetics Assessment: Models, Datasets and Benchmarks.", [[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/He_Thinking_Image_Color_Aesthetics_Assessment_Models_Datasets_and_Benchmarks_ICCV_2023_paper.pdf) [[code]](https://github.com/woshidandan/Image-Color-Aesthetics-Assessment), ICCV 2023.
+ "EAT: An Enhancer for Aesthetics-Oriented Transformers.", [[pdf]](https://github.com/woshidandan/Image-Aesthetics-Assessment/blob/main/Paper_ID_847_EAT%20An%20Enhancer%20for%20Aesthetics-Oriented%20Transformers.pdf) [[code]](https://github.com/woshidandan/Image-Aesthetics-Assessment/tree/main) ACMMM 2023.
+ "Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks.", [[pdf]](https://www.ijcai.org/proceedings/2022/0132.pdf) [[code]](https://github.com/woshidandan/TANet) IJCAI 2022.
