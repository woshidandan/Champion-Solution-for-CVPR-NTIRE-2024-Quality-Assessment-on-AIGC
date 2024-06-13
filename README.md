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
Download [AGIQA-1K](https://github.com/lcysyzxdxc/AGIQA-1k-Database), [AGIQA-3K](https://github.com/lcysyzxdxc/AGIQA-3k-Database), [AIGCIQA2023](https://github.com/wangjiarui153/AIGCIQA2023) and [AIGCQA-30K-Image](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image/summary) datasets and unzip them into the "<u>./data</u>" directory.

# Training and Testing

After preparing the code environment and downloading the data, run the following codes to train and test model.

```bash
#AIGCQA-30K-Image
python train_aigcqa30k.py
#AGIQA-1K
python train_aigc_agiqa1k.py
#AGIQA-3K
python train_aigc_agiqa3k.py
#AIGCIQA2023
python train_aigc_aigciqa2023.py
```

For AIGCQA-30-Image dataset, run the following codes to get val and test output.

```bash
AIGC_DB_AIGCQA30K_VAL.py
AIGC_DB_AIGCQA30K_TEST.py
```
# Citation

If you find our work useful in your research, please consider citing our paper:
```bash
@InProceedings{Peng_2024_CVPR,
    author    = {Peng, Fei and Fu, Huiyuan and Ming, Anlong and Wang, Chuanming and Ma, Huadong and He, Shuai and Dou, Zifei and Chen, Shu},
    title     = {AIGC Image Quality Assessment via Image-Prompt Correspondence},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {6432-6441}
}
```

# Our other works:
+ "Thinking Image Color Aesthetics Assessment: Models, Datasets and Benchmarks.", [[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/He_Thinking_Image_Color_Aesthetics_Assessment_Models_Datasets_and_Benchmarks_ICCV_2023_paper.pdf) [[code]](https://github.com/woshidandan/Image-Color-Aesthetics-Assessment), ICCV 2023.
+ "EAT: An Enhancer for Aesthetics-Oriented Transformers.", [[pdf]](https://github.com/woshidandan/Image-Aesthetics-Assessment/blob/main/Paper_ID_847_EAT%20An%20Enhancer%20for%20Aesthetics-Oriented%20Transformers.pdf) [[code]](https://github.com/woshidandan/Image-Aesthetics-Assessment/tree/main) ACMMM 2023.
+ "Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks.", [[pdf]](https://www.ijcai.org/proceedings/2022/0132.pdf) [[code]](https://github.com/woshidandan/TANet) IJCAI 2022.
