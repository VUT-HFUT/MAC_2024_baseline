# Micro-Action Analysis Grand Challenge

The baseline code for Track 1: Micro-Action Recognition (MAR) of [Micro-Action Analysis Grand Challenge](https://sites.google.com/view/micro-action) hosted in [ACM MM 2024](https://2024.acmmm.org/). 

## Data Preparation

Fetch the video data from [Challenge page](https://sites.google.com/view/micro-action/challenge/data), and place it to `./data`.

---

## 🛠️ Installation
This code build based on [MMAction2](https://github.com/open-mmlab/mmaction2). 
<summary>Quick instructions</summary>
<details close>

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/VUT-HFUT/MAC_2024_baseline.git
cd MAC_2024_baseline/MAR/mmaction2
pip install -v -e .
```
</details>

---

## 🔥 Train
You can use the following command to train a model.
```shell
# Swin Video Transformer base version
python tools/train.py configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_ma52-rgb.py --seed=0 --deterministic
```
---
## Test

```shell

python tools/test.py configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_ma52-rgb.py work_dirs/swin/base/best_acc_top1_epoch_39.pth --dump submit/test_result.pickle
## build the submission file
python ./submit/generate.py
```
---
## Results

| Model | Top-1 (Action) | Top-5 (Action) | F1_mean | ckpt |
| :-: | :-: | :-: | :-: | :-: |
| VSwin-small |**58.88** | 89.63 | **65.64**| [Link](https://huggingface.co/kunli-cs/VSwin_MA52_Weights/tree/main/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-40e_ma52-rgb) |

## Acknowledgments
This code began with [MMAction2](https://github.com/open-mmlab/mmaction2). We thank the developers for doing most of the heavy-lifting.

---
## Citation
Please consider citing the related paper in your publications if it helps your research.
```
@article{guo2024benchmarking,
  title={Benchmarking Micro-action Recognition: Dataset, Methods, and Applications},
  author={Guo, Dan and Li, Kun and Hu, Bin and Zhang, Yan and Wang, Meng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  volume={34},
  number={7},
  pages={6238-6252},
}
```
---

## Contact
**For any questions, feel free to contact: [kunli.hfut@gmail.com](mailto:kunli.hfut@gmail.com?subject=Micro-Action%20Grand%20Chanllenge).**
