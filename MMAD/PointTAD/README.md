# Micro-Action Analysis Grand Challenge

The baseline code for Track 2: Multi-label Micro-Action Detection (MMAD) of [Micro-Action Analysis Grand Challenge](https://sites.google.com/view/micro-action) hosted in [ACM MM 2024](https://2024.acmmm.org/). 

## Dependencies

Create environment based on conda.

PyTorch 1.8.1 or higher, opencv-python, scipy, terminaltables, ruamel-yaml, ffmpeg

```
conda create -n mmad python=3.8
conda activate mmad
pip install -r requirements.txt
```

## Data Preparation

 - Clone the repository and `cd PointTAD`.
 - Put the raw videos into `./data/mmad_video/`
 - Prepare the RGB frames and corresponding annotations

```bash
python util/extrac_frames_fast.py
python util/generate_frame_dict.py
```

[Optional] Once you had the raw frames, you can convert them into tensors with `./util/frames2tensor.py` to speed up IO. By enabling `--img_tensor` in train.sh and test.sh, the model takes in image tensors instead of frames.

## Training
```bash
bash train.sh
```

## Oneline Evaluation

### Evaluation on Validation set

```bash
bash eval_val.sh
python online_eval/convert.py
```

### Evaluation on Test set
```bash
bash eval_test.sh
python online_eval/convert.py
```

Then, submit the `submission.zip` on the Codabench. 


If you have any questions, please feel free to cotact us by [email](mailto:kunli.hfut@gmail.com).


