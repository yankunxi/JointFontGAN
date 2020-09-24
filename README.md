# JointFontGAN in PyTorch

This is the implementation of the [](https://arxiv.org/abs/). The code was written by [](https://github.com/).
If you use this code or our [](https://github.com/) for your research, please cite:


## Prerequisites:
- Linux or macOS
- Python 3.6 or later (latest built on Python 3.8)
- Pytorch 1.2 or later (latest built on Pytorch 1.6)
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
```bash
pip install visdom
pip install dominate
pip install scikit-image
```

- Clone this repo:
```bash
mkdir xifontgan
cd FontTransfer
git clone https://github.com/azadis/MC-GAN
cd MC-GAN
```

### MC-GAN train/test
- Download our gray-scale 10K font data set:
<img src="https://people.eecs.berkeley.edu/~sazadi/MCGAN/datasets/github_bw.png" width="90%"/>

```bash
./datasets/download_font_dataset.sh Capitals64
```

```../datasets/Capitals64/test_dict/dict.pkl``` makes observed random glyphs be similar at different test runs on Capitals64 dataset. It is a dictionary with font names as keys and random arrays containing indices from 0 to 26 as their values. Lengths of the arrays are equal to the number of non-observed glyphs in each font.

```../datasets/Capitals64/BASE/Code New Roman.0.0.png``` is a fixed simple font used for training the conditional GAN in the End-to-End model.


- Download our collected in-the-wild font data set (downloaded from http://www6.flamingtext.com/All-Logos):

```bash
./datasets/download_font_dataset.sh public_web_fonts
```
Given a few letters of font ```${DATA}``` for examples 5 letters {T,O,W,E,R}, training directory ```${DATA}/A``` should contain 5 images each with dimension ```64x(64x26)x3``` where ```5 - 1 = 4``` letters are given and the rest are zeroed out. Each image should be saved as ```${DATA}_${IND}.png``` where ```${IND}``` is the index (in [0,26) ) of the letter omitted from the observed set. Training directory ```${DATA}/B``` contains images each with dimension ```64x64x3``` where only the omitted letter is given. Image names are similar to the ones in ```${DATA}/A``` though. ```${DATA}/A/test/${DATA}.png``` contains all 5 given letters as a ```64x(64x26)x3```-dimensional image. Structure of the directories for above real-world fonts (including only a few observed letters) is as follows.  One can refer to the examples in ```../datasets/public_web_fonts``` for more information.

```
../datasets/public_web_fonts
                      └── ${DATA}/
                          ├── A/
                          │  ├──train/${DATA}_${IND}.png
                          │  └──test/${DATA}.png
                          └── B/
                             ├──train/${DATA}_${IND}.png
                             └──test/${DATA}.png
```

- (Optional) Download our synthetic color gradient font data set:
<img src="https://people.eecs.berkeley.edu/~sazadi/MCGAN/datasets/github_color.png" width="90%"/>

```bash
./datasets/download_font_dataset.sh Capitals_colorGrad64
```


- Train Glyph Network:

```bash
./scripts/train_cGAN.sh Capitals64
```

Model parameters will be saved under ```./checkpoints/GlyphNet_pretrain```. 

- Test Glyph Network after specific numbers of epochs (e.g. 400 by setting ```EPOCH=400``` in ```./scripts/test_cGAN.sh```):
```bash
./scripts/test_cGAN.sh Capitals64
```

- (Optional) View the generated images (e.g. after 400 epochs):
```bash
cd ./results/GlyphNet_pretrain/test_400/
```
If you are running the code in your local machine, open ```index.html```. If you are running remotely via ssh, on your remote machine run:

```bash
python -m SimpleHTTPServer 8881
```

Then on your local machine, start an SSH tunnel: ```ssh -N -f -L localhost:8881:localhost:8881 remote_user@remote_host``` Now open your browser on the local machine and type in the address bar:

```bash
localhost:8881
```

- (Optional) Plot loss functions values during training, from MC-GAN directory:
```bash
python util/plot_loss.py --logRoot ./checkpoints/GlyphNet_pretrain/
```

- Train End-to-End network (e.g. on ```DATA=ft37_1```):
You can train Glyph Network following instructions above or download our pre-trained model by running:

```bash
./pretrained_models/download_cGAN_models.sh
```

Now, you can train the full model:
```bash
./scripts/train_StackGAN.sh ${DATA}
```

- Test End-to-End network:
```bash
./scripts/test_StackGAN.sh ${DATA}
```
results will be saved under ```./results/${DATA}_MCGAN_train```.

- (Optional) Make a video from your results in different training epochs:

First, train your model and save model weights in every epoch by setting ```opt.save_epoch_freq=1``` in ```scripts/train_StackGAN.sh```. Then test in different epochs and make the video by:

```bash
./scripts/make_video.sh ${DATA}
```

Follow the previous steps to visualize generated images and training curves where you replace ```GlyphNet_train``` with ```${DATA}_StackGAN_train```.

### Training/test Details

- Flags: see ```options/train_options.py```, ```options/base_options.py``` and ```options/test_options.py``` for explanations on each flag. 

- Baselines: if you want to use this code to get results of Image Translation baseline or want to try tiling glyphs rather than stacking, refer to the end of ```scripts/train_cGAN.sh``` . If you only want to train OrnaNet on top of clean glyphs, refer to the end of ```scripts/train_StackGAN.sh```. 

- Image Dimension: We have tried this network only on ```64x64``` images of letters. We do not scale and crop images since we set both ```opt.FineSize``` and ```opt.LoadSize``` to ```64```.


### Citation
If you use this code or the provided dataset for your research, please cite our paper:
```
@inproceedings{azadi2018multi,
  title={Multi-content gan for few-shot font style transfer},
  author={Azadi, Samaneh and Fisher, Matthew and Kim, Vladimir and Wang, Zhaowen and Shechtman, Eli and Darrell, Trevor},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  volume={11},
  pages={13},
  year={2018}
}
```

### Acknowledgements
We thank [Elena Sizikova](http://www.cs.princeton.edu/~sizikova/) for downloading all fonts used in the 10K font data set.

Code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).






