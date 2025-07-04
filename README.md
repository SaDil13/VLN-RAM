<div align="center">
<h2 align="center">
   <b>Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation</b> 
   <!-- <br /> <font size=3>Under Review</font></b>  -->
</h2>
<div>
<a target="_blank" href="http://sadil13.github.io/">Ziming&#160;Wei</a><sup>1*</sup>,
<a href="https://expectorlin.github.io/" target="_blank">Bingqian&#160;Lin</a><sup>2*</sup>,
<a href="https://scholar.google.com/citations?user=jV19-sIAAAAJ" target="_blank">Yunshuang&#160;Nie</a><sup>1</sup>,
<a href="https://chen-judge.github.io/" target="_blank">Jiaqi&#160;Chen</a><sup>3</sup>,
<a href="https://openreview.net/profile?id=~Shikui_Ma1" target="_blank">Shikui&#160;Ma</a><sup>4</sup>,
<a href="https://xuhangcn.github.io/" target="_blank">Hang&#160;Xu</a><sup>5</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=voxznZAAAAAJ">Xiaodan&#160;Liang</a><sup>1&#9993</sup>
</div>
<sup>1</sup>Shenzhen Campus of Sun Yat-Sen University,&#160</span>
<sup>2</sup>Shanghai Jiao Tong University,&#160</span><br>
<sup>3</sup>The University of Hong Kong,&#160</span>
<sup>4</sup>Dataa Robotics,&#160</span>
<sup>5</sup>Huawei Noah’s Ark Lab</span>
<br />
<sup>*&#160;</sup>Equal contribution&#160;&#160;</span>
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <a href="https://arxiv.org/abs/2503.18065" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
</div>
</div>

______________________________________________________________________

<font size=2>
Data scarcity is a long-standing challenge in the Vision-Language Navigation (VLN) field, which extremely hinders the generalization of agents to unseen environments. Previous works primarily rely on additional simulator data or web-collected images/videos to improve the generalization. However, the simulator environments still face limited diversity, and the web-collected data often requires extensive labor to remove the noise. In this paper, we propose a Rewriting-driven AugMentation (RAM) paradigm for VLN, which directly creates the unseen observation-instruction pairs via rewriting human-annotated training data. Benefiting from our rewriting mechanism, new observation-instruction can be obtained in both simulator-free and labor-saving manners to promote generalization. Specifically, we first introduce Object-Enriched Observation Rewriting, where we combine Vision-Language Models (VLMs) and Large Language Models (LLMs) to derive rewritten object-enriched scene descriptions, enabling observation synthesis with diverse objects and spatial layouts via Text-to-Image Generation Models (T2IMs). Then, we propose Observation-Contrast Instruction Rewriting, which generates observation-aligned rewritten instructions by requiring LLMs to reason the difference between original and new observations. We further develop a mixing-then-focusing training strategy with a random observation cropping scheme, effectively enhancing data distribution diversity while suppressing augmentation data noise during training. Experiments on both the discrete environments (R2R, REVERIE, and R4R datasets) and continuous environments (R2R-CE dataset) show the superior performance and impressive generalization ability of our method.</font>

![motivation](assets/motivation.png)

## :new: Updates
- [03/2025] [Arxiv paper](https://arxiv.org/abs/2503.18065) and code released.
<!-- - [03/2025] We will release our visual features and instructions for VLN training soon. -->
- [04/2025] Our visual features and instructions for VLN training released.

______________________________________________________________________


# Contents

- [Contents](#contents)
- [Installation](#installation)
- [Get VLN-RAM Data](#get-vln-ram-data)
  - [1. Generate Caption Data](#1-generate-caption-data)
  - [2. Generate Panorama](#2-generate-panorama)
  - [3. Generate Instructions](#3-generate-instructions)
- [VLN Training](#vln-training)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)




# Installation

The environment installation of VLN-RAM follows that in [VLN-DUET](https://github.com/cshizhe/VLN-DUET).
1. Follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator) to install Matterport3D simulators.
2. Installation requirements for VLN training:
```setup
cd VLN-RAM
conda create --name vlnram python=3.8.5
conda activate vlnram
pip install -r requirements.txt
```




# Get VLN-RAM Data


## 1. Generate Caption Data
(1) Follow [this](https://github.com/peteanderson80/Matterport3DSimulator) to get the Matterport3D images. 

(2) Follow [this](https://github.com/xinyu1205/recognize-anything) to install the Tag2Text model and caption the panoramas by Tag2Text.

(3) You should fill in the missing paths in this code.
```
cd data_gen
python generate_caption_data.py
```


## 2. Generate Panorama
(1) Follow [this](https://github.com/omerbt/MultiDiffusion) to get the text2pano model.

(2) Follow [this](https://github.com/fuenwang/Equirec2Perspec) to get the discretization algorithm.

(3) You should fill in the missing paths in this code.
```
cd data_gen
python generate_panorama.py
```


## 3. Generate Instructions
(1) Get your own openai key.

(2) You should fill in the missing paths in this code.
```
cd data_gen
python instr_data.py
```


# VLN Training

1. Follow [this](https://github.com/cshizhe/VLN-DUET) to install our baseline method VLN-DUET.

2. Extract the CLIP ViT B/16 features or CLIP ViT L/14 features following [this](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-VLN) or utilize our pre-extracted visual features from [Google Drive](https://drive.google.com/drive/folders/1Abq9QN3YFOxRxSfSBTAgeX0ujoNZ_NVP?usp=sharing).

3. Pretrain and then finetune based on the scripts.

```
cd VLN-DUET
cd pretrain_src
bash run_r2r.sh
bash run_reverie.sh

cd map_nav_src
bash scripts/run_r2r.sh
bash scripts/run_reverie.sh
bash scripts/run_r4r.sh
```



# Citation
If you find this work useful, please consider citing:
```bibtex
@article{wei2025unseen,
  title={Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation},
  author={Wei, Ziming and Lin, Bingqian and Nie, Yunshuang and Chen, Jiaqi and Ma, Shikui and Xu, Hang and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2503.18065},
  year={2025}
}
```



# Acknowledgement
Some of the codes are built upon [VLN-DUET](https://github.com/cshizhe/VLN-DUET), [Equirec2Perspec](https://github.com/fuenwang/Equirec2Perspec), [Tag2Text](https://github.com/xinyu1205/recognize-anything) and [MultiDiffusion](https://github.com/omerbt/MultiDiffusion). Thanks them for their great works!


