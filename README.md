<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> SC-Tune: Unleashing Self-Consistent Referential Comprehension in Large Vision Language Models </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a target="_blank" style="text-decoration: none;">Tongtian Yue<sup>1,3*</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=IOiro9MAAAAJ&hl=zh-CN" target="_blank" style="text-decoration: none;">Jie Cheng<sup>2,3*</sup></a>&nbsp;,&nbsp;
    <a target="_blank" style="text-decoration: none;">Longteng Guo<sup>1,3*</sup></a>&nbsp;,&nbsp;
    <a target="_blank" style="text-decoration: none;">Xingyuan Dai<sup>2,3</sup></a>&nbsp;,&nbsp;
    <a target="_blank" style="text-decoration: none;">Zijia Zhao<sup>1,3</sup></a>&nbsp;,&nbsp; 
	<a target="_blank" style="text-decoration: none;">Xingjian He<sup>1,3</sup></a>&nbsp;&nbsp;
    <a target="_blank" style="text-decoration: none;">Gang Xiong<sup>2,3</sup></a>&nbsp;&nbsp;
    <a target="_blank" style="text-decoration: none;">Yisheng Lv<sup>2,3</sup></a>&nbsp;&nbsp;
    <a target="_blank" style="text-decoration: none;">Jing Liu<sup>1,3†</sup></a>&nbsp;&nbsp;
	<br>
<sup>1</sup>Laboratory of Cognition and Decision Intelligence for Complex Systems, CASIA&nbsp;&nbsp;&nbsp;
<sup>2</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA&nbsp;&nbsp;&nbsp;
<sup>3</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences&nbsp;&nbsp;&nbsp;
</p>


<p align='center';>
<b>
<em>CVPR, 2024</em> <br>
</b>
</p>

<!-- ## Abstract

Recent trends in Large Vision Language Models (LVLMs) research have been increasingly focusing on advancing beyond general image understanding towards more nuanced, object-level referential comprehension. In this paper, we present and delve into the self-consistency capability of LVLMs, a crucial aspect that reflects the models' ability to both generate informative captions for specific objects and subsequently utilize these captions to accurately re-identify the objects in a closed-loop process. This capability significantly mirrors the precision and reliability of fine-grained visual-language understanding. Our findings reveal that the self-consistency level of existing LVLMs falls short of expectations, posing limitations on their practical applicability and potential. To address this gap, we introduce a novel fine-tuning paradigm named **Self-Consistency Tuning (SC-Tune)**. It features the synergistic learning of a cyclic describer-locator system. This paradigm is not only data-efficient but also exhibits generalizability across multiple LVLMs. Through extensive experiments, we demonstrate that SC-Tune significantly elevates performance across a spectrum of object-level vision-language benchmarks and maintains competitive or improved performance on image-level vision-language benchmarks. -->

## Requirements

### Installation

Create a conda environment and install dependencies:

```bash
conda create -n sc_tune python=3.10
conda activate sc_tune
pip install -r requirements.txt
```

### Data

Download the [Qwen-VL-Chat checkpoint](https://huggingface.co/Qwen/Qwen-VL-Chat/tree/main) (10 *.bin files in total) to the path `Qwen-VL-Chat/` and [Object365 images](https://www.objects365.org/download.html).

## Get Started

### Configs

Set the path of Object365 images in `scripts/finetune_ds.sh`. Other hyperparameters can also be found in this file.

### Running

```bash
sh scripts/finetune_ds.sh
```

### Main codes

The main codes to implement sc-tune method is in `transformers/trainer.py` and `transformers/trainer_utils.py`.

## Acknowledgement

This repo benefits from [Qwen-VL](https://github.com/QwenLM/Qwen-VL), [TRL](https://github.com/huggingface/trl), and [MOSS](https://github.com/OpenLMLab/MOSS-RLHF). Thanks for their wonderful work.

## Citation

```latex
@article{yue2024sc,
  title={SC-Tune: Unleashing Self-Consistent Referential Comprehension in Large Vision Language Models},
  author={Yue, Tongtian and Cheng, Jie and Guo, Longteng and Dai, Xingyuan and Zhao, Zijia and He, Xingjian and Xiong, Gang and Lv, Yisheng and Liu, Jing},
  journal={arXiv preprint arXiv:2403.13263},
  year={2024}
}
```