# ECGFounder: An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains

This is the official implementation of our paper "[An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains](https://arxiv.org/abs/2410.04133)".

> Authors: Jun Li, Aaron Aguirre, Junior Moura, Jiarui Jin, Che Liu, Lanhai Zhong, Chenxi Sun, Gari Clifford, Brandon Westover, Shenda Hong.



## 🚀 Getting Started

🚩 **News** (Mar 2025): The pre-training checkpoint is now available on [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main)!



### Installation

To clone this repository:

```
git clone https://github.com/PKUDigitalHealth/ECGFounder.git
```

### Environment Set Up

Install required packages:

```
conda create -n ECGFounder python=3.10
conda activate ECGFounder
pip install -r requirements.txt
```



### Fine-tune on Downstream Tasks

In our paper, downstream datasets we used are as follows:

* **MIMIC-ECG**: Please download the [MIMIC-ECG](https://physionet.org/content/mimiciv/2.2/) dataset from physionet.

Next, please download the model's checkpoint from the  [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main). And place the model weights in path *./checkpoint*



You can run the jupyter notebook to finetune the model by the example dataset.



## References

If you found our work useful in your research, please consider citing our works at:
> ```
> @article{li2024electrocardiogram,
>   title={An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains},
>   author={Li, Jun and Aguirre, Aaron and Moura, Junior and Liu, Che and Zhong, Lanhai and Sun, Chenxi and Clifford, Gari and Westover, Brandon and Hong, Shenda},
>   journal={arXiv preprint arXiv:2410.04133},
>   year={2024}
> }
> ```
