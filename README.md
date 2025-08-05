# ECGFounder: An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains

This is the official implementation of our paper "[An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains](https://arxiv.org/abs/2410.04133)".

> Authors: Jun Li, Aaron Aguirre, Junior Moura, Jiarui Jin, Che Liu, Lanhai Zhong, Chenxi Sun, Gari Clifford, Brandon Westover, Shenda Hong.



## 🚀 Getting Started

🚩 **News** 
(Aug 2025): The out-of-the-box feature — the 150-class classification validation function of ECGFounder — is now online.

(Mar 2025): The pre-training checkpoint is now available on [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main)!


> ⚠️ **Important Notice**  
> If you intend to use the pretrained model weights for validation or fine-tuning, you must **strictly** follow the preprocessing steps in **dataset.py** — including **filtering**, **z-score normalization**, and any other specified procedures.  
> Failure to do so will make it difficult to reproduce the results reported in the paper!


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

### 150-class classification validation

* **PTB-XL**: Please download the [PTB-XL](https://www.physionet.org/content/ptb-xl/1.0.3/) dataset from physionet.

Next, please download the model's checkpoint from the  [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main). And place the model weights in path *./checkpoint*

You can run the *ptbxl_eval.py* to do the 150-class classification validation on PTB-XL dataset.


### Fine-tune on Downstream Tasks

In our paper, downstream datasets we used are as follows:

* **MIMIC-ECG**: Please download the [MIMIC-ECG](https://physionet.org/content/mimiciv/2.2/) dataset from physionet.


You can run the jupyter notebook to finetune the model by the example dataset.



## References

If you found our work useful in your research, please consider citing our works at:
> ```
> @article{li2025electrocardiogram,
>   title={An Electrocardiogram Foundation Model Built on over 10 Million Recordings},
>   author={Li, Jun and Aguirre, Aaron D and Junior, Valdery Moura and Jin, Jiarui and Liu, Che and Zhong, Lanhai and Sun, Chenxi and Clifford, Gari and Brandon Westover, M and Hong, Shenda},
>   journal={NEJM AI},
>   volume={2},
>   number={7},
>   pages={AIoa2401033},
>   year={2025},
>   publisher={Massachusetts Medical Society}
> }
> ```
