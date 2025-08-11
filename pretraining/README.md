## GPT-2 pretraining results

The table below shows the validation results of different methods for GPT-2 small pretraining. SASSHA achieves the lowest validation loss and perplexity.

|   Method   | Loss |  Perplexity |
|:---------:|:-------------:|:-------------:|
|   AdamW  |      2.9622      |     19.353    |
|  SAM_{AdamW} |      2.9558     |     19.196    |
|  Sophia-G |     2.9307    |     18.751    |
|  Sophia-G (with SAM) |     2.9319    |     18.773    |
|  SPlus |     2.9435     |     18.982    |
|  **SASSHA** |     **2.9173**     |     **18.491**    |

We set hessian power $\alpha $ to 0.8, which yielded the best results in our experiments. Detailed hyperparameter settings for each method can be found in the `config` directory.

## Reproduce GPT-2 Results

Prepare the [OpenWebText](https://huggingface.co/datasets/openwebtext) data following [nanoGPT](https://github.com/karpathy/nanoGPT/):
```
$ python data/openwebtext/prepare.py
```
Start pre-training GPT2 Small (125M):

We ran our experiments on a machine with 6 A100 (80GB) GPUs.
```
$ torchrun --standalone --nproc_per_node=6 \
      train_sassha.py \
      config/train_gpt2_small_sassha.py \
      --batch_size=10 \
      --gradient_accumulation_steps=8