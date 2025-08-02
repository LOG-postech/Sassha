## GPT2 pretraining results

The table below shows the validation results of different methods for GPT-2 pretraining. SASSHA achieves the lowest validation loss and perplexity.

|   Method   | Loss |  Perplexity |
|:---------:|:-------------:|:-------------:|
|   AdamW  |      2.9622      |     19.353    |
|  SAM_{AdamW} |      2.9558     |     19.196    |
|  Sophia-G |     2.9307    |     18.751    |
|  Sophia-G (with SAM) |     2.9319    |     18.773    |
|  SPlus |     2.9435     |     18.982    |
|  **SASSHA** |     **2.9173**     |     **18.491**    |

We set hessian power $\alpha $ to 0.8, which yielded the best results in our experiments. Detailed hyperparameter settings for each method can be found in the `config` directory.