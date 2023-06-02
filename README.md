This repository contains the code for the model described in my Bachelor's thesis on error classification. The model's purpose is to classify 19 classes of errors in the [REALEC](https://realec.org/index.xhtml#/) learner corpus (for the full list of tags, see the _Evaluation_ secion below; for the logic behind how the classes were selected, see the thesis).

The chronological order of the files for training the model is as follows: `create_training_data.ipynb`, `split_train_by_class.ipynb`, `model_training.ipynb`. To replicate the model's results, however, one may take advantage of the `predict.ipynb` file using the uploaded training files and final models.

### 'Train' folder
The `train` folder contains several legacy train files used in the initial stages of development to establish what training format the model would perform with best. I did not remove those files after settling on the final format; an interested reader may try to train the models with the older training format and compare the results. The code in `model_training.ipynb` is not dependent on the format of the training files (some of them may have numerical values instead of text labels for classes, however; and in that case minor tweaks to the code structure may be in order for it to run correctly).

The final training files that were used to train the models are named `split_train_{lexical/gram/disc}_eo.pickle` for the three smaller models, and `split_train_eo.pickle` for the general model.

### 'Data' folder
The `data` folder contains .xlsx files created in `statistics_1.ipynb` and `statistics_2.ipynb`.

### Setting up the model
This repository contains the code necessary to train the models. The fine-tuned models described in the thesis are hosted on the HuggingFace hub and can be set up as follows:

```
from transformers import (TextClassificationPipeline,
                          RobertaTokenizerFast,
                          RobertaForSequenceClassification)

general = TextClassificationPipeline(
    model=RobertaForSequenceClassification.from_pretrained("lightcarrieson/general_model"),
    tokenizer=RobertaTokenizerFast.from_pretrained("lightcarrieson/general_model"),
    top_k=None
)

discourse = TextClassificationPipeline(
    model=RobertaForSequenceClassification.from_pretrained("lightcarrieson/discourse_model"),
    tokenizer=RobertaTokenizerFast.from_pretrained("lightcarrieson/discourse_model"),
)

grammar = TextClassificationPipeline(
    model=RobertaForSequenceClassification.from_pretrained("lightcarrieson/grammar_model"),
    tokenizer=RobertaTokenizerFast.from_pretrained("lightcarrieson/grammar_model"),
)

lexical = TextClassificationPipeline(
    model=RobertaForSequenceClassification.from_pretrained("lightcarrieson/lexical_model"),
    tokenizer=RobertaTokenizerFast.from_pretrained("lightcarrieson/lexical_model"),
)
```
### Prediction
Mind that the models were trained on the data in the format of `error → correction`. Predictions on data of any other format may be ... unpredictable.

To predict the tag for the error using a single model, simply input the error as an argument to the TextClassificationPipeline object:

```
>>> discourse("can't → cannot")

[{'label': 'Inappropriate_register', 'score': 0.9809362888336182}]
```
To predict the tags for a dataset, see the `predict.ipynb` pipeline.

### Evaluation

The results for the pipeline performance on the test dataset are the following:

1. General model (all errors are split into three overarching classes, `lexical`, `discourse`., and `grammar`.

| Weighted F0.5 | Micro F0.5 | Macro F0.5 | 
| :---: | :---: | :---: |
| 0.83 | 0.83 | 0.81 |

2. Lexical model

| Weighted F0.5 | Micro F0.5 | Macro F0.5 | 
| :---: | :---: | :---: |
| 0.94 | 0.94 | 0.89 |

3. Discourse model

| Weighted F0.5 | Micro F0.5 | Macro F0.5 | 
| :---: | :---: | :---: |
| 0.73 | 0.73 | 0.73 |

4. Grammar model

| Weighted F0.5 | Micro F0.5 | Macro F0.5 | 
| :---: | :---: | :---: |
| 0.82 | 0.82 | 0.78 |

5. Pipeline

In the final pipeline, three confidence thresholds are applied: the first one equals 0.85 and is applied to the General model's confidence scores to separate the predictions into ones we're more certain of and ones we're less certain of. Then the 'confident' predictions are classified by only the model respective to the class they were predicted for; the 'uncertain' predictions are classified by all three models. The final score is simply the smaller model's score for 'certain' predictions and the product of general class and tag score for 'uncertain' predictions. Finally, we only accept predictions with the final score of 0.7 for 'certain' predictions and 0.63 for 'uncertain' predictions. The following metrics are computed given these thresholds.


<table>
  <tr>
    <th>Predictions accepted</th>
    <td> 75% </td>
  </tr>
  <tr>
    <th>of them correct</th>
    <td>68.3%</td>
  </tr>
  <tr>
    <th>of them wrong</th>
    <td>6.7%</td>
  </tr>
  <tr>
    <th> Predictions refused </th>
    <td> 25% </td>
  </tr>
  <tr>
    <th>Weighted F0.5</th>
    <td>0.90</td>
  </tr>
  <tr>
    <th>Micro F0.5</th>
    <td>0.91</td>
  </tr>
  <tr>
    <th>Macro F0.5</th>
    <td>0.79</td>
  </tr>
</table>

For the full metrics per each of the 19 classes, see the thesis.
