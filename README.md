This repository contains the code for the model described in my Bachelor's thesis on error classification. The model's purpose is to classify 19 classes of errors in the [REALEC](https://realec.org/index.xhtml#/) learner corpus (for more information on what the classes are and how they are selected, see the thesis).

The chronological order of the files for training the model is as follows: `create_training_data.ipynb`, `split_train_by_class.ipynb`, `model_training.ipynb`. To replicate the model's results, however, one may take advantage of the `predict.ipynb` file using the uploaded training files and final models.

### 'Train' folder
The `train` folder contains several legacy train files used in the initial stages of development to establish what training format the model would perform with best. I did not remove those files after settling on the final format; an interested reader may try to train the models with the older training format and compare the results. The code in `model_training.ipynb` is not dependent on the format of the training files (some of them may have numerical values instead of text labels for classes, however; and in that case minor tweaks to the code structure may be in order for it to run correctly).

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

To predict the tag for the error, simply input the error as an argument to the TextClassificationPipeline object:

```
>>> discourse("can't → cannot")

[{'label': 'Inappropriate_register', 'score': 0.9809362888336182}]
```
