---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: The startup is building a new mobile app.
- text: The tennis champion won another grand slam.
- text: The stock market crashed today.
- text: The defendant pleaded not guilty in court.
- text: Vaccination helps prevent infectious diseases.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-MiniLM-L6-v2
---

# SetFit with sentence-transformers/paraphrase-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 128 tokens
- **Number of Classes:** 13 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label         | Examples                                                                                                                                                                                        |
|:--------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sports        | <ul><li>'The football match ended in a draw.'</li><li>'He scored a hat-trick in the final.'</li><li>'The Olympic games will be held next year.'</li></ul>                                       |
| politics      | <ul><li>'The government passed a new law today.'</li><li>'The election results were announced last night.'</li><li>'The president gave a speech at the summit.'</li></ul>                       |
| technology    | <ul><li>'The new smartphone features a better camera.'</li><li>'Artificial intelligence is transforming many industries.'</li><li>'The software update improved system performance.'</li></ul>  |
| entertainment | <ul><li>'The movie received excellent reviews from critics.'</li><li>'The singer released a new album.'</li><li>'The TV series finale shocked many fans.'</li></ul>                             |
| business      | <ul><li>'The company reported record profits this quarter.'</li><li>'The merger created the largest bank in the country.'</li><li>'The startup raised funding from investors.'</li></ul>        |
| health        | <ul><li>'The patient needs surgery immediately.'</li><li>'Regular exercise improves cardiovascular health.'</li><li>'The doctor prescribed a new medication.'</li></ul>                         |
| law           | <ul><li>'The contract is valid for five years.'</li><li>'The defendant pleaded not guilty in court.'</li><li>'The lawyer prepared documents for the trial.'</li></ul>                           |
| finance       | <ul><li>'The stock market crashed today.'</li><li>'Inflation rates are rising globally.'</li><li>'He invested his savings in mutual funds.'</li></ul>                                           |
| science       | <ul><li>'The researchers published a paper on quantum physics.'</li><li>'The telescope captured images of a distant galaxy.'</li><li>'The lab experiment yielded unexpected results.'</li></ul> |
| education     | <ul><li>'The university offers a new degree program.'</li><li>'Students are preparing for their final exams.'</li><li>'The teacher assigned homework for the weekend.'</li></ul>                |
| environment   | <ul><li>'The forest fire destroyed acres of land.'</li><li>'Pollution levels in the city have decreased.'</li><li>'Renewable energy sources are being adopted.'</li></ul>                       |
| travel        | <ul><li>'The flight to Paris was delayed by two hours.'</li><li>'We booked a hotel room with a sea view.'</li><li>'The tour guide explained the history of the castle.'</li></ul>               |
| food          | <ul><li>'The restaurant serves authentic Italian cuisine.'</li><li>'The recipe requires fresh ingredients.'</li><li>'The chef prepared a delicious three-course meal.'</li></ul>                |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("The stock market crashed today.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 5   | 6.7077 | 9   |

| Label         | Training Sample Count |
|:--------------|:----------------------|
| business      | 5                     |
| education     | 5                     |
| entertainment | 5                     |
| environment   | 5                     |
| finance       | 5                     |
| food          | 5                     |
| health        | 5                     |
| law           | 5                     |
| politics      | 5                     |
| science       | 5                     |
| sports        | 5                     |
| technology    | 5                     |
| travel        | 5                     |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0061 | 1    | 0.2153        | -               |
| 0.3067 | 50   | 0.1436        | -               |
| 0.6135 | 100  | 0.0538        | -               |
| 0.9202 | 150  | 0.0353        | -               |

### Framework Versions
- Python: 3.11.3
- SetFit: 1.1.3
- Sentence Transformers: 3.4.1
- Transformers: 4.48.3
- PyTorch: 2.9.0+cpu
- Datasets: 4.4.1
- Tokenizers: 0.21.0

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->