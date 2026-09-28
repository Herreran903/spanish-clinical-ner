# Spanish Clinical NER

Named entity recognition over Spanish clinical text, approached twice: once with a
BiLSTM-CRF trained from scratch, and once by fine-tuning Spanish transformers with LoRA
adapters. Built for the Natural Language Processing course (750108M) at Universidad del
Valle, 2025.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers%20%2B%20PEFT-FFD21E.svg)](https://huggingface.co/docs/peft)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%20%2B%20CRF-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/model-beto__prostata__peft-blue.svg)](https://huggingface.co/NicolasUnivalle/beto_prostata_peft)

**Joint work with [John Freddy Belalcázar](https://github.com/JohnFredd).** Both of us
worked on every part; the notebooks are published here with his agreement.

## Two approaches, two corpora

| | Corpus | Model | Why |
|---|---|---|---|
| **Part 1** | [CodiEsp](https://zenodo.org/records/3837305) — public, SciELO clinical cases | BiLSTM + CRF, Word2Vec trained in-domain | A neural baseline with no pretrained language model, so the transformer has something to beat |
| **Part 2** | Prostate oncology reports, 10 entity types | BETO and XLM-RoBERTa, fine-tuned with LoRA | What pretrained Spanish language knowledge is actually worth |

## Results

Evaluated with `seqeval`, at entity level rather than token level. A boundary off by one
token is not the same mistake as a wrong entity type, and token-level accuracy hides that.

### Public benchmark, CoNLL-2002 Spanish

| Model | Precision | Recall | F1 | Epochs | Batch |
|---|---|---|---|---|---|
| BETO | 83.1% | 85.2% | 84.1% | 2 | 16 |
| XLM-RoBERTa | 87.0% | 88.0% | **87.7%** | 1 | 4 |

### Prostate corpus, LoRA fine-tuning

Read the macro column. **Macro F1 averages the ten entity types equally; weighted F1 is
dominated by the `O` tag**, which is most of any NER corpus and is trivially easy. The
weighted figure is reported because it is what the raw `classification_report` prints
first, not because it is the number that matters.

| Model | **Macro F1** | Weighted F1 | Support |
|---|---|---|---|
| BETO + LoRA | **0.905 – 0.933** | 0.960 – 0.971 | 29,217 tokens (test) |
| XLM-RoBERTa + LoRA | not recovered from the saved run | 0.982 – 0.988 | — |

Both models were swept over a grid of 3, 4 and 5 epochs against batch sizes 4, 8, 16 and
32, with precision, recall and F1 recorded separately on validation and on test for every
cell. Best results land at 5 epochs.

### Entity schema

Ten entity types, 21 BIO classes:

`BIOMARCADOR` · `CANCER` · `CIRUGIA` · `DOSIS` · `EDAD` · `FECHA` · `GLEASON` ·
`MEDICAMENTO` · `TNM` · `TRATAMIENTO`

`GLEASON` and `TNM` are prostate cancer grading and staging scales, so the schema
encodes real clinical structure rather than generic person-place-organization tags.

## Method

**BiLSTM-CRF.** `Embedding` → `SpatialDropout1D` → `Bidirectional LSTM` → CRF layer from
`tensorflow_addons`. The CRF log-likelihood replaces per-token cross-entropy as the
objective, which is what lets the model learn that `I-CANCER` cannot follow `O`.
Embeddings are Word2Vec vectors trained with gensim on the corpus itself rather than
downloaded, so the vocabulary matches the domain.

**LoRA fine-tuning.** `LoraConfig(task_type=TOKEN_CLS, r=8, lora_alpha=16)` over
`AutoModelForTokenClassification`, with `DataCollatorForTokenClassification` for dynamic
padding and subword-to-word label alignment. Only the adapter is trained, so a full
sweep fits in a Colab session.

## Models

| | Hugging Face | Owner |
|---|---|---|
| BETO + prostate + LoRA | [`NicolasUnivalle/beto_prostata_peft`](https://huggingface.co/NicolasUnivalle/beto_prostata_peft) | Nicolás |
| XLM-RoBERTa + prostate + LoRA | [`JohnFreddy/roberta_prostata_peft`](https://huggingface.co/JohnFreddy/roberta_prostata_peft) | John |

## Data

**CodiEsp is not redistributed here.** It is CC-BY 4.0 and freely available, so download
it from Zenodo rather than from a copy that may drift out of date:

- Corpus: https://zenodo.org/records/3837305
- Guidelines: https://zenodo.org/records/4121205

> Miranda-Escalada, A., Gonzalez-Agirre, A., Armengol-Estapé, J. and Krallinger, M.
> *Overview of automatic clinical coding: annotations, guidelines, and solutions for
> non-English clinical cases at CodiEsp track of CLEF eHealth 2020.* CC-BY 4.0,
> © Secretaría de Estado para el Avance Digital, 2019.

**The prostate corpus is not published.** It is clinical material and is not ours to
release. The notebooks show the full pipeline over it; reproducing the numbers requires
your own annotated set in the same BIO format.

## Layout

```
notebooks/
  01-bilstm-crf-codiesp.ipynb          BiLSTM-CRF, corpus loading, experiment harness
  02-bilstm-crf-codiesp-word2vec.ipynb the same with in-domain Word2Vec embeddings
  03-beto-lora-prostate.ipynb          BETO fine-tuned with LoRA, full metric sweep
  04-xlm-roberta-lora-prostate.ipynb   XLM-RoBERTa fine-tuned with LoRA
```

Written for Colab: they install their own dependencies and read from a mounted Drive.
Paths need adjusting to run elsewhere.

## Scope

Coursework, not a library. It exists to compare approaches on the same problem and to
keep that comparison legible.
