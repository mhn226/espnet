<div align="left"><img src="doc/image/espnet_logo1.png" width="550"/></div>

# ESPnet: end-to-end speech processing toolkit

### This is a development branch of EPSnet, specifically dedicating to the work of the [ON-TRAC Consortium*'s](https://on-trac.univ-avignon.fr/on-trac-consortium/) submission to [INTERSPEECH 2020](http://www.interspeech2020.org/).

## Investigating Self-supervised Pre-training for End-to-end Speech Translation
Ha Nguyen, Fethi Bougares, Natalia Tomashenko, Yannick Estève, Laurent Besacier
> **Abstract:** Self-supervised learning from raw speech has been proven beneficial to improve automatic speech recognition (ASR). We investigate here its impact on end-to-end automatic speech translation (AST) performance. We use a contrastive predictive coding (CPC) model pre-trained from unlabeled speech as a feature extractor for a downstream AST task. We show that self-supervised pre-training is particularly efficient in low resource settings and that fine-tuning CPC models on the AST training data further improves performance. Even in higher resource settings, ensembling AST models trained with filter-bank and CPC representations leads to near state-of-the-art models without using any ASR pre-training. This might be particularly beneficial when one needs to develop a system that translates from speech in a language with poorly standardized orthography or even from speech in an unwritten language.

*ON-TRAC Consortium is composed of researchers from three French academic laboratories: LIA (Avignon Université), LIG (Université Grenoble Alpes), and LIUM (Le Mans Université).

## Updated features
In this work , we made the following modifications to ESPnet v.0.6.3:
### ST: Speech Translation & MT: Machine Translation
- [**_wav2vec_**](https://arxiv.org/abs/1904.05862) feature extraction
- **_wav2vec_** speech encoder
- Ensembling ST models at decoding time

## Results and demo
### INTERSPEECH 2020 ST results

#### end-to-end system
##### Results on How2 EN-PT (BLEU scores on How2 val set)
| No. | Features | 10% (28h) | 20% (56h) | 30% (84h) |  60% (169h) |  100% (281h) 
| :----: | ---- | :----: | :----: | :----: | :----: | :----: |
| 1 | wav2vec |  11.33 | 26.75 | 30.83 | 36.33 | 41.02 |
| 2 | wav2vec + FT | 12.52 | 27.30 | 32.11 | 37.78 | 42.32 |
| 3 | wav2vec + norm | 16.52 | 27.33 | 31.27 | 37.62 | 41.08 |
| 4 | wav2vec + FT + norm | 18.50 | 27.68 | 32.17 | 37.75 | 41.30 |
| 5 | fbanks | 1.03 | 18.61 | 27.32 | 37.23 | 41.63 |
| 6 | fbanks + norm | 2.11 | 24.58 | 30.21 | 37.56 | 42.51 |
| 7 | Ensemble [5, 6] | | 25.28 | 31.90 | 40.39 | 44.35 |
| 8 | Ensemble [4, 6] | | 29.87 | 34.67 | 41.22 | 45.02 |
| 9 | Ensemble [1,2,3,4,5,6] | | 31.88 | 36.80 | 42.62 | 46.16 |

##### Results on MuST-C EN-DE 56 hours (BLEU)
| No. | Features | MuST-C tst-COMMON | MuST-C tst-HE |
| :----: | ---- | :----: | :----: |
| 1 | wav2vec | 7.56 | 7.21 |
| 2 | wav2vec + norm | 7.83 | 8.12 |
| 3 | fbanks | 1.50 | 1.09 |
| 4 | fbanks+norm | 4.89 | 4.87 |

##### Results on MuST-C EN-FR 56 hours (BLEU)
| No. | Features | MuST-C tst-COMMON | MuST-C tst-HE |
| :----: | ---- | :----: | :----: |
| 1 | wav2vec | 12.08 | 12.41 |
| 2 | wav2vec + norm | 12.58 | 12.58 |
| 3 | fbanks | 0.54 | 0.00 |
| 4 | fbanks + norm | 7.10 | 6.37 |

### INTERSPEECH 2020 phone discrimination results
##### Results on TIMIT (Phone Error Rate (PER%))
| No. | Features | TIMIT dev | TIMIT test |
| :----: | ---- | :----: | :----: |
| 1 | wav2vec | 13.0 | 15.0 |
| 2 | wav2vec + norm | 13.9 | 15.8 |
| 3 | fbanks | 22.2 | 24.9 |
| 4 | fbanks + norm | 20.7 | 23.5 | 

### ST demo
Please navigate yourself to this [directory](https://github.com/mhn226/espnet/tree/interspeech2020/egs/interspeech2020/) for our detailed recipes.


## Acknowledgements
This work was funded by the French Research Agency (ANR) through the ON-TRAC project under contract number ANR-18-CE23-0021.
