# Synthetic Tabular Detection Audit

This audit is inspired by Kindji et al., `Cross-table Synthetic Tabular Data Detection`, arXiv:2412.13227.

Important limitation: the paper reports that cross-table synthetic tabular detection is challenging. For the simple character-trigram logistic-regression baseline, the reported cross-table AUC is about 0.58. Therefore this script provides risk indicators, not a proof that the Kaggle dataset is synthetic.

## Inputs

- Target rows used: 2000
- Reference real rows used: 2000
- Reference real path: `balanced_passenger_survey_dataset/passenger_survey_balanced.csv`

## Detector Results

| View | Setup | Generator | AUC | Separability AUC | Accuracy | Interpretation |
|---|---|---|---:|---:|---:|---|
| full_content | same_table_c2st | independent_marginals | 0.4095 | 0.5905 | 0.4450 | Same-table C2ST. High AUC means the script can distinguish the Kaggle rows from a simple synthetic control generated from the same table. This does not prove the Kaggle rows are real; it only checks whether naive synthetic controls have detectable artifacts. |
| full_content | same_table_c2st | column_permutation | 0.2193 | 0.7807 | 0.2967 | Same-table C2ST. High AUC means the script can distinguish the Kaggle rows from a simple synthetic control generated from the same table. This does not prove the Kaggle rows are real; it only checks whether naive synthetic controls have detectable artifacts. |
| full_content | same_table_c2st | numeric_smoothed_marginals | 1.0000 | 1.0000 | 1.0000 | Same-table C2ST. High AUC means the script can distinguish the Kaggle rows from a simple synthetic control generated from the same table. This does not prove the Kaggle rows are real; it only checks whether naive synthetic controls have detectable artifacts. |
| full_content | cross_table_reference_detector | all_simple_controls | 0.4469 | 0.5531 | 0.4871 | Cross-table detector inspired by arXiv:2412.13227. It is trained on a reference real passenger survey table and simple synthetic controls, then applied to the Kaggle table. Because the paper reports weak cross-table performance for the 3-gram logistic baseline (AUC about 0.58 under cross-table shift), this score is only a risk indicator, not proof of synthetic origin. |
| feature_only | same_table_c2st | independent_marginals | 0.4064 | 0.5936 | 0.4400 | Same-table C2ST. High AUC means the script can distinguish the Kaggle rows from a simple synthetic control generated from the same table. This does not prove the Kaggle rows are real; it only checks whether naive synthetic controls have detectable artifacts. |
| feature_only | same_table_c2st | column_permutation | 0.2466 | 0.7534 | 0.3108 | Same-table C2ST. High AUC means the script can distinguish the Kaggle rows from a simple synthetic control generated from the same table. This does not prove the Kaggle rows are real; it only checks whether naive synthetic controls have detectable artifacts. |
| feature_only | same_table_c2st | numeric_smoothed_marginals | 1.0000 | 1.0000 | 1.0000 | Same-table C2ST. High AUC means the script can distinguish the Kaggle rows from a simple synthetic control generated from the same table. This does not prove the Kaggle rows are real; it only checks whether naive synthetic controls have detectable artifacts. |
| feature_only | cross_table_reference_detector | all_simple_controls | 0.4469 | 0.5531 | 0.4871 | Cross-table detector inspired by arXiv:2412.13227. It is trained on a reference real passenger survey table and simple synthetic controls, then applied to the Kaggle table. Because the paper reports weak cross-table performance for the 3-gram logistic baseline (AUC about 0.58 under cross-table shift), this score is only a risk indicator, not proof of synthetic origin. |

## Cross-table Target Scores

### full_content

- Target mean oriented synthetic probability: 0.3357
- Target median oriented synthetic probability: 0.2839
- Target mean raw synthetic probability: 0.6643
- Holdout separability AUC: 0.5531
- Orientation flipped: True
- Reference real holdout mean: 0.4787645203024892
- Reference synthetic holdout mean: 0.500799641676845

### feature_only

- Target mean oriented synthetic probability: 0.3766
- Target median oriented synthetic probability: 0.3325
- Target mean raw synthetic probability: 0.6234
- Holdout separability AUC: 0.5531
- Orientation flipped: True
- Reference real holdout mean: 0.4787645203024892
- Reference synthetic holdout mean: 0.500799641676845

## How to Use This Result

A high score should be described as an additional warning signal, especially together with missing dataset provenance. It should not be written as definitive evidence that the Kaggle dataset is fake.