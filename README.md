# planning-classifier

Classifies new house building applications from planning application proposal records, and estimates how many new homes are proposed.

- **Stage 1, is it house building?** TF-IDF features and logistic regression, trained on the proposal text of manually checked rows.
- **Stage 2, how many homes?** Regex rules in `n_homes.py`, which read the number from the proposal text. The result is mapped to 11 classes: `1` to `10` and `more than 10`.

## Setup

```
conda env create -f environment.yml
conda activate planning-classifier
```

## Usage

### Train

Prints cross-validation results, then fits on all checked rows and saves the model.

```
python train.py --input inputs/planning.xlsx --model-output models/housebuilding.joblib --oof-output outputs/train_oof.csv
```

Optional arguments are `--threshold`, `--folds` and `--seed`. The threshold is saved with the model. A `.json` file beside the model records the input file hash, row counts, CV metrics and library versions.

### Predict

```
python predict.py --input "inputs/Planning_Applications__Official_PREMIUM_-_Scotland.csv" --model models/housebuilding.joblib --output outputs/scotland_pred.csv
```
Predictions are only made on a subset of all planning application records. The filter used is specified in `config.py`. 

The filter keeps rows where `appl_desc_is` is one of `APPLICATION_TYPES` and `dev_desc_is` is one of `DEVELOPMENT_TYPES` or blank (see `config.py`). Use `--no-filter` to predict on every row. Other optional arguments are `--threshold`, `--text-col` and `--keep-cols`.

Writes one row per application that passes the filter, with `pred_prob`, `is_housebuilding`, `n_homes`, `n_homes_class` and `n_homes_method`. The `n_homes` columns are blank where `is_housebuilding` is 0.

The same large site can appear in several applications (conditions, s42 and so on), so summing `n_homes` double counts.

### Validate

Compares a predictions file with another labelled copy of the same rows, joined on an ID column.

```
python validate.py --predictions outputs/scotland_pred.csv --labels "inputs/Planning_Applications__Official_PREMIUM_-_Scotland_labelled.csv" --train-input inputs/planning.xlsx --errors-output outputs/validation_errors.csv
```

Prints stage 1 scores, a threshold table and stage 2 accuracy. `--train-input` adds scores that exclude text seen in training. If your labels use other column names, set `--id-col`, `--label-col` and `--n-homes-col`.

The predictions file must include the predicted negatives, otherwise missed house building can't be counted.

### Analysis

`house_building.ipynb` explores the files written by the commands above, so run those first.

## Folders

| Folder | Contents | Tracked by git? |
|---|---|---|
| `inputs/` | Source planning data and labelled records | No |
| `models/` | Trained models and their metadata | No |
| `outputs/` | Predictions, out-of-fold predictions and validation errors | No |

## Labels

- `planning.xlsx`: rows checked by a human reviewer. These are the training data.
- `..._Scotland_labelled.csv`: labelled by LLM from the proposal text, following the `planning.xlsx` conventions. It wasn't used in training, but it is the currently available comparison for validate.
