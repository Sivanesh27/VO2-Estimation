# VO2 / VO2max from HR + motion data

Works with BOTH layouts automatically:
1. Your raw export folders:  dataset\<Person>\  ->  *_Corrected-HR-Training-Sessions_*.zip, *_Accelerometer-RMS_*.zip,
   *_HR-Quality-*.zip, *_Corrected-HR-Readiness-Sessions_*.zip, *_Readiness-Metrics_*.csv, *_Training-Metrics_*.csv
2. The earlier "reliable_dataset" layout (training\ and readiness\ folders, or a zip of them)

Missing files are fine (e.g. a person with no training files or no readiness files). The script says what was missing.

## 0. Setup (once)
    pip install -r requirements.txt

Put everything in one folder:

    VO2 and VO2max prediction model\
        vo2_pipeline.py
        labels_template.csv        <- your labels (person_id must equal the dataset sub-folder name)
        dataset\Ajay D\ ... dataset\Yuvan Shravan S\

Open a terminal in that folder (VS Code: Terminal > New Terminal; Colab: put `!` in front of each command).

## 1. Train + validate + save metrics (one command)
    python vo2_pipeline.py train --data dataset --labels labels_template.csv --out model_output\vo2_model.joblib

What happens:
- every person is loaded, unreliable samples are filtered, features are built
- the model is validated with Leave-One-Person-Out: each person is predicted by a model that never saw them
  (this is the honest "test" score; with 10 people a normal train/test split would be meaningless)
- the best model is chosen by that score, retrained on everyone, and saved

Saved in model_output\ :
- `vo2_model.joblib`                the model you reuse
- `reports\metrics.json`            MAE / RMSE / R2 / bias for every candidate model + settings
- `reports\cv_metrics.csv`          the same as a table
- `reports\cv_predictions.csv`      true vs predicted VO2max for every person
- `reports\cv_scatter.png`          true vs predicted plot

How to read it: compare the best model's RMSE with the printed "always-predict-the-mean" RMSE.
If it is not clearly lower, the model is not adding information.

### Check suspicious labels
The script prints a per-person error table and warns when one person is far off. In your data that is
**Nirmal Kumar** (resting HR 89 but VO2max 61.5; the sensor data points to roughly 32-40). Verify that label.
If it is wrong, retrain without him:

    python vo2_pipeline.py train --data dataset --labels labels_template.csv --out model_output\vo2_model.joblib --exclude "Nirmal Kumar"

## 2. Predict (VO2max + VO2 curve) for people in this format
All people in a folder, using age / resting HR / sex / weight from the labels file:

    python vo2_pipeline.py predict --input dataset --model model_output\vo2_model.joblib --profile labels_template.csv --out results

One new person (a folder in the same format):

    python vo2_pipeline.py predict --input "D:\new\Person X" --model model_output\vo2_model.joblib --age 23 --resting-hr 52 --out results

Per person you get `prediction.json`, `session_summary.csv`, `vo2_timeseries.csv` (VO2 every 3 s), `example_session.png`.
Plus `results\all_predictions.csv`.
If VO2max is already known (lab test) add `--vo2max 55` to skip the model and just get the VO2 curve.

## 3. A single recorded HR file
Any CSV with `timestamp,hr[,rms]` (raw export files with a `value` column also work):

    python vo2_pipeline.py predict-csv --input session.csv --model model_output\vo2_model.joblib --age 23 --resting-hr 52

## 4. Real time
```python
import joblib
from vo2_pipeline import LiveVO2

bundle = joblib.load(r"model_output\vo2_model.joblib")

# Option A: calibrate from the person's past folder (predicts their VO2max once)
est = LiveVO2.from_model(bundle, r"dataset\Ajay D", {"age": 23, "resting_hr": 51})
# Option B: numbers you already know
# est = LiveVO2(resting_hr=51, hrmax=189, vo2max=57)

for hr in heart_rate_stream():       # your Movesense / BLE loop, one reading per sample
    vo2, pct_hrr = est.update(hr)    # VO2 in ml/kg/min, pct_hrr 0..1
    print(round(vo2, 1))
```
Test it without a sensor by replaying a file:

    python vo2_pipeline.py live-demo --input session.csv --resting-hr 51 --hrmax 189 --vo2max 57

## Colab
    from google.colab import files; files.upload()      # upload vo2_pipeline.py + labels_template.csv
    # upload the dataset folder (or a zip of it, then !unzip -q dataset.zip)
    !pip -q install scikit-learn joblib matplotlib
    !python vo2_pipeline.py train --data dataset --labels labels_template.csv --out model_output/vo2_model.joblib

## Limits to keep in mind
- VO2max accuracy is bounded by 9-10 labelled people. Treat results as about +/- 5 ml/kg/min.
- VO2 (instantaneous) is derived from HR via %HRR, not learned; it has no ground truth here and overestimates at low intensity.
- Add more labelled people (and measured VO2max) and re-run train; the learned models start to help at roughly 30+ people.
