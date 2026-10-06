# VO₂ and VO₂max Prediction Model

Part of [VO₂ & VO₂max Estimation](https://github.com/Sivanesh27/VO2-Estimation). This module builds physiological and movement features from reliable segments and predicts **VO₂max** (per subject) and **VO₂(t)** (over time).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xG61bwV37L5DtPnkZ7q37d2pUNbLI3i_?usp=sharing)  ·  [Project document](https://docs.google.com/document/d/1iVSv0T32S8UnDJSDZ31XoRWubMcw7VvJZ_CcjniTnN4/edit?usp=drive_link)  ·  [Back to main README](https://github.com/Sivanesh27/VO2-Estimation#readme)

> **Prerequisite:** run the [Reliable Data Extraction](https://github.com/Sivanesh27/VO2-Estimation/tree/main/Reliable%20Data%20Extraction) step first.

---

## Approach in one line

```
Predicted VO₂max = Uth VO₂max + Ridge-predicted residual
VO₂(t)           = f(VO₂max, %HRR(t))
```

A known physiological formula does the heavy lifting; a small regularised ML model only learns the **correction**. This suits a dataset of ~10 labelled subjects.

## Features (~11 subject-level)

| Group | Feature | Definition / note |
|---|---|---|
| Physiological baseline | Uth VO₂max | `15.3 × HRmax / HRrest` |
| HR characteristics | HRmax | Observed high-HR distribution; Tanaka reference `208 − 0.7 × age` |
| | HRrest | Readiness/resting HR, or estimated from reliable readiness data |
| | HRR and %HRR | `HRmax − HRrest`; `(HR − HRrest) / (HRmax − HRrest)` |
| Movement response | RMS features | High movement, low movement, HR response to movement |
| | HR–movement slope | How strongly HR rises with movement |
| Recovery | 60 s HR recovery | After sufficiently intense activity |
| HRV | `ln(RMSSD)` | From readiness data; log reduces skew |
| Demographics | Age, sex, BMI | BMI from height and weight |

## Models

| Model | Role |
|---|---|
| **Uth physiological model** | Baseline, no training needed |
| **Linear Regression** calibration | Simple calibration of the baseline |
| **Ridge Regression residual model** ⭐ | Preferred: regularised, learns the correction to Uth |

**Why Ridge?** With ~10 subjects and ~11 features, an unregularised model overfits. Ridge shrinks coefficients and is far more stable.

**Why not TCN?** Only ~10 independent labelled subjects and no continuous time-aligned VO₂ ground truth, so a deep temporal network would overfit. The problem is framed as:

```
HR + Movement + HRV + Demographics  →  VO₂max
VO₂max + %HRR                       →  VO₂(t)
```

## Validation: Leave-One-Person-Out (LOPO)

Each subject is held out entirely once; the model trains on the other 9 and predicts the held-out person. This measures **generalisation to a new person** and avoids leakage from sessions of the same individual.

## Results

Mean absolute difference across the 10 subjects ≈ **5.82 ml/kg/min**.

| Athlete | Target VO₂max (ml/kg/min) | Predicted VO₂max (ml/kg/min) | Difference | Sequential VO₂ data | Full model output |
|---|:---:|:---:|:---:|:---:|:---:|
| Ajay D | 59.4 | 52.5 | 6.9 | [Click here](https://drive.google.com/file/d/1hpUqVKVhJk3HxHU1-1p4VG6aDynAIsUA/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/18jrJb2qIHER-UKGIEdYaWNWB14HJqufH?usp=drive_link) |
| Akash Raj | 57 | 52.6 | 4.4 | [Click here](https://drive.google.com/file/d/1Wgua9gPGSBkZBBhXsntipxCmiOxxNVbJ/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/1LPkzHoNhJMSDTP3m7qFUEhVIgq7fPMpf?usp=drive_link) |
| Yuvan Shravan S | 42.2 | 53 | 10.8 | [Click here](https://drive.google.com/file/d/1uwRKNEDd68lThVN_38_AwsZ9bM-hXH80/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/1MaZ527gmqu2CBybT-zF7EgHYKxbEH1De?usp=drive_link) |
| Gokul Krishna | 47.1 | 52.7 | 5.6 | [Click here](https://drive.google.com/file/d/1H7FpbpEbvvujLGSaFNNHn1SBG1vyWAfj/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/1eLpiW_lnzjfPXcWYqFlYyxDTYfTbAS3-?usp=drive_link) |
| Jayaraman T | 58.8 | 52.6 | 6.2 | [Click here](https://drive.google.com/file/d/1vzLFSt_rXqVCpAPScyhXQs9jd47-e9Oc/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/1IcjgFHz3bzZOVv9mU7Xr_8-YvKhRrFop?usp=drive_link) |
| Manoj Kumar S | 45.4 | 52.7 | 7.3 | [Click here](https://drive.google.com/file/d/16XLt64J1Aj493ORuzP3EGkmhlSwTGpSQ/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/1XaI4VbBGNZ67LnSI7AA-cB6lfbZFOq2i?usp=drive_link) |
| Mithra M R | 50 | 53.1 | 3.1 | [Click here](https://drive.google.com/drive/folders/13RCPDtwZzrXJEs6uGdym1-KctiPcyBZK?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/13RCPDtwZzrXJEs6uGdym1-KctiPcyBZK?usp=drive_link) |
| Nirmal Kumar | 61.5 | 53.6 | 7.9 | [Click here](https://drive.google.com/file/d/1qV5f9PA_miflq4ErYEZ2OuwQy1ciFiRz/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/1hhIm4BAbVBCRgJAEoW9ha4xZI5owLJY_?usp=drive_link) |
| Sarabeshwar L | 50.6 | 52.9 | 2.3 | [Click here](https://drive.google.com/drive/folders/1X-CkOg5E6-bW_YsAAZAUPw_SZEde6SqY?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/1X-CkOg5E6-bW_YsAAZAUPw_SZEde6SqY?usp=drive_link) |
| Vignesh V | 56.3 | 52.6 | 3.7 | [Click here](https://drive.google.com/file/d/1jXQVhFvwyEEL7A9-vd_3EeuxP0NLOWCs/view?usp=drive_link) | [Click here](https://drive.google.com/drive/folders/11i2aO6KYR88_25blDmnMCEVcUcAEKx7-?usp=drive_link) |

**Reading these results honestly:** predictions fall in a narrow range (≈ 52.5–53.6) while targets range from 42.2 to 61.5. The model is near the cohort mean for everyone, so it under-predicts high-fitness athletes (e.g. Nirmal Kumar, Jayaraman T, Ajay D) and over-predicts lower-fitness ones (e.g. Yuvan Shravan S). More subjects and stronger fitness-discriminating features are the next steps.

## How to run

**Colab:** open the [prediction notebook](https://colab.research.google.com/drive/1xG61bwV37L5DtPnkZ7q37d2pUNbLI3i_?usp=sharing), upload the extracted segments and the subject table (age, sex, height, weight, VO₂max label), then run all cells.

**Locally:**

```bash
git clone https://github.com/Sivanesh27/VO2-Estimation.git
cd VO2-Estimation
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib jupyter
jupyter notebook
```

Open the notebook in the `VO2 and VO2max prediction model` folder and run all cells.

## Outputs

- Predicted VO₂max per subject (with LOPO evaluation)
- Sequential VO₂(t) series per subject
- Full model prediction outputs (see the table links above)

## Related

- [Main README](https://github.com/Sivanesh27/VO2-Estimation#readme)
- [Reliable Data Extraction README](https://github.com/Sivanesh27/VO2-Estimation/tree/main/Reliable%20Data%20Extraction)
- [Project document](https://docs.google.com/document/d/1iVSv0T32S8UnDJSDZ31XoRWubMcw7VvJZ_CcjniTnN4/edit?usp=drive_link)
