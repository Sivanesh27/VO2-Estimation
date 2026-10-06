#!/usr/bin/env python3
"""
vo2_pipeline.py  -  Estimate VO2 (time series) and VO2max (per person)
from Movesense-style HR + motion data laid out as:

    <person>/
        training/<session>/reliable_only.csv      (hr, rms, timestamp ...)
        readiness/<session>/reliable_only.csv     (hr, timestamp ...)
        readiness_validation.csv                  (optional, has rmssd)
    (a person can be a folder or a .zip of that folder)

HOW IT WORKS
------------
VO2max (one number per person)
    * Physiological prior: Uth et al. (2004)  VO2max = 15.3 * HRmax / HRrest
    * Learned correction: a heavily regularised Ridge model that predicts the
      *residual* (true VO2max - prior) from features such as resting HR,
      HRmax, HR response to motion, HR recovery, HRV and (optional) age/sex/BMI.
    * Trained on your labelled people, validated Leave-One-Person-Out (LOPO).

VO2 (time series, ml/kg/min)
    * No VO2 labels exist for individual time-points, so this is NOT learned.
      It uses the Swain relation  %VO2R ~= %HRR :
          VO2(t) = 3.5 + %HRR(t) * (VO2max - 3.5)
      anchored on the VO2max predicted above.

USAGE
-----
    python vo2_pipeline.py train   --data data/ --labels labels.csv --out model.joblib
    python vo2_pipeline.py predict --input new_person.zip --model model.joblib --out results/

labels.csv columns:  person_id, vo2max [, age, sex, weight_kg, height_cm]
person_id must equal the zip / folder name without ".zip".
"""
from __future__ import annotations

import argparse
import json
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    max_gap_s: float = 6.0            # gap that splits a continuous run of samples
    hr_smooth_win: int = 5            # rolling-median window (samples) -> 15 s
    hrmax_percentile: float = 99.5    # observed HRmax = this percentile of smoothed HR
    rest_percentile: float = 20.0     # resting HR = this percentile of readiness session means
    vo2_rest: float = 3.5             # ml/kg/min (1 MET)
    rms_high: float = 90.0            # "running-level" motion threshold (device units)
    rms_low: float = 20.0             # "low motion" threshold
    recovery_samples: int = 20        # 20 samples * 3 s = 60 s HR recovery window
    recovery_min_pct_hrr: float = 0.6 # peak must be >= 60 % HRR to count as an event
    recovery_max_rms: float = 40.0    # athlete must be fairly still after the peak
    min_recovery_events: int = 3
    n_augment: int = 20               # session-subsample copies per person for training
    seed: int = 42
    ridge_alphas: tuple = (0.1, 1, 10, 100, 1000, 10000)


CFG = Config()
FEATURES_PHYSIO = ["uth_vo2max", "rhr", "hrmax", "hrr_high", "hrr_low",
                   "hr_motion_slope", "hr_recovery_60", "ln_rmssd"]
FEATURES_PROFILE = ["age", "sex", "bmi"]
ALL_FEATURES = FEATURES_PHYSIO + FEATURES_PROFILE


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@dataclass
class Session:
    name: str
    df: pd.DataFrame  # columns: t(sec), hr, hr_s, run, [rms], timestamp


@dataclass
class Person:
    person_id: str
    training: list = field(default_factory=list)
    readiness: list = field(default_factory=list)
    rmssd: dict = field(default_factory=dict)   # readiness session -> rmssd
    rec60: dict = field(default_factory=dict)   # training session -> device "recovery_beats_60"
    notes: list = field(default_factory=list)   # data-availability messages


def _find_root(p: Path) -> Path:
    if (p / "training").is_dir() or (p / "readiness").is_dir():
        return p
    for sub in sorted(p.rglob("*")):
        if sub.is_dir() and ((sub / "training").is_dir() or (sub / "readiness").is_dir()):
            return sub
    raise FileNotFoundError(f"No 'training' or 'readiness' folder found under {p}")


def _prep_session(name: str, df: pd.DataFrame, cfg: Config) -> Session | None:
    if "hr" not in df or "timestamp" not in df:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "hr"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) < 3:
        return None
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["run"] = (df["t"].diff() > cfg.max_gap_s).cumsum()
    df["hr_s"] = df.groupby("run")["hr"].transform(
        lambda s: s.rolling(cfg.hr_smooth_win, center=True, min_periods=1).median())
    if "rms" in df:
        df["rms_s"] = df.groupby("run")["rms"].transform(
            lambda s: s.rolling(5, center=True, min_periods=1).median())
    return Session(name, df)


def _read_sessions(folder: Path, cfg: Config) -> list:
    out = []
    if not folder.is_dir():
        return out
    for sd in sorted(d for d in folder.iterdir() if d.is_dir()):
        f = sd / "reliable_only.csv"
        if f.exists():
            df = pd.read_csv(f)
        elif (sd / "reliability_flags.csv").exists():
            df = pd.read_csv(sd / "reliability_flags.csv")
            col = "reliable_final" if "reliable_final" in df else "reliable"
            df = df[df[col].astype(bool)]
        else:
            continue
        s = _prep_session(sd.name, df, cfg)
        if s is not None:
            out.append(s)
    return out


def _load_from_root(person_id: str, root: Path, cfg: Config) -> Person:
    p = Person(person_id)
    p.training = _read_sessions(root / "training", cfg)
    p.readiness = _read_sessions(root / "readiness", cfg)
    rv = root / "readiness_validation.csv"
    if rv.exists():
        v = pd.read_csv(rv)
        if {"session", "rmssd"} <= set(v.columns):
            p.rmssd = {str(a): float(b) for a, b in zip(v["session"], v["rmssd"]) if pd.notna(b)}
    return p



# ---- Raw export format: <person>/<Name>_Corrected-HR-Training-Sessions_*.zip, *_Accelerometer-RMS_*.zip, ... ----
RAW_KEYS = {"hr_train": "corrected-hr-training", "rms": "accelerometer-rms", "q_train": "hr-quality-training",
            "hr_ready": "corrected-hr-readiness", "q_ready": "hr-quality-readiness"}


def is_raw_person_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    names = " ".join(f.name.lower() for f in p.iterdir())
    return any(k in names for k in ("corrected-hr-", "training-metrics", "readiness-metrics"))


def _short(sid) -> str:
    return str(sid).split("__")[-1]


def _read_zip_csvs(path) -> pd.DataFrame:
    """All CSV members of a zip, concatenated. Empty / corrupt zips give an empty frame."""
    try:
        with zipfile.ZipFile(path) as z:
            frames = [pd.read_csv(z.open(n)) for n in z.namelist() if n.lower().endswith(".csv")]
    except (zipfile.BadZipFile, OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reliability_filter(df: pd.DataFrame, kind: str, cfg: Config) -> pd.DataFrame:
    """Keep only trustworthy samples (same spirit as the config.txt thresholds in your reliable_dataset).
    df columns: timestamp, hr, [hr_quality], [rms]."""
    hr_lo, hr_hi, spike = (40, 210, 10.0) if kind == "training" else (35, 120, 6.0)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    good = df["hr"].notna() & df["hr"].between(hr_lo, hr_hi)
    if "hr_quality" in df and df["hr_quality"].notna().any():
        good &= df["hr_quality"].fillna(0) >= 0.5
    med = df["hr"].rolling(5, center=True, min_periods=1).median()
    good &= (df["hr"] - med).abs() <= spike
    if "rms" in df and df["rms"].notna().sum() > 20:
        lr = np.log1p(df["rms"])
        mad = np.nanmedian(np.abs(lr - np.nanmedian(lr))) * 1.4826 + 1e-9
        good &= ~(((lr - np.nanmedian(lr)) / mad) > 5.0)
    df = df[good].reset_index(drop=True)
    if df.empty:
        return df
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    run = (t.diff() > cfg.max_gap_s).cumsum()
    dur = t.groupby(run).transform(lambda x: x.max() - x.min() + 3.0)   # 3 s sampling
    return df[dur >= 30.0].reset_index(drop=True)                        # min segment 30 s


def _raw_sessions(hr_zip, q_zip, rms_zip, kind: str, cfg: Config) -> list:
    hr = _read_zip_csvs(hr_zip) if hr_zip else pd.DataFrame()
    if hr.empty:
        return []
    q = _read_zip_csvs(q_zip) if q_zip else pd.DataFrame()
    r = _read_zip_csvs(rms_zip) if rms_zip else pd.DataFrame()
    out = []
    for sid, h in hr.groupby("session_id"):
        d = pd.DataFrame({"timestamp": pd.to_datetime(h["timestamp"], utc=True, errors="coerce"),
                          "hr": pd.to_numeric(h["value"], errors="coerce")}).dropna(subset=["timestamp"])
        d = d.sort_values("timestamp")
        for other, col, tol in ((q, "hr_quality", "35s"), (r, "rms", "2s")):
            if other.empty:
                continue
            o = other[other["session_id"] == sid]
            if o.empty:
                continue
            o = pd.DataFrame({"timestamp": pd.to_datetime(o["timestamp"], utc=True, errors="coerce"),
                              col: pd.to_numeric(o["value"], errors="coerce")}).dropna(subset=["timestamp"])
            d = pd.merge_asof(d, o.sort_values("timestamp"), on="timestamp", direction="nearest",
                              tolerance=pd.Timedelta(tol))
        d = reliability_filter(d, kind, cfg)
        s = _prep_session(_short(sid), d, cfg) if len(d) else None
        if s is not None:
            out.append(s)
    return out


def load_person_raw(folder: Path, cfg: Config = CFG) -> Person:
    folder = Path(folder)
    p = Person(folder.name)

    def find(key, ext):
        hits = sorted(f for f in folder.iterdir() if key in f.name.lower() and f.suffix.lower() == ext)
        return hits[0] if hits else None

    rms = find(RAW_KEYS["rms"], ".zip")
    p.training = _raw_sessions(find(RAW_KEYS["hr_train"], ".zip"), find(RAW_KEYS["q_train"], ".zip"), rms, "training", cfg)
    p.readiness = _raw_sessions(find(RAW_KEYS["hr_ready"], ".zip"), find(RAW_KEYS["q_ready"], ".zip"), None, "readiness", cfg)
    rm, tm = find("readiness-metrics", ".csv"), find("training-metrics", ".csv")
    if rm:
        v = pd.read_csv(rm)
        if {"session_id", "rmssd"} <= set(v.columns):
            p.rmssd = {_short(a): float(b) for a, b in zip(v["session_id"], v["rmssd"]) if pd.notna(b)}
    if tm:
        v = pd.read_csv(tm)
        if {"session_id", "recovery_beats_60"} <= set(v.columns):
            p.rec60 = {_short(a): float(b) for a, b in zip(v["session_id"], v["recovery_beats_60"]) if pd.notna(b) and b > 0}
    if not p.training:
        p.notes.append("no usable training sessions")
    if not p.readiness:
        p.notes.append("no usable readiness sessions")
    return p


def load_person(src, cfg: Config = CFG) -> Person:
    src = Path(src)
    pid = src.stem if src.suffix.lower() == ".zip" else src.name
    if is_raw_person_dir(src):
        return load_person_raw(src, cfg)
    if src.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            return _load_from_root(pid, _find_root(Path(tmp)), cfg)
    if not ((src / "training").is_dir() or (src / "readiness").is_dir()):
        zips = sorted(src.glob("*.zip"))
        if zips:                       # person folder that only holds the zip
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zips[0]) as z:
                    z.extractall(tmp)
                return _load_from_root(pid, _find_root(Path(tmp)), cfg)
    return _load_from_root(pid, _find_root(src), cfg)


def discover_people(path) -> list:
    """A zip, a person folder, or a folder containing many of those."""
    path = Path(path)
    if path.suffix.lower() == ".zip" or is_raw_person_dir(path):
        return [path]
    if (path / "training").is_dir() or (path / "readiness").is_dir():
        return [path]
    kids = [k for k in sorted(path.iterdir()) if k.suffix.lower() == ".zip" or k.is_dir()]
    if not kids:
        raise FileNotFoundError(f"No people found in {path}")
    return kids


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def _sex_to_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if s in ("m", "male", "1", "man"):
        return 1.0
    if s in ("f", "female", "0", "woman"):
        return 0.0
    return np.nan


def _resting_hr(readiness: list, training: list, cfg: Config) -> float:
    means = [s.df["hr"].mean() for s in readiness if len(s.df) >= 5]
    if len(means) >= 3:
        return float(np.percentile(means, cfg.rest_percentile))
    if means:
        return float(min(means))
    return float("nan")   # unknown: filled with the training-cohort median later (never guessed from exercise HR)


def _hrmax(training: list, age, cfg: Config) -> float:
    hr = np.concatenate([s.df["hr_s"].values for s in training]) if training else np.array([])
    if len(hr) == 0:
        return float(208 - 0.7 * age) if age and not np.isnan(age) else 190.0
    obs = float(np.percentile(hr, cfg.hrmax_percentile)) if len(hr) >= 200 else float(hr.max())
    if age is not None and not np.isnan(age):
        tanaka = 208.0 - 0.7 * age
        return max(obs, 0.5 * (obs + tanaka))   # never below what was actually seen
    return obs


def person_features(training: list, readiness: list, rmssd: dict, profile: dict | None,
                    cfg: Config = CFG, rec60: dict | None = None) -> dict:
    profile = profile or {}
    age = profile.get("age", np.nan)
    age = np.nan if age is None else float(age)
    given_rhr = profile.get("resting_hr", np.nan)
    if given_rhr is not None and not pd.isna(given_rhr):
        rhr = float(given_rhr)                       # user-supplied value wins
    else:
        rhr = _resting_hr(readiness, training, cfg)  # else derived from readiness sessions
    hrmax = _hrmax(training, age, cfg)
    rhr_for_calc = rhr if not np.isnan(rhr) else 55.0   # only for the %HRR features below; rhr itself stays NaN
    f = {"rhr": rhr, "hrmax": hrmax, "uth_vo2max": 15.3 * hrmax / max(rhr, 30.0) if not np.isnan(rhr) else np.nan}
    span = max(hrmax - rhr_for_calc, 20.0)

    hr_all, rms_all = [], []
    for s in training:
        if "rms_s" in s.df:
            hr_all.append(s.df["hr_s"].values)
            rms_all.append(s.df["rms_s"].values)
    if hr_all:
        hr_c, rms_c = np.concatenate(hr_all), np.concatenate(rms_all)
        pct = (hr_c - rhr_for_calc) / span
        hi, lo = rms_c >= cfg.rms_high, rms_c <= cfg.rms_low
        f["hrr_high"] = float(np.median(pct[hi])) if hi.sum() >= 30 else np.nan
        f["hrr_low"] = float(np.median(pct[lo])) if lo.sum() >= 30 else np.nan
        m = rms_c > 5
        f["hr_motion_slope"] = float(np.polyfit(np.log10(rms_c[m]), pct[m], 1)[0]) if m.sum() >= 100 else np.nan
    else:
        f["hrr_high"] = f["hrr_low"] = f["hr_motion_slope"] = np.nan

    # HR recovery over 60 s after the peak of a continuous run
    drops, n = [], cfg.recovery_samples
    for s in training:
        d = s.df
        if "rms_s" not in d:
            continue
        for _, h in d.groupby("run"):
            if len(h) < 30:
                continue
            hs, rs = h["hr_s"].values, h["rms_s"].values
            i = int(np.argmax(hs))
            if len(hs) - i <= n or (hs[i] - rhr_for_calc) / span < cfg.recovery_min_pct_hrr:
                continue
            if np.mean(rs[i + 1:i + n + 1]) > cfg.recovery_max_rms:
                continue
            drops.append(hs[i] - hs[i + n])
    f["hr_recovery_60"] = float(np.median(drops)) if len(drops) >= cfg.min_recovery_events else np.nan
    dev = [rec60[x.name] for x in training if rec60 and x.name in rec60]
    if len(dev) >= 3:                                  # device-computed recovery is cleaner than our fragments
        f["hr_recovery_60"] = float(np.median(dev))

    vals = [np.log(rmssd[s.name]) for s in readiness if s.name in rmssd and rmssd[s.name] > 0]
    f["ln_rmssd"] = float(np.median(vals)) if vals else np.nan

    h = profile.get("height_cm", np.nan)
    w = profile.get("weight_kg", np.nan)
    f["age"] = age
    f["sex"] = _sex_to_num(profile.get("sex"))
    f["bmi"] = (float(w) / (float(h) / 100) ** 2) if (w and h and not np.isnan(w) and not np.isnan(h)) else np.nan
    return f


def augmented_features(person: Person, profile: dict | None, cfg: Config, rng) -> pd.DataFrame:
    """Row 0 = all sessions. Remaining rows = random session subsets (robust to varying session counts)."""
    rows = [dict(person_features(person.training, person.readiness, person.rmssd, profile, cfg, person.rec60), is_full=1)]
    for _ in range(cfg.n_augment):
        def pick(lst):
            if len(lst) <= 1:
                return lst
            k = max(1, int(round(len(lst) * rng.uniform(0.4, 1.0))))
            return [lst[i] for i in rng.choice(len(lst), k, replace=False)]
        rows.append(dict(person_features(pick(person.training), pick(person.readiness),
                                         person.rmssd, profile, cfg, person.rec60), is_full=0))
    df = pd.DataFrame(rows)
    df["person_id"] = person.person_id
    return df


# ----------------------------------------------------------------------------
# Models (all share fit(X, y, groups) / predict(X))
# ----------------------------------------------------------------------------
class PriorOnly(BaseEstimator, RegressorMixin):
    """VO2max = 15.3 * HRmax / HRrest (no learning)."""
    def __init__(self, prior_idx=0):
        self.prior_idx = prior_idx
    def fit(self, X, y, groups=None):
        return self
    def predict(self, X):
        return np.asarray(X)[:, self.prior_idx]


class PriorCalibrated(BaseEstimator, RegressorMixin):
    """VO2max = a + b * prior."""
    def __init__(self, prior_idx=0):
        self.prior_idx = prior_idx
    def fit(self, X, y, groups=None):
        self.lr_ = LinearRegression().fit(np.asarray(X)[:, [self.prior_idx]], y)
        return self
    def predict(self, X):
        return self.lr_.predict(np.asarray(X)[:, [self.prior_idx]])


class ResidualModel(BaseEstimator, RegressorMixin):
    """prediction = prior + f(features); f is Ridge (alpha chosen by grouped CV) or a shallow forest."""
    def __init__(self, kind="ridge", prior_idx=0, alphas=CFG.ridge_alphas, seed=0):
        self.kind, self.prior_idx, self.alphas, self.seed = kind, prior_idx, alphas, seed
    def fit(self, X, y, groups=None):
        X = np.asarray(X, float)
        resid = np.asarray(y, float) - X[:, self.prior_idx]
        if self.kind == "ridge":
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
            n_groups = len(np.unique(groups)) if groups is not None else 0
            if n_groups >= 4:
                gs = GridSearchCV(pipe, {"ridge__alpha": list(self.alphas)},
                                  cv=GroupKFold(n_splits=min(5, n_groups)),
                                  scoring="neg_mean_squared_error")
                gs.fit(X, resid, groups=groups)
                self.model_ = gs.best_estimator_
            else:
                self.model_ = pipe.set_params(ridge__alpha=100.0).fit(X, resid)
        else:
            self.model_ = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(n_estimators=300, min_samples_leaf=15, max_features=0.6,
                                      random_state=self.seed, n_jobs=-1)).fit(X, resid)
        return self
    def predict(self, X):
        X = np.asarray(X, float)
        return X[:, self.prior_idx] + self.model_.predict(X)


def model_zoo(n_people: int, seed: int) -> dict:
    zoo = {"prior_uth": PriorOnly(), "prior_calibrated": PriorCalibrated(),
           "prior+ridge": ResidualModel("ridge", seed=seed)}
    if n_people >= 30:
        zoo["prior+forest"] = ResidualModel("forest", seed=seed)
    return zoo


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
            "bias": float(np.mean(err))}


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def read_labels(path) -> pd.DataFrame:
    lab = pd.read_csv(path)
    lab.columns = [c.strip().lower() for c in lab.columns]
    need = {"person_id", "vo2max"}
    if not need <= set(lab.columns):
        raise ValueError(f"labels file needs columns {need}; found {list(lab.columns)}")
    lab["person_id"] = lab["person_id"].astype(str).str.strip()
    return lab


def profile_of(lab_row) -> dict:
    row = {("resting_hr" if k in ("rhr", "resting_hr", "resting hr") else k): v for k, v in lab_row.items()}
    return {k: row[k] for k in ("age", "sex", "weight_kg", "height_cm", "resting_hr") if k in row and pd.notna(row[k])}


def train(data_dir, labels_path, out_path, cfg: Config = CFG, report_dir=None, exclude=()):
    lab = read_labels(labels_path).set_index("person_id")
    exclude = {e.strip() for e in (exclude or [])}
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for src in discover_people(data_dir):
        person = load_person(src, cfg)
        if person.person_id in exclude:
            print(f"  [excluded by you] {person.person_id}")
            continue
        if person.person_id not in lab.index:
            print(f"  [skip] {person.person_id}: no row in labels file")
            continue
        row = lab.loc[person.person_id]
        df = augmented_features(person, profile_of(row), cfg, rng)
        df["y"] = float(row["vo2max"])
        frames.append(df)
        full = df[df.is_full == 1].iloc[0]
        note = ("  <-- " + "; ".join(person.notes)) if person.notes else ""
        print(f"  loaded {person.person_id}: {len(person.training)} training / {len(person.readiness)} readiness "
              f"sessions | rhr={full.rhr:.1f} hrmax={full.hrmax:.1f} uth={full.uth_vo2max:.1f} | label={row['vo2max']}{note}")
    if len(frames) < 3:
        raise SystemExit("Need at least 3 labelled people to train.")
    D = pd.concat(frames, ignore_index=True)
    rhr_fill = float(D.loc[D.is_full == 1, "rhr"].median())          # people with no resting HR get the cohort median
    miss = D["rhr"].isna()
    if miss.any():
        print(f"  note: {D.loc[miss & (D.is_full == 1), 'person_id'].tolist()} have no resting HR -> filled with cohort median {rhr_fill:.1f}")
        D.loc[miss, "rhr"] = rhr_fill
        D.loc[miss, "uth_vo2max"] = 15.3 * D.loc[miss, "hrmax"] / rhr_fill
    feat_cols = [c for c in ALL_FEATURES if D[c].notna().any()]
    prior_idx = feat_cols.index("uth_vo2max")
    people = D["person_id"].unique()
    n = len(people)
    print(f"\n{n} labelled people, {len(D)} training rows (with session-subsample augmentation)")
    print("features used:", feat_cols)

    # ---- Leave-One-Person-Out validation
    zoo = model_zoo(n, cfg.seed)
    for m in zoo.values():
        if hasattr(m, "prior_idx"):
            m.prior_idx = prior_idx
    preds = {k: [] for k in zoo}
    truth = []
    for held in people:
        tr, te = D[D.person_id != held], D[(D.person_id == held) & (D.is_full == 1)]
        truth.append(te["y"].iloc[0])
        for name, proto in zoo.items():
            m = ResidualModel(**proto.get_params()) if isinstance(proto, ResidualModel) else type(proto)(**proto.get_params())
            m.fit(tr[feat_cols].values, tr["y"].values, groups=tr["person_id"].values)
            preds[name].append(float(m.predict(te[feat_cols].values)[0]))
    table = pd.DataFrame({k: metrics(truth, v) for k, v in preds.items()}).T.sort_values("RMSE")
    mean_base = metrics(truth, [np.mean(truth)] * n)   # reference: always predict the average
    print("\nLeave-One-Person-Out results (ml/kg/min):")
    print(table.round(3).to_string())
    print(f"(always-predict-the-mean reference: MAE={mean_base['MAE']:.2f}  RMSE={mean_base['RMSE']:.2f})")

    best = table.index[0]
    err = pd.DataFrame({"person": people, "true": truth, "pred_LOPO": np.round(preds[best], 1)})
    err["error"] = (err["pred_LOPO"] - err["true"]).round(1)
    print(f"\nPer-person leave-one-out error for '{best}' (largest first):")
    print(err.reindex(err["error"].abs().sort_values(ascending=False).index).to_string(index=False))
    ae = err["error"].abs().sort_values(ascending=False).values
    if len(ae) >= 4 and ae[0] > 2.0 * max(np.median(ae[1:]), 1.0):
        w = err.loc[err["error"].abs().idxmax(), "person"]
        print(f"\n!! '{w}' is far off the rest. Their label may be wrong or the test may not match the sensor data. "
              f"Check it, then optionally retrain with --exclude \"{w}\".")
    print(f"\nSelected model: {best}")
    final = model_zoo(n, cfg.seed)[best]
    if hasattr(final, "prior_idx"):
        final.prior_idx = prior_idx
    final.fit(D[feat_cols].values, D["y"].values, groups=D["person_id"].values)

    full = D[D.is_full == 1]
    fit_pred = final.predict(full[feat_cols].values)
    train_fit = metrics(full["y"].values, fit_pred)        # optimistic: model has seen these people
    bundle = {"model": final, "model_name": best, "feature_cols": feat_cols, "cfg": cfg,
              "cv_rmse": float(table.loc[best, "RMSE"]), "cv_table": table, "n_people": n,
              "train_medians": D[feat_cols].median().to_dict(), "rhr_fill": rhr_fill}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Saved model -> {out_path}")

    report_dir = Path(report_dir) if report_dir else Path(out_path).parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "cv_metrics.csv")
    per_person = pd.DataFrame({"person_id": people, "true_vo2max": truth,
                               **{f"pred_{k}": np.round(v, 2) for k, v in preds.items()}})
    per_person["abs_error_best"] = (per_person[f"pred_{best}"] - per_person["true_vo2max"]).abs().round(2)
    per_person.to_csv(report_dir / "cv_predictions.csv", index=False)
    summary = {"selected_model": best, "n_people": int(n), "features": feat_cols,
               "leave_one_person_out": table.round(4).to_dict(orient="index"),
               "reference_always_predict_mean": mean_base,
               "selected_model_fit_on_all_people_OPTIMISTIC": train_fit,
               "label_stats": {"mean": float(np.mean(truth)), "std": float(np.std(truth)),
                               "min": float(np.min(truth)), "max": float(np.max(truth))},
               "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}}
    (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(truth, preds[best])
        for pid, a_, b_ in zip(people, truth, preds[best]):
            ax.annotate(pid, (a_, b_), fontsize=7)
        lim = [min(truth + preds[best]) - 3, max(truth + preds[best]) + 3]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("True VO2max"); ax.set_ylabel("Predicted (person held out)")
        ax.set_title(f"Leave-one-person-out: {best}  RMSE={table.loc[best, 'RMSE']:.2f}")
        fig.tight_layout(); fig.savefig(report_dir / "cv_scatter.png", dpi=130); plt.close(fig)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"Metrics + plots saved -> {report_dir}")
    return bundle


# ----------------------------------------------------------------------------
# VO2 time series + prediction
# ----------------------------------------------------------------------------
def vo2_timeseries(session: Session, rhr: float, hrmax: float, vo2max: float, cfg: Config = CFG) -> pd.DataFrame:
    d = session.df
    pct = np.clip((d["hr_s"].values - rhr) / max(hrmax - rhr, 20.0), 0.0, 1.0)
    out = pd.DataFrame({"timestamp": d["timestamp"], "hr": d["hr_s"], "pct_hrr": pct,
                        "vo2_ml_kg_min": cfg.vo2_rest + pct * (vo2max - cfg.vo2_rest)})
    if "rms_s" in d:
        out["rms"] = d["rms_s"]
    out.insert(0, "session", session.name)
    return out


def predict_person(src, bundle, profile: dict | None = None, out_dir=None, vo2max_known=None) -> dict:
    person = src if isinstance(src, Person) else load_person(src, bundle["cfg"])
    return _predict(person, bundle, profile, out_dir, vo2max_known)


def _predict(person: Person, bundle, profile, out_dir, vo2max_known=None) -> dict:
    cfg = bundle["cfg"]
    feats = person_features(person.training, person.readiness, person.rmssd, profile, cfg, person.rec60)
    if np.isnan(feats["rhr"]):
        fill = bundle.get("rhr_fill", 55.0)
        print(f"  [{person.person_id}] no resting HR given and no readiness data -> using cohort median {fill:.0f} bpm "
              f"(pass --resting-hr / a resting_hr column for a better estimate)")
        feats["rhr"] = fill
        feats["uth_vo2max"] = 15.3 * feats["hrmax"] / fill
    cols = bundle["feature_cols"]
    x = np.array([[feats.get(c, np.nan) for c in cols]], float)
    missing = [c for c, v in zip(cols, x[0]) if np.isnan(v)]
    for i, c in enumerate(cols):                      # impute with training medians
        if np.isnan(x[0, i]):
            x[0, i] = bundle["train_medians"][c]
    if missing:
        print(f"  [{person.person_id}] features missing, filled with training median: {missing}")
    vo2max = float(vo2max_known) if vo2max_known is not None else float(bundle["model"].predict(x)[0])
    rmse = bundle["cv_rmse"]
    res = {"person_id": person.person_id, "vo2max_pred": round(vo2max, 1),
           "vo2max_range_1rmse": [round(vo2max - rmse, 1), round(vo2max + rmse, 1)],
           "rhr": round(feats["rhr"], 1), "hrmax": round(feats["hrmax"], 1),
           "uth_prior": round(feats["uth_vo2max"], 1), "model": bundle["model_name"]}

    weight = (profile or {}).get("weight_kg", np.nan)
    rows, series = [], []
    for s in person.training:
        ts = vo2_timeseries(s, feats["rhr"], feats["hrmax"], vo2max, cfg)
        series.append(ts)
        dur_min = (s.df["t"].iloc[-1] - s.df["t"].iloc[0]) / 60
        r = {"session": s.name, "reliable_minutes": round(len(s.df) * 3 / 60, 1), "span_minutes": round(dur_min, 1),
             "mean_hr": round(s.df["hr_s"].mean(), 1),
             "mean_vo2": round(ts["vo2_ml_kg_min"].mean(), 1),
             "peak_vo2_p95": round(ts["vo2_ml_kg_min"].quantile(0.95), 1),
             "mean_pct_vo2max": round(100 * ts["vo2_ml_kg_min"].mean() / vo2max, 1)}
        if weight and not np.isnan(weight):
            r["mean_kcal_per_min"] = round(ts["vo2_ml_kg_min"].mean() * weight / 1000 * 5.0, 2)
        rows.append(r)
    sess = pd.DataFrame(rows)
    res["n_training_sessions"] = len(person.training)

    if out_dir:
        od = Path(out_dir) / person.person_id
        od.mkdir(parents=True, exist_ok=True)
        sess.to_csv(od / "session_summary.csv", index=False)
        if series:
            pd.concat(series).to_csv(od / "vo2_timeseries.csv", index=False)
        (od / "prediction.json").write_text(json.dumps(res, indent=2))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if series:
                big = max(series, key=len)
                fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                tt = (pd.to_datetime(big["timestamp"]) - pd.to_datetime(big["timestamp"]).iloc[0]).dt.total_seconds() / 60
                ax[0].plot(tt, big["hr"], lw=0.8); ax[0].set_ylabel("HR (bpm)")
                ax[1].plot(tt, big["vo2_ml_kg_min"], lw=0.8, color="tab:red")
                ax[1].axhline(vo2max, ls="--", color="k", lw=1, label=f"VO2max {vo2max:.1f}")
                ax[1].set_ylabel("VO2 (ml/kg/min)"); ax[1].set_xlabel("minutes since first sample (gaps = unreliable data removed)")
                ax[1].legend(); fig.suptitle(f"{person.person_id} - session {big['session'].iloc[0]}")
                fig.tight_layout(); fig.savefig(od / "example_session.png", dpi=130); plt.close(fig)
        except Exception as e:
            print("plot skipped:", e)
    return res



class LiveVO2:
    """Real-time VO2 from a stream of HR samples.

        est = LiveVO2(resting_hr=52, hrmax=188, vo2max=48.5)   # vo2max from the trained model or a lab test
        vo2, pct = est.update(hr=131)                          # call once per new HR reading
    """
    def __init__(self, resting_hr: float, hrmax: float, vo2max: float, vo2_rest: float = 3.5, smooth: int = 5):
        self.rhr, self.hrmax, self.vo2max, self.vo2_rest = resting_hr, hrmax, vo2max, vo2_rest
        self.buf = []
        self.smooth = smooth
        self.peak_hr = 0.0

    @classmethod
    def from_model(cls, bundle, person_src, profile=None):
        """Calibrate resting HR / HRmax / VO2max from a person's past data, then go live."""
        r = predict_person(person_src, bundle, profile)
        return cls(r["rhr"], r["hrmax"], r["vo2max_pred"])

    def update(self, hr: float):
        self.buf = (self.buf + [float(hr)])[-self.smooth:]
        h = float(np.median(self.buf))
        self.peak_hr = max(self.peak_hr, h)
        if self.peak_hr > self.hrmax:                 # person exceeded the HRmax estimate -> raise it
            self.hrmax = self.peak_hr
        pct = float(np.clip((h - self.rhr) / max(self.hrmax - self.rhr, 20.0), 0, 1))
        return self.vo2_rest + pct * (self.vo2max - self.vo2_rest), pct


def person_from_csv(path, cfg: Config = CFG) -> Person:
    """Any single CSV with columns timestamp, hr [, rms] -> Person with one training session."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "hr" not in df and "value" in df:           # raw export files use 'value' for the HR column
        df = df.rename(columns={"value": "hr"})
    s = _prep_session(Path(path).stem, df, cfg)
    if s is None:
        raise ValueError("CSV needs columns 'timestamp' and 'hr' with at least 3 rows")
    return Person(Path(path).stem, training=[s])


def load_profiles(path) -> dict:
    if not path:
        return {}
    p = pd.read_csv(path)
    p.columns = [c.strip().lower() for c in p.columns]
    p["person_id"] = p["person_id"].astype(str).str.strip()
    return {r["person_id"]: profile_of(r) for _, r in p.iterrows()}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VO2 / VO2max from HR + motion sessions")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on a folder of labelled people")
    t.add_argument("--data", required=True, help="e.g. D:/dataset  (one sub-folder or .zip per person)")
    t.add_argument("--labels", required=True, help="csv: person_id, vo2max, age, resting_hr [, sex, weight_kg, height_cm]")
    t.add_argument("--out", default="vo2_model.joblib")
    t.add_argument("--report", default=None, help="folder for metrics/plots (default: <model folder>/reports)")
    t.add_argument("--n-augment", type=int, default=CFG.n_augment)
    t.add_argument("--exclude", nargs="*", default=[], help='person_ids to leave out, e.g. --exclude "Nirmal Kumar"')

    def add_profile_args(p):
        p.add_argument("--profile", help="csv: person_id, age, resting_hr, sex, weight_kg, height_cm")
        p.add_argument("--age", type=float); p.add_argument("--resting-hr", type=float)
        p.add_argument("--sex"); p.add_argument("--weight", type=float); p.add_argument("--height", type=float)
        p.add_argument("--vo2max", type=float, help="use this known VO2max instead of the model's prediction")

    p = sub.add_parser("predict", help="new person(s) in the same zip/folder format")
    p.add_argument("--input", required=True); p.add_argument("--model", default="vo2_model.joblib")
    p.add_argument("--out", default="results"); add_profile_args(p)

    c = sub.add_parser("predict-csv", help="one CSV with columns timestamp,hr[,rms] (new / recorded data)")
    c.add_argument("--input", required=True); c.add_argument("--model", default="vo2_model.joblib")
    c.add_argument("--out", default="results"); add_profile_args(c)

    l = sub.add_parser("live-demo", help="replay a CSV sample-by-sample the way a live stream would arrive")
    l.add_argument("--input", required=True); l.add_argument("--resting-hr", type=float, required=True)
    l.add_argument("--hrmax", type=float, required=True); l.add_argument("--vo2max", type=float, required=True)
    a = ap.parse_args()

    def single_profile(a):
        d = {"age": a.age, "resting_hr": a.resting_hr, "sex": a.sex, "weight_kg": a.weight, "height_cm": a.height}
        return {k: v for k, v in d.items() if v is not None}

    if a.cmd == "train":
        CFG.n_augment = a.n_augment
        train(a.data, a.labels, a.out, CFG, a.report, a.exclude)
    elif a.cmd == "predict":
        bundle = joblib.load(a.model)
        profiles = load_profiles(a.profile)
        results = []
        for src in discover_people(a.input):
            pid = src.stem if src.suffix.lower() == ".zip" else src.name
            r = predict_person(src, bundle, profiles.get(pid, single_profile(a)), a.out, a.vo2max)
            results.append(r)
            print(f"{pid}: VO2max = {r['vo2max_pred']} ml/kg/min  (approx. +/-{bundle['cv_rmse']:.1f})  "
                  f"[rhr {r['rhr']}, hrmax {r['hrmax']}, sessions {r['n_training_sessions']}]")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(Path(a.out) / "all_predictions.csv", index=False)
        print(f"Outputs written to {a.out}/")
    elif a.cmd == "predict-csv":
        bundle = joblib.load(a.model)
        r = predict_person(person_from_csv(a.input, bundle["cfg"]), bundle, single_profile(a), a.out, a.vo2max)
        print(json.dumps(r, indent=2))
    else:  # live-demo
        est = LiveVO2(a.resting_hr, a.hrmax, a.vo2max)
        df = pd.read_csv(a.input); df.columns = [c.strip().lower() for c in df.columns]
        if "hr" not in df and "value" in df:
            df = df.rename(columns={"value": "hr"})
        for i, hr in enumerate(df["hr"].dropna()):
            v, pct = est.update(hr)
            if i % 10 == 0:
                print(f"sample {i:5d}  HR {hr:6.1f}  VO2 {v:5.1f} ml/kg/min  ({100 * v / a.vo2max:4.0f}% of VO2max)")


if __name__ == "__main__":
    main()
