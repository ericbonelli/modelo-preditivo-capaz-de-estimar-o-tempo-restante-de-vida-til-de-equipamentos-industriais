# app/main.py
# -----------------------------------------------------------
# RUL API (FD001) – FastAPI + Keras (SavedModel ou .h5)
# - Carrega scaler/features/config e modelo (pasta "model" e/ou "model.h5")
# - Pré-processa JSON: ordena por cycle, normaliza, janela com padding
# - Inferência robusta:
#     * .predict() quando existir
#     * __call__(training=...) quando possível
#     * assinatura SavedModel "serving_default" como fallback
# - MC-Dropout opcional (incerteza) – prefere .h5 para ativar dropout
# - Calibração:
#     * linear: y' = CALIB_A * y + CALIB_B
#     * piecewise: y' = y + (B_LOW se y<=T; B_HIGH se y>T)
# - Clamps configuráveis: RUL_MIN (default 0), RUL_MAX (opcional)
# -----------------------------------------------------------
import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.models import load_model

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MODEL_DIR   = os.getenv("MODEL_DIR", "models/fd001_lstm_v1")

# ---------- Calibração & Clamps ----------
CALIB_A = float(os.getenv("CALIB_A", "1.0"))
CALIB_B = float(os.getenv("CALIB_B", "0.0"))

# Modo de calibração: "linear" (padrão) ou "piecewise"
CALIB_MODE   = os.getenv("CALIB_MODE", "linear").lower()
CALIB_T      = float(os.getenv("CALIB_T", "120"))     # limiar do piecewise
CALIB_B_LOW  = float(os.getenv("CALIB_B_LOW", "0.0")) # offset para y<=T
CALIB_B_HIGH = float(os.getenv("CALIB_B_HIGH", "-8.0")) # offset para y>T

CLAMP_MIN = float(os.getenv("RUL_MIN", "0"))  # piso padrão = 0
_RUL_MAX  = os.getenv("RUL_MAX", "")
CLAMP_MAX = float(_RUL_MAX) if _RUL_MAX not in ("", None) else math.inf

def _clamp(v: float) -> float:
    v = max(CLAMP_MIN, v)
    if math.isfinite(CLAMP_MAX):
        v = min(CLAMP_MAX, v)
    return v

def _calib_scalar(y: float) -> float:
    """Aplica calibração linear ou piecewise a um escalar."""
    if CALIB_MODE == "piecewise":
        return y + (CALIB_B_LOW if y <= CALIB_T else CALIB_B_HIGH)
    # linear (padrão)
    return CALIB_A * y + CALIB_B

def _calib_array(arr: np.ndarray) -> np.ndarray:
    """Aplica calibração elemento a elemento a um array."""
    return np.vectorize(_calib_scalar)(arr)

# ---------- Carregar artefatos ----------
try:
    with open(os.path.join(MODEL_DIR, "features.json")) as f:
        FEATURES: List[str] = json.load(f)
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        CFG = json.load(f)
    WS: int = int(CFG["window_size"])
except Exception as e:
    raise RuntimeError(f"Falha ao ler features/config em {MODEL_DIR}: {e}")

try:
    SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
except Exception as e:
    raise RuntimeError(f"Falha ao carregar scaler.pkl: {e}")

MODEL_PATH_DIR = os.path.join(MODEL_DIR, "model")
MODEL_PATH_H5  = os.path.join(MODEL_DIR, "model.h5")

# Modelo principal (SavedModel ou H5)
try:
    if os.path.isdir(MODEL_PATH_DIR):
        MODEL = load_model(MODEL_PATH_DIR)   # SavedModel (pasta)
        MODEL_KIND = "SavedModel"
    elif os.path.exists(MODEL_PATH_H5):
        MODEL = load_model(MODEL_PATH_H5)    # H5
        MODEL_KIND = "H5"
    else:
        raise FileNotFoundError("Nem 'model/' (SavedModel) nem 'model.h5' encontrados.")
except Exception as e:
    raise RuntimeError(f"Falha ao carregar o modelo principal: {e}")

# H5 adicional (preferido no MC-Dropout, se existir)
try:
    MODEL_H5 = load_model(MODEL_PATH_H5) if os.path.exists(MODEL_PATH_H5) else None
except Exception:
    MODEL_H5 = None

# ---------- Schemas ----------
Record = Dict[str, float]

class PredictRequest(BaseModel):
    unit: int = Field(..., ge=1)
    records: List[Record] = Field(..., min_items=1, description="Últimos N ciclos (valores brutos)")
    mc_passes: int = Field(0, ge=0, le=100, description="Se >0, ativa MC-Dropout")

class PredictResponse(BaseModel):
    unit: int
    rul_pred: float
    rul_std: Optional[float] = None
    ci95: Optional[Tuple[float, float]] = None

# ---------- App ----------
app = FastAPI(title="RUL API – FD001 LSTM", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------- Helpers ----------
def build_window(records: List[Record]) -> np.ndarray:
    """
    Constrói o tensor (1, WS, F) a partir do payload:
    - Ordena por 'cycle' se existir
    - Valida campos (FEATURES)
    - Normaliza com SCALER (fit do treino)
    - Faz padding à esquerda se N < WS
    """
    df = pd.DataFrame(records)
    if "cycle" in df.columns:
        df = df.sort_values("cycle").reset_index(drop=True)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Campos faltantes no payload: {sorted(missing)}")

    X = df[FEATURES].astype(float)
    Xs = SCALER.transform(X)

    if len(Xs) < WS:
        pad = np.repeat(Xs[[0]], WS - len(Xs), axis=0)
        Xs = np.vstack([pad, Xs])
    else:
        Xs = Xs[-WS:]

    return Xs[None, :, :]  # (1, WS, F)

def _to_numpy(x):
    try:
        return x.numpy()
    except Exception:
        return np.asarray(x)

def _call_serving_signature(x: np.ndarray):
    """
    Chama a assinatura 'serving_default' de um SavedModel.
    Tenta no objeto atual; se não houver, recarrega com tf.saved_model.load.
    """
    sigs = getattr(MODEL, "signatures", None)
    if isinstance(sigs, dict) and "serving_default" in sigs:
        out = sigs["serving_default"](tf.convert_to_tensor(x, dtype=tf.float32))
        if isinstance(out, dict):
            out = next(iter(out.values()))
        return out
    # fallback: carrega diretamente o SavedModel bruto
    loaded = tf.saved_model.load(MODEL_PATH_DIR)
    out = loaded.signatures["serving_default"](tf.convert_to_tensor(x, dtype=tf.float32))
    if isinstance(out, dict):
        out = next(iter(out.values()))
    return out

def _infer_array(x: np.ndarray, training: bool = False) -> np.ndarray:
    """
    Caminhos de inferência (ordem):
      a) MODEL.predict(x)
      b) MODEL(x, training=...)
      c) assinatura SavedModel 'serving_default'
    Retorna shape (batch,)
    """
    if hasattr(MODEL, "predict"):
        y = MODEL.predict(x, verbose=0)
        return _to_numpy(y).reshape(-1)
    try:
        y = MODEL(x, training=training)  # pode falhar em _UserObject
        return _to_numpy(y).reshape(-1)
    except Exception:
        pass
    y = _call_serving_signature(x)
    return _to_numpy(y).reshape(-1)

def _mc_dropout_predict(x: np.ndarray, passes: int) -> np.ndarray:
    """
    Executa várias passagens para estimar incerteza.
    Preferimos o H5 (se disponível) para forçar training=True (dropout ativo).
    """
    m = MODEL_H5 if MODEL_H5 is not None else MODEL
    preds = []
    for _ in range(passes):
        try:
            y = m(x, training=True)  # tenta ativar dropout
            y = _to_numpy(y).reshape(-1)[0]
        except Exception:
            y = _infer_array(x, training=False)[0]  # determinístico
        preds.append(float(y))
    return np.array(preds)

# ---------- Endpoints ----------
@app.get("/health")
def health():
    sigs = getattr(MODEL, "signatures", None)
    sig_keys = list(sigs.keys()) if isinstance(sigs, dict) else []
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model_kind": MODEL_KIND,
        "h5_loaded": bool(MODEL_H5 is not None),
        "model_dir": MODEL_DIR,
        "window_size": WS,
        "n_features": len(FEATURES),
        "rul_min": CLAMP_MIN,
        "rul_max": None if not math.isfinite(CLAMP_MAX) else CLAMP_MAX,
        "calib_mode": CALIB_MODE,
        "calib_a": CALIB_A,
        "calib_b": CALIB_B,
        "calib_t": CALIB_T,
        "calib_b_low": CALIB_B_LOW,
        "calib_b_high": CALIB_B_HIGH,
        "signatures": sig_keys,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        x = build_window(req.records)  # (1, WS, F)

        # MC-Dropout (incerteza)
        if req.mc_passes and req.mc_passes > 0:
            preds = _mc_dropout_predict(x, req.mc_passes)  # (passes,)
            preds = _calib_array(preds)                    # calibração (linear/piecewise)
            preds = np.vectorize(_clamp)(preds)            # clamp
            mean = float(np.mean(preds))
            std  = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
            ci95 = (mean - 1.96 * std, mean + 1.96 * std)
            return PredictResponse(unit=req.unit, rul_pred=mean, rul_std=std, ci95=ci95)

        # Inferência simples
        pred = float(_infer_array(x, training=False)[0])
        pred = float(_calib_scalar(pred))   # calibração (linear/piecewise)
        pred = _clamp(pred)                 # clamp
        return PredictResponse(unit=req.unit, rul_pred=pred)

    except HTTPException as e:
        raise e
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {ex}")
