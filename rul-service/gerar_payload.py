#!/usr/bin/env python3
# gerar_payload.py — Gera payload para /predict a partir do test_FD001.txt (sem numpy/pandas)

import argparse, json, os, re, sys

def read_features(model_dir="models/fd001_lstm_v1"):
    path = os.path.join(model_dir, "features.json")
    if not os.path.exists(path):
        sys.exit(f"[ERRO] features.json não encontrado em {path}")
    with open(path) as f:
        return json.load(f)

def parse_test_file(test_path, unit_id):
    # colunas originais do C-MAPSS
    cols = ['unit','cycle'] + [f'setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    rows = []
    with open(test_path, "r") as fh:
        for line in fh:
            parts = [p for p in re.split(r"\s+", line.strip()) if p]
            if len(parts) < len(cols):
                continue
            rec = {}
            for i, name in enumerate(cols):
                v = float(parts[i])
                if name in ("unit","cycle"):
                    rec[name] = int(v)
                else:
                    rec[name] = v
            if rec["unit"] == unit_id:
                rows.append(rec)
    if not rows:
        sys.exit(f"[ERRO] Unit {unit_id} não encontrada em {test_path}")
    # ordenar por ciclo
    rows.sort(key=lambda r: r["cycle"])
    return rows

def build_window(rows, ws=30):
    """Últimos ws ciclos; se faltar, faz padding repetindo o primeiro registro (ajustando cycle para trás)."""
    if len(rows) >= ws:
        return rows[-ws:]
    need = ws - len(rows)
    first = rows[0].copy()
    first_cycle = first["cycle"]
    padded = []
    # ciclos anteriores: first_cycle - need ... first_cycle - 1
    start = first_cycle - need
    for c in range(start, first_cycle):
        r = first.copy()
        r["cycle"] = c
        padded.append(r)
    return padded + rows

def main():
    ap = argparse.ArgumentParser(description="Gera payload JSON para /predict (FD001)")
    ap.add_argument("--test", required=True, help="Caminho para test_FD001.txt (bruto)")
    ap.add_argument("--unit", required=True, type=int, help="Unit ID (>=1)")
    ap.add_argument("--ws", type=int, default=30, help="Window size (default=30)")
    ap.add_argument("--model_dir", default="models/fd001_lstm_v1", help="Diretório do modelo (para ler features.json)")
    ap.add_argument("--out", default=None, help="Arquivo de saída (default=payload_fd001_unit{unit}.json)")
    ap.add_argument("--mc", type=int, default=30, help="mc_passes (default=30)")
    args = ap.parse_args()

    if args.unit < 1:
        sys.exit("[ERRO] --unit deve ser >= 1")

    features = read_features(args.model_dir)
    rows = parse_test_file(args.test, args.unit)
    sel = build_window(rows, ws=args.ws)

    # montar records com exatas features
    records = []
    for r in sel:
        item = {"cycle": int(r["cycle"])}
        for f in features:
            if f not in r:
                sys.exit(f"[ERRO] Feature '{f}' ausente no arquivo de teste.")
            item[f] = float(r[f])
        records.append(item)

    out_path = args.out or f"payload_fd001_unit{args.unit}.json"
    payload = {"unit": int(args.unit), "mc_passes": int(args.mc), "records": records}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[OK] Payload salvo em {out_path}")

if __name__ == "__main__":
    main()


