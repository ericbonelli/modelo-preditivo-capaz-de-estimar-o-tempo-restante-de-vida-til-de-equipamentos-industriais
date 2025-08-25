#!/usr/bin/env python3
# benchmark_fd001.py — Avalia a API de RUL com o conjunto FD001 (sem numpy/pandas)

import argparse, json, math, os, re, sys, csv, statistics
from urllib import request, error

def nasa_score(y_true_list, y_pred_list):
    s = 0.0
    for yt, yp in zip(y_true_list, y_pred_list):
        d = yp - yt
        s += (math.exp(-d/13.0) - 1.0) if d < 0 else (math.exp(d/10.0) - 1.0)
    return s

def read_features(model_dir):
    path = os.path.join(model_dir, "features.json")
    if not os.path.exists(path):
        sys.exit(f"[ERRO] Não achei features.json em {path}")
    with open(path) as f:
        return json.load(f)

def read_test_fd001(test_path):
    cols = ['unit','cycle'] + [f'setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    data = {}
    with open(test_path, "r") as fh:
        for line in fh:
            parts = [p for p in re.split(r"\s+", line.strip()) if p]
            if len(parts) < len(cols): 
                continue
            rec = {}
            for i, name in enumerate(cols):
                v = float(parts[i])
                rec[name] = int(v) if name in ("unit","cycle") else v
            uid = rec["unit"]
            data.setdefault(uid, []).append(rec)
    # ordenar por ciclo
    for uid in data:
        data[uid].sort(key=lambda r: r["cycle"])
    return data  # dict[unit] -> list[records]

def read_rul_file(rul_path):
    vals = []
    with open(rul_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            vals.append(float(line.split()[0]))
    return vals  # 1 valor por unidade (na ordem das units)

def last_window(records, ws=30):
    if len(records) >= ws:
        return records[-ws:]
    need = ws - len(records)
    first = records[0]
    first_cycle = first["cycle"]
    pad = []
    for c in range(first_cycle - need, first_cycle):
        r = dict(first)
        r["cycle"] = c
        pad.append(r)
    return pad + records

def post_predict(api, payload):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(api.rstrip("/") + "/predict", data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Benchmark FD001 contra a RUL API")
    ap.add_argument("--api", required=False, default="http://localhost:8000", help="Base URL da API")
    ap.add_argument("--test", required=True, help="/Users/ericbonellileonel/Desktop/Semantix/CMAPSSData/test_FD001.txt")
    ap.add_argument("--rul",  required=True, help="/Users/ericbonellileonel/Desktop/Semantix/CMAPSSData/RUL_FD001.txt")
    ap.add_argument("--model_dir", default="models/fd001_lstm_v1", help="Para ler features.json")
    ap.add_argument("--ws", type=int, default=30)
    ap.add_argument("--mc", type=int, default=0, help="mc_passes para /predict (0 = desliga incerteza)")
    ap.add_argument("--out", default="predicoes_fd001.csv")
    args = ap.parse_args()

    features = read_features(args.model_dir)
    test_by_unit = read_test_fd001(args.test)
    units_sorted = sorted(test_by_unit.keys())
    rul_list = read_rul_file(args.rul)
    if len(rul_list) != len(units_sorted):
        print(f"[ALERTA] qtd de RULs ({len(rul_list)}) != qtd de units ({len(units_sorted)}). Usando mapeamento por ordem crescente de unit.", file=sys.stderr)

    preds, trues, rows_out = [], [], []
    for idx, uid in enumerate(units_sorted, start=1):
        recs = test_by_unit[uid]
        win = last_window(recs, ws=args.ws)
        # construir records com as FEATURES corretas
        records = []
        for r in win:
            item = {"cycle": int(r["cycle"])}
            for f in features:
                if f not in r:
                    sys.exit(f"[ERRO] Feature ausente no teste: {f}")
                item[f] = float(r[f])
            records.append(item)
        payload = {"unit": int(uid), "mc_passes": int(args.mc), "records": records}
        try:
            resp = post_predict(args.api, payload)
        except error.HTTPError as e:
            sys.exit(f"[ERRO HTTP] unit {uid}: {e.read().decode()}")
        except Exception as e:
            sys.exit(f"[ERRO REQ] unit {uid}: {e}")

        y_pred = float(resp["rul_pred"])
        # alinhar RUL verdadeiro: i-ésima linha do arquivo corresponde à i-ésima unit (ordenada)
        y_true = float(rul_list[idx-1]) if idx-1 < len(rul_list) else float("nan")

        preds.append(y_pred); trues.append(y_true)
        rows_out.append({"unit": uid, "RUL_true": y_true, "RUL_pred": y_pred, "erro": y_pred - y_true, "abs_erro": abs(y_pred - y_true)})

    # métricas
    n = len(trues)
    mae = sum(abs(p-t) for p,t in zip(preds,trues)) / n
    rmse = math.sqrt(sum((p-t)**2 for p,t in zip(preds,trues)) / n)
    nasa = nasa_score(trues, preds)
    super_pct = 100.0 * sum(1 for p,t in zip(preds,trues) if (p - t) > 0) / n

    # salvar CSV
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit","RUL_true","RUL_pred","erro","abs_erro"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print("\n== Benchmark FD001 ==")
    print(f"Unidades avaliadas: {n}")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"NASA: {nasa:.2f}  (quanto menor, melhor)")
    print(f"% superestimação: {super_pct:.1f}%")
    print(f"CSV salvo em: {args.out}")

if __name__ == "__main__":
    main()
