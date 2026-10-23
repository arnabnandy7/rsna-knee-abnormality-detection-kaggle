"""
V2 Label Extraction — qwen3:latest (8B) with medical-grade prompting.
Runs on CPU via Ollama. ~30-60s per report on 8GB RAM.
Checkpoints every 50 studies for safe resume.

Usage: python extract_labels_v2.py
"""

import csv
import json
import re
import time
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
    import requests

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:latest"  # 8B model — much better reasoning than 1.5B
TRAIN_CSV = Path("data/train.csv")
OUTPUT_CSV = Path("llm_labels_v2.csv")
CHECKPOINT_CSV = Path("llm_labels_v2_checkpoint.csv")

TARGET_COLS = [
    'ACL', 'MCL', 'Medial_Meniscus', 'Lateral_Meniscus',
    'Medial_OA', 'Lateral_OA', 'PF_OA', 'Effusion',
    'Synovitis', 'Bakers', 'Contusion', 'Fracture'
]

ORIGINAL_NAMES = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# --- Medical-grade prompt with few-shot examples and negation handling ---
PROMPT_TEMPLATE = """/no_think
You are an expert musculoskeletal radiologist extracting structured findings from a knee MRI report.

CRITICAL RULES:
- "No evidence of X" / "X is intact" / "Normal X" / "X unremarkable" = 0.0 (ABSENT)
- "X tear" / "X rupture" / "X injury" / "X abnormal" = 1.0 (PRESENT)
- "Mild X" / "Small X" / "Trace X" = 0.7 (LIKELY PRESENT)
- "Possible X" / "Cannot exclude X" / "Questionable X" = 0.5 (UNCERTAIN)
- If a structure is NOT MENTIONED AT ALL, use 0.3 (unknown, lean toward possible)
- Reports may be in ANY language (German, French, Dutch, etc.) — translate mentally first

CATEGORIES:
1. ACL: Anterior cruciate ligament tear, partial tear, rupture, sprain, laxity
2. MCL: Medial collateral ligament tear, sprain, thickening, injury
3. Medial_Meniscus: Medial meniscus tear, degeneration, extrusion, signal abnormality
4. Lateral_Meniscus: Lateral meniscus tear, degeneration, signal abnormality
5. Medial_OA: Medial compartment osteoarthritis, cartilage loss/thinning, joint space narrowing, osteophytes
6. Lateral_OA: Lateral compartment osteoarthritis, cartilage loss, joint space narrowing
7. PF_OA: Patellofemoral osteoarthritis, trochlear/patellar cartilage loss, chondromalacia patellae
8. Effusion: Joint effusion, suprapatellar fluid, synovial fluid
9. Synovitis: Synovial thickening, synovial proliferation, synovial enhancement
10. Bakers: Baker's cyst, popliteal cyst
11. Contusion: Bone contusion, bone bruise, bone marrow edema, subchondral edema
12. Fracture: Any fracture, stress fracture, avulsion, cortical irregularity

FEW-SHOT EXAMPLES:

Report: "ACL is intact. MCL shows grade 2 sprain. Complex tear of the posterior horn of the medial meniscus. Moderate joint effusion. No bone contusion or fracture."
Answer: {{"ACL":0.0,"MCL":1.0,"Medial_Meniscus":1.0,"Lateral_Meniscus":0.3,"Medial_OA":0.3,"Lateral_OA":0.3,"PF_OA":0.3,"Effusion":1.0,"Synovitis":0.3,"Bakers":0.3,"Contusion":0.0,"Fracture":0.0}}

Report: "Normales Knie-MRT. Keine Meniskusläsion. Kreuzbänder intakt. Kein Erguss."
Answer: {{"ACL":0.0,"MCL":0.3,"Medial_Meniscus":0.0,"Lateral_Meniscus":0.0,"Medial_OA":0.3,"Lateral_OA":0.3,"PF_OA":0.3,"Effusion":0.0,"Synovitis":0.3,"Bakers":0.3,"Contusion":0.3,"Fracture":0.3}}

Report: "Trace effusion. Mild chondromalacia patellae grade 2. Small Baker's cyst. Otherwise unremarkable."
Answer: {{"ACL":0.0,"MCL":0.0,"Medial_Meniscus":0.0,"Lateral_Meniscus":0.0,"Medial_OA":0.0,"Lateral_OA":0.0,"PF_OA":0.7,"Effusion":0.7,"Synovitis":0.0,"Bakers":0.7,"Contusion":0.0,"Fracture":0.0}}

NOW EXTRACT FROM THIS REPORT:
{report}

Respond with ONLY a JSON object. No explanation, no markdown, no text before or after.
JSON:"""


def query_ollama(report_text, retries=2):
    """Send report to qwen3 8B and parse JSON response."""
    prompt = PROMPT_TEMPLATE.format(report=report_text[:2000])

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "num_predict": 200,
                    "num_ctx": 2048,
                }
            }, timeout=180)  # 3 min timeout for 8B on CPU

            if resp.status_code != 200:
                time.sleep(1)
                continue

            text = resp.json().get("response", "")

            # Extract JSON
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result = {}
                for col in TARGET_COLS:
                    val = data.get(col, 0.3)
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = 0.3
                    result[col] = max(0.01, min(0.99, val))
                return result

        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt+1}")
            time.sleep(2)
        except json.JSONDecodeError:
            print(f"  JSON parse error on attempt {attempt+1}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(1)

    # Fallback: use 0.3 (unknown) instead of 0.05 (confident negative)
    return {col: 0.30 for col in TARGET_COLS}


def load_checkpoint():
    done = {}
    if CHECKPOINT_CSV.exists():
        with open(CHECKPOINT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row['StudyInstanceUID']
                done[sid] = {col: float(row.get(col, 0.3)) for col in TARGET_COLS}
    return done


def save_checkpoint(results):
    with open(CHECKPOINT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID'] + TARGET_COLS)
        writer.writeheader()
        for sid, vals in results.items():
            row = {'StudyInstanceUID': sid}
            row.update(vals)
            writer.writerow(row)


def main():
    if not TRAIN_CSV.exists():
        print(f"ERROR: {TRAIN_CSV} not found!")
        sys.exit(1)

    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("ERROR: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    with open(TRAIN_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"Total studies: {len(rows)}")
    print(f"Model: {MODEL} (8B, medical-grade prompts)")

    results = load_checkpoint()
    print(f"Checkpoint: {len(results)} already done")

    remaining = [r for r in rows if r['StudyInstanceUID'] not in results]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
    else:
        start_time = time.time()
        for i, row in enumerate(remaining):
            sid = row['StudyInstanceUID']
            report = row.get('Report', '').strip()

            if not report:
                results[sid] = {col: 0.30 for col in TARGET_COLS}
                continue

            result = query_ollama(report)
            results[sid] = result

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = ((len(remaining) - i - 1) / rate) / 60 if rate > 0 else 0

            if (i + 1) % 10 == 0:
                print(f"  [{len(results)}/{len(rows)}] {rate:.3f} rep/s | ETA: {eta_min:.0f} min ({eta_min/60:.1f}h)")

            if (i + 1) % 50 == 0:
                save_checkpoint(results)
                print(f"    --> Checkpoint saved ({len(results)} total)")

        save_checkpoint(results)

    # Write final output
    print(f"\nWriting {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID'] + ORIGINAL_NAMES)
        writer.writeheader()
        for row in rows:
            sid = row['StudyInstanceUID']
            vals = results.get(sid, {col: 0.30 for col in TARGET_COLS})
            out_row = {'StudyInstanceUID': sid}
            for tcol, ocol in zip(TARGET_COLS, ORIGINAL_NAMES):
                out_row[ocol] = f"{vals[tcol]:.4f}"
            writer.writerow(out_row)

    print(f"Done! {len(results)} studies -> {OUTPUT_CSV}")

    # ===== VALIDATION against 58 GT studies =====
    import pandas as pd
    import numpy as np
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("sklearn not available, skipping validation")
        return

    train_df = pd.read_csv(str(TRAIN_CSV))
    llm_df = pd.read_csv(str(OUTPUT_CSV))
    labeled_mask = train_df['ACL'].notna()
    df_gt = train_df[labeled_mask]
    df_llm = llm_df[labeled_mask]

    print(f"\n{'='*60}")
    print(f"VALIDATION against {len(df_gt)} ground-truth studies:")
    print(f"{'='*60}")
    aucs = []
    for tcol, ocol in zip(TARGET_COLS, ORIGINAL_NAMES):
        y_true = df_gt[ocol].values.astype(float)
        y_pred = df_llm[ocol].values.astype(float)
        pos = int(y_true.sum())
        try:
            auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        except:
            auc = 0.5
        aucs.append(auc)
        marker = "OK" if auc > 0.65 else ("WARN" if auc > 0.55 else "BAD")
        print(f"  {ocol:20s} (pos={pos:2d}): AUC = {auc:.4f}  [{marker}]")

    mean_auc = np.mean(aucs)
    print(f"\n  MEAN AUC: {mean_auc:.4f}")
    if mean_auc > 0.70:
        print("  >>> PASS: Labels are good enough to use for training!")
    elif mean_auc > 0.60:
        print("  >>> MARGINAL: Labels may help but proceed with caution")
    else:
        print("  >>> FAIL: Labels are too noisy — DO NOT use for training")


if __name__ == "__main__":
    main()
