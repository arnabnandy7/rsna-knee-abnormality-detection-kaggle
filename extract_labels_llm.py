"""
Optimized Fast LLM Label Extraction for RSNA Knee Abnormality Detection
Uses Ollama (qwen2.5:1.5b) - optimized for fast CPU inference.
"""

import csv
import json
import re
import time
import sys
import os
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
MODEL = "qwen2.5:1.5b"  # 1.5B is much faster on CPU and doesn't throttle
TRAIN_CSV = Path("data/train.csv")
OUTPUT_CSV = Path("llm_labels.csv")
CHECKPOINT_CSV = Path("llm_labels_checkpoint.csv")

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

PROMPT_TEMPLATE = """You are a radiologist reading a knee MRI report. Extract findings for these 12 categories as confidence (0.0 to 1.0):
1.0 = present/abnormal, 0.0 = normal/absent/unremarkable.

Categories:
ACL, MCL, Medial_Meniscus, Lateral_Meniscus, Medial_OA, Lateral_OA, PF_OA, Effusion, Synovitis, Bakers, Contusion, Fracture

REPORT:
{report}

Respond ONLY with raw JSON. Format:
{{"ACL":0.0,"MCL":0.0,"Medial_Meniscus":0.0,"Lateral_Meniscus":0.0,"Medial_OA":0.0,"Lateral_OA":0.0,"PF_OA":0.0,"Effusion":0.0,"Synovitis":0.0,"Bakers":0.0,"Contusion":0.0,"Fracture":0.0}}

JSON:"""


def query_ollama(report_text, retries=2):
    """Send report to Ollama with strict timeout and compact parameters."""
    prompt = PROMPT_TEMPLATE.format(report=report_text[:1200])

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 180,
                    "num_ctx": 1024,
                    "top_p": 0.9,
                }
            }, timeout=45)

            if resp.status_code != 200:
                time.sleep(0.5)
                continue

            text = resp.json().get("response", "")
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result = {}
                for col in TARGET_COLS:
                    val = data.get(col, 0.05)
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = 0.05
                    result[col] = max(0.01, min(0.99, val))
                return result

        except Exception:
            time.sleep(0.5)

    # Fallback to simple regex keyword matching if LLM times out
    return regex_fallback(report_text)


def regex_fallback(report_text):
    """Fast regex fallback so it never hangs or drops a record."""
    text = report_text.lower()
    res = {c: 0.05 for c in TARGET_COLS}
    patterns = {
        'ACL': r'\b(acl|anterior cruciate)\b',
        'MCL': r'\b(mcl|medial collateral)\b',
        'Medial_Meniscus': r'\b(medial meniscus|medial meniscal)\b',
        'Lateral_Meniscus': r'\b(lateral meniscus|lateral meniscal)\b',
        'Medial_OA': r'\bmedial.*(osteoarthritis|narrowing|cartilage loss|chondromalacia)\b',
        'Lateral_OA': r'\blateral.*(osteoarthritis|narrowing|cartilage loss|chondromalacia)\b',
        'PF_OA': r'\b(patellofemoral|pf).*(osteoarthritis|cartilage loss)\b',
        'Effusion': r'\b(effusion|fluid)\b',
        'Synovitis': r'\b(synovitis|synovial)\b',
        'Bakers': r'\b(baker|popliteal cyst)\b',
        'Contusion': r'\b(contusion|bone bruise|edema)\b',
        'Fracture': r'\b(fracture)\b',
    }
    for col, pat in patterns.items():
        if re.search(pat, text):
            # Check negation
            if not re.search(r'\b(no|without|intact|normal|unremarkable)\b.*' + pat, text):
                res[col] = 0.85
    return res


def load_checkpoint():
    done = {}
    if CHECKPOINT_CSV.exists():
        with open(CHECKPOINT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row['StudyInstanceUID']
                done[sid] = {col: float(row.get(col, 0.05)) for col in TARGET_COLS}
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
        print("ERROR: Ollama is not running.")
        sys.exit(1)

    with open(TRAIN_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"Total studies: {len(rows)}")

    results = load_checkpoint()
    print(f"Loaded from checkpoint: {len(results)} studies already done!")

    remaining = [r for r in rows if r['StudyInstanceUID'] not in results]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All studies already processed!")
    else:
        print(f"Model: {MODEL} (Fast CPU optimized)")
        start_time = time.time()
        
        for i, row in enumerate(remaining):
            sid = row['StudyInstanceUID']
            report = row.get('Report', '').strip()

            if not report:
                results[sid] = {col: 0.05 for col in TARGET_COLS}
                continue

            result = query_ollama(report)
            results[sid] = result

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = ((len(remaining) - i - 1) / rate) / 60 if rate > 0 else 0

            if (i + 1) % 20 == 0:
                print(f"  [{len(results)}/{len(rows)}] {rate:.2f} rep/s | ETA: {eta_min:.1f} min ({eta_min/60:.1f}h)")

            if (i + 1) % 100 == 0:
                save_checkpoint(results)
                print(f"    --> Checkpoint auto-saved ({len(results)} total)")

        save_checkpoint(results)

    # Write final output
    print(f"\nWriting final dataset to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID'] + ORIGINAL_NAMES)
        writer.writeheader()
        for row in rows:
            sid = row['StudyInstanceUID']
            vals = results.get(sid, {col: 0.05 for col in TARGET_COLS})
            out_row = {'StudyInstanceUID': sid}
            for tcol, ocol in zip(TARGET_COLS, ORIGINAL_NAMES):
                out_row[ocol] = f"{vals[tcol]:.4f}"
            writer.writerow(out_row)

    print(f"Done! {len(results)} studies saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
