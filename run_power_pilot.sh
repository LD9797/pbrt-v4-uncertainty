#!/usr/bin/env bash

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCENE="${1:-../pbrt-v4-scenes/barcelona-pavilion/pavilion-day.pbrt}"
OUTDIR="${2:-experiments/PowerPilot/Barcelona}"
GT_REFERENCE="${3:-experiments/GT_PBRT_BARCELONA_2048.exr}"
SEED_COUNT="${4:-10}"
ALL_GT_SUMMARY="${5:-$REPO_ROOT/all_groundtruth_metrics_summary.csv}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/loss/.venv/bin/python}"

if [[ "$SCENE" != /* ]]; then
    SCENE="$REPO_ROOT/$SCENE"
fi
if [[ "$OUTDIR" != /* ]]; then
    OUTDIR="$REPO_ROOT/$OUTDIR"
fi
if [[ "$GT_REFERENCE" != /* ]]; then
    GT_REFERENCE="$REPO_ROOT/$GT_REFERENCE"
fi
if [[ "$ALL_GT_SUMMARY" != /* ]]; then
    ALL_GT_SUMMARY="$REPO_ROOT/$ALL_GT_SUMMARY"
fi

mkdir -p "$OUTDIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

# Seeds piloto
SEEDS=()
for ((i=0; i < SEED_COUNT; i++)); do
    SEEDS+=("$i")
done

# Formato:
# ID|RANGO_A_CERO
#
# FULL no tiene rango.
CONFIGS=(
    "FULL|"
    "F1_No_Position|0,35"
    "F2_No_Direction|36,37"
    "F3_No_Normal|38,39"
    "F4_No_Roughness|40,40"
    "F5_No_Diffuse|41,43"
    "F6_No_Specular|44,46"
    "ZERO_INPUTS|0,46"
)

LOGFILE="$OUTDIR/run.log"

echo "========================================" | tee "$LOGFILE"
echo "Power Analysis Pilot" | tee -a "$LOGFILE"
echo "Scene:    $SCENE" | tee -a "$LOGFILE"
echo "Output:   $OUTDIR" | tee -a "$LOGFILE"
echo "GT:       $GT_REFERENCE" | tee -a "$LOGFILE"
echo "Seeds:    ${SEEDS[*]}" | tee -a "$LOGFILE"
echo "Started:  $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

render_ground_truth() {
    mkdir -p "$(dirname "$GT_REFERENCE")"

    echo "" | tee -a "$LOGFILE"
    echo "----------------------------------------" | tee -a "$LOGFILE"
    echo "Rendering ground truth: $GT_REFERENCE" | tee -a "$LOGFILE"
    echo "Start: $(date)" | tee -a "$LOGFILE"

    ./build/pbrt \
        --gpu \
        --spp 2048 \
        --seed 0 \
        --outfile "$GT_REFERENCE" \
        "$SCENE" 2>&1 | tee -a "$LOGFILE"

    local status=${PIPESTATUS[0]}
    echo "End:   $(date)" | tee -a "$LOGFILE"
    echo "Status: $status" | tee -a "$LOGFILE"

    if [[ $status -ne 0 ]]; then
        echo "ERROR: ground truth render failed." | tee -a "$LOGFILE"
        return "$status"
    fi
}

run_render() {
    local seed="$1"
    local config="$2"
    local zero_range="$3"

    local exr="$OUTDIR/${config}_seed${seed}.exr"
    local metrics="$OUTDIR/${config}_seed${seed}_metrics.csv"

    echo "" | tee -a "$LOGFILE"
    echo "----------------------------------------" | tee -a "$LOGFILE"
    echo "Seed:   $seed" | tee -a "$LOGFILE"
    echo "Config: $config" | tee -a "$LOGFILE"
    echo "Start:  $(date)" | tee -a "$LOGFILE"

    CMD=(
        ./build/pbrt
        --gpu
        --enable-nrc
        --spp 64
        --seed "$seed"
        --nrc-warmup-samples 16
        --nrc-train-steps 2
        --nrc-brightness-oversampling=true
        --nrc-config src/nrc/nrc_config.json
        --outfile "$exr"
        --nrc-metrics-csv "$metrics"
    )

    # Solo agrega --nrc-zero-inputs cuando corresponde.
    if [[ -n "$zero_range" ]]; then
        CMD+=(--nrc-zero-inputs "$zero_range")
    fi

    CMD+=("$SCENE")

    "${CMD[@]}" 2>&1 | tee -a "$LOGFILE"
    status=${PIPESTATUS[0]}

    echo "End:    $(date)" | tee -a "$LOGFILE"
    echo "Status: $status" | tee -a "$LOGFILE"

    if [[ $status -ne 0 ]]; then
        echo "ERROR: $config seed $seed failed." | tee -a "$LOGFILE"
    fi
}

compare_render_to_gt() {
    local exr="$1"
    local out="${exr%.exr}_groundtruth_metrics.txt"

    if [[ ! -f "$GT_REFERENCE" ]]; then
        echo "WARNING: ground truth reference not found: $GT_REFERENCE" | tee -a "$LOGFILE"
        return 0
    fi

    echo "" | tee -a "$LOGFILE"
    echo "Comparing $exr against $GT_REFERENCE" | tee -a "$LOGFILE"
    "$PYTHON_BIN" "$REPO_ROOT/loss/compare_images.py" "$GT_REFERENCE" "$exr" "$out" 2>&1 | tee -a "$LOGFILE"
}

render_ground_truth

for seed in "${SEEDS[@]}"; do

    echo "" | tee -a "$LOGFILE"
    echo "========================================" | tee -a "$LOGFILE"
    echo "SEED $seed" | tee -a "$LOGFILE"
    echo "========================================" | tee -a "$LOGFILE"

    # Randomiza el orden de las configuraciones para cada seed.
    mapfile -t SHUFFLED_CONFIGS < <(
        printf "%s\n" "${CONFIGS[@]}" | shuf
    )

    echo "Execution order:" | tee -a "$LOGFILE"
    printf '%s\n' "${SHUFFLED_CONFIGS[@]}" | tee -a "$LOGFILE"

    for entry in "${SHUFFLED_CONFIGS[@]}"; do

        IFS='|' read -r config zero_range <<< "$entry"

        run_render "$seed" "$config" "$zero_range"

    done
done

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "Running post-render image comparisons" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

for exr in "$OUTDIR"/*.exr; do
    if [[ -f "$exr" ]]; then
        compare_render_to_gt "$exr"
    fi
done

# Build the Barcelona ground-truth summary CSV from the per-image comparison outputs.
summary_file="$OUTDIR/ground_truth_summary.csv"
python3 - <<'PY' "$OUTDIR" "$summary_file"
import csv, glob, os, re, sys
outdir, summary_path = sys.argv[1], sys.argv[2]
rows = []
for txt in sorted(glob.glob(os.path.join(outdir, '*_groundtruth_metrics.txt'))):
    values = {}
    with open(txt, 'r', encoding='utf-8') as fh:
        for line in fh:
            m = re.match(r'^(MSE|RMSE|MAE|relMSE|PSNR \(dB\)|SSIM)\s+(.+)$', line.strip())
            if m:
                values[m.group(1)] = float(m.group(2))
    base = os.path.basename(txt)
    if not values:
        continue
    rows.append([
        os.path.splitext(base)[0].replace('_groundtruth_metrics', ''),
        values.get('MSE', float('nan')),
        values.get('relMSE', float('nan')),
        values.get('PSNR (dB)', float('nan')),
        values.get('SSIM', float('nan')),
    ])
with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['file', 'MSE', 'relMSE', 'PSNR(dB)', 'SSIM'])
    writer.writerows(rows)
print(f'WROTE {summary_path}')
PY

python3 - <<'PY' "$ALL_GT_SUMMARY"
import csv, glob, os, re, sys
summary_path = sys.argv[1]
rows = []
for txt in sorted(glob.glob('**/*_groundtruth_metrics.txt', recursive=True)):
    values = {}
    with open(txt, 'r', encoding='utf-8') as fh:
        for line in fh:
            m = re.match(r'^(MSE|RMSE|MAE|relMSE|PSNR \(dB\)|SSIM)\s+(.+)$', line.strip())
            if m:
                values[m.group(1)] = float(m.group(2))
    if not values:
        continue
    rows.append([
        os.path.relpath(txt),
        values.get('MSE', float('nan')),
        values.get('relMSE', float('nan')),
        values.get('PSNR (dB)', float('nan')),
        values.get('SSIM', float('nan')),
    ])
with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['file', 'MSE', 'relMSE', 'PSNR(dB)', 'SSIM'])
    writer.writerows(rows)
print(f'WROTE {summary_path}')
PY

all_metrics_file="$OUTDIR/all_render_metrics.csv"
python3 - <<'PY' "$OUTDIR" "$all_metrics_file"
import csv, glob, os, sys
outdir, summary_path = sys.argv[1], sys.argv[2]
rows = []
for csv_path in sorted(glob.glob(os.path.join(outdir, '*_metrics.csv'))):
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append([
                os.path.basename(csv_path),
                row.get('sample', ''),
                row.get('records', ''),
                row.get('mse', ''),
                row.get('rel_mse', ''),
                row.get('mae', ''),
                row.get('nrc_loss', ''),
            ])
with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['origin', 'sample', 'records', 'mse', 'rel_mse', 'mae', 'nrc_loss'])
    writer.writerows(rows)
print(f'WROTE {summary_path}')
PY

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "Experiment finished: $(date)" | tee -a "$LOGFILE"
echo "Results: $OUTDIR" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"