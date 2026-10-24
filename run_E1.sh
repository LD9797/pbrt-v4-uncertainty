#!/usr/bin/env bash

set -u

SCENE="../pbrt-v4-scenes/crown/crown.pbrt"
OUTDIR="experiments/E1"
TIMINGS_FILE="$OUTDIR/timings.csv"
LOG_FILE="$OUTDIR/run.log"

mkdir -p "$OUTDIR"

echo "case,output,elapsed_seconds,exit_status" > "$TIMINGS_FILE"
echo "==== Experiment E1 started at $(date) ====" | tee "$LOG_FILE"

run_case() {
    local case_name="$1"
    local output_file="$2"
    shift 2

    echo "" | tee -a "$LOG_FILE"
    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Running: $case_name" | tee -a "$LOG_FILE"
    echo "Output : $output_file" | tee -a "$LOG_FILE"
    echo "Start  : $(date)" | tee -a "$LOG_FILE"

    local start_ns end_ns elapsed_ns elapsed_sec status
    start_ns=$(date +%s%N)

    "$@" --outfile "$output_file" "$SCENE"
    status=$?

    end_ns=$(date +%s%N)
    elapsed_ns=$((end_ns - start_ns))
    elapsed_sec=$(awk "BEGIN {printf \"%.3f\", $elapsed_ns/1000000000}")

    echo "End    : $(date)" | tee -a "$LOG_FILE"
    echo "Status : $status" | tee -a "$LOG_FILE"
    echo "Time   : ${elapsed_sec}s" | tee -a "$LOG_FILE"

    echo "${case_name},${output_file},${elapsed_sec},${status}" >> "$TIMINGS_FILE"
}

# --------------------------------------------------
# 1) Ground Truth
# --------------------------------------------------
run_case "GT_PBRT_2048" "$OUTDIR/GT_PBRT_2048.exr" \
    ./build/pbrt \
    --gpu \
    --spp 2048 \
    --seed 0

# --------------------------------------------------
# 2) Controls
# --------------------------------------------------

# C0 - PBRT normal (sin NRC)
run_case "C0_PBRT_64" "$OUTDIR/C0_PBRT_64.exr" \
    ./build/pbrt \
    --gpu \
    --spp 64 \
    --seed 0

# C1 - Full NRC
run_case "C1_Full_NRC" "$OUTDIR/C1_Full_NRC.exr" \
    ./build/pbrt \
    --gpu \
    --enable-nrc \
    --spp 64 \
    --seed 0 \
    --nrc-warmup-samples 16 \
    --nrc-train-steps 2 \
    --nrc-brightness-oversampling=true \
    --nrc-config src/nrc/nrc_config.json

# C2 - Zero-input NRC
run_case "C2_Zero_Inputs_NRC" "$OUTDIR/C2_Zero_Inputs_NRC.exr" \
    ./build/pbrt \
    --gpu \
    --enable-nrc \
    --spp 64 \
    --seed 0 \
    --nrc-warmup-samples 16 \
    --nrc-train-steps 2 \
    --nrc-brightness-oversampling=true \
    --nrc-zero-inputs 0,52 \
    --nrc-config src/nrc/nrc_config.json

# --------------------------------------------------
# 3) Ablations (random order)
# --------------------------------------------------

ABLATIONS=(
    "F1_No_Position|$OUTDIR/F1_No_Position.exr|0,35"
    "F2_No_Outgoing_Direction|$OUTDIR/F2_No_Outgoing_Direction.exr|36,37"
    "F3_No_Shading_Normal|$OUTDIR/F3_No_Shading_Normal.exr|38,39"
    "F4_No_Roughness|$OUTDIR/F4_No_Roughness.exr|40,40"
    "F5_No_Diffuse_Reflectance|$OUTDIR/F5_No_Diffuse_Reflectance.exr|41,43"
    "F6_No_Specular_Reflectance|$OUTDIR/F6_No_Specular_Reflectance.exr|44,46"
)

echo "" | tee -a "$LOG_FILE"
echo "Randomizing ablation order..." | tee -a "$LOG_FILE"

mapfile -t SHUFFLED_ABLATIONS < <(printf "%s\n" "${ABLATIONS[@]}" | shuf)

for entry in "${SHUFFLED_ABLATIONS[@]}"; do
    IFS="|" read -r case_name output_file zero_range <<< "$entry"

    run_case "$case_name" "$output_file" \
        ./build/pbrt \
        --gpu \
        --enable-nrc \
        --spp 64 \
        --seed 0 \
        --nrc-warmup-samples 16 \
        --nrc-train-steps 2 \
        --nrc-brightness-oversampling=true \
        --nrc-zero-inputs "$zero_range" \
        --nrc-config src/nrc/nrc_config.json
done

echo "" | tee -a "$LOG_FILE"
echo "==== Experiment E1 finished at $(date) ====" | tee -a "$LOG_FILE"
echo "Timings saved to: $TIMINGS_FILE" | tee -a "$LOG_FILE"