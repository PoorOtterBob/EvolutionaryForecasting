#!/usr/bin/env bash
# Evolutionary Forecasting on TFB (https://github.com/decisionintelligence/TFB).
#
# TFB already rolls a deep model forward block by block whenever the evaluation
# horizon (--strategy-args "horizon") exceeds the model horizon
# (--model-hyper-params "horizon"); see DeepForecastingModelBase.batch_forecast
# and _perform_rolling_predictions. EF therefore needs no code change in TFB:
# train ONE model with block length L and evaluate it at every horizon H.
#
# Run from the TFB repository root:
#   bash /path/to/EvolutionaryForecasting/integrations/tfb/run_ef_tfb.sh
# Override any setting through the environment, e.g. MODEL=iTransformer L=192 bash run_ef_tfb.sh

DATA=${DATA:-ETTh1.csv}
MODEL=${MODEL:-PatchTST}          # any time_series_library.* model shipped with TFB
SEQ_LEN=${SEQ_LEN:-720}           # input length T
L=${L:-96}                        # block length: the model's own output horizon
HORIZONS=${HORIZONS:-"96 192 336 720"}
HPARAMS=${HPARAMS:-'"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4, "norm": true'}
GPU=${GPU:-0}

for H in $HORIZONS; do
  python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "$DATA" \
    --strategy-args "{\"horizon\": $H}" \
    --model-name "time_series_library.$MODEL" \
    --model-hyper-params "{\"horizon\": $L, \"seq_len\": $SEQ_LEN, $HPARAMS}" \
    --adapter "transformer_adapter" \
    --gpus "$GPU" --num-workers 1 --timeout 60000 \
    --save-path "EF/${DATA%.csv}/${MODEL}_T${SEQ_LEN}_L${L}"
done
