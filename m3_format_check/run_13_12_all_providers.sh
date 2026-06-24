#!/bin/bash
# Run test_13_12_tool_name_mismatch_prompt against all M3 providers.
# Each provider runs in a clean env via subshell; results saved to logs/13_12_<ts>/.

set +e  # do not stop on a single provider failure

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="logs/13_12_${TS}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.txt"
echo "=== M3 test_13_12 (tool_name_mismatch_prompt) verification @ ${TS} UTC ===" | tee "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_one () {
  local name="$1"
  local url="$2"
  local key="$3"
  local model="$4"
  local log="$OUT_DIR/${name}.log"
  echo "----- $name -----" | tee -a "$SUMMARY"
  echo "URL=$url" | tee -a "$SUMMARY"
  echo "Model=$model" | tee -a "$SUMMARY"
  local started=$(date +%s)
  M3_BASE_URL="$url" M3_API_KEY="$key" M3_MODEL="$model" M3_RUN_LOG="$OUT_DIR/${name}.jsonl" \
    python3 -m pytest -k test_13_12 -v --timeout=300 --no-header 2>&1 | tee "$log" >/dev/null
  local rc=${PIPESTATUS[0]}
  local elapsed=$(( $(date +%s) - started ))
  grep -E "test_13_12.*(PASSED|FAILED|ERROR|SKIPPED)" "$log" | sed 's/.*::test_13_12_tool_name_mismatch_prompt/  13_12/' | tee -a "$SUMMARY"
  echo "  -> rc=$rc, elapsed=${elapsed}s" | tee -a "$SUMMARY"
  echo "" | tee -a "$SUMMARY"
}

run_one "official" \
  "https://api.minimaxi.com" \
  "sk-api-vKJ3B0k7Zx8N4CJBybUASNHIZ_GcINLybUcxl44EUpmMZ3oF3ZsjmjROzOKTO2CUBSUVRu51ggmol_BoQ1acEQ5PHIcjmBwbHyAJim-uf6QAIA253XG-cF4" \
  "minimax-m3"

run_one "fireworks" \
  "https://api.fireworks.ai/inference" \
  "fw_9rDXpaQQ3ibAo6VJss638g" \
  "accounts/sanchuan-zhengyu/routers/minimax-m3"

run_one "together_b300" \
  "https://us-west-2.api-aws.together.ai" \
  "tgp_v1_Rj7MGLKklqcUTkGlFEv7rFWe8tEwvOzDdkmzlOpdh14" \
  "minimax/minimax-m3-0602-dedicated-b200"

run_one "together_gb300" \
  "https://us-west-2.api-aws.together.ai" \
  "tgp_v1_Rj7MGLKklqcUTkGlFEv7rFWe8tEwvOzDdkmzlOpdh14" \
  "minimax/minimax-m3-0602-dedicated"

run_one "together_fp4" \
  "https://us-west-2.api-aws.together.ai" \
  "tgp_v1_Rj7MGLKklqcUTkGlFEv7rFWe8tEwvOzDdkmzlOpdh14" \
  "shadow/minimax-m3-0603-fp4-gb300"

run_one "huawei" \
  "https://api.modelarts-maas.com" \
  "IMfveeS44GB7pPf9r7OsYWgtT5y868ANsaWA9goUnMemS-DMHPqJGeD4hWhA4TYpwXtlT_aQSKxHdOwcLB1Pog" \
  "Minimax-M3-0608"

run_one "magikcloud" \
  "https://api.magikcloud.cn" \
  "magik-19b3207790fe4a30a8545d736a760754" \
  "ep-minimax-m3-1136e9"

run_one "inferact" \
  "https://minimax.svc.inferact.dev" \
  "sk-inf-4ca7a278ded31ef3d5a57f3ee559528fb2dcdbe27601e8be6c46d6270b8df943" \
  "MiniMaxAI/MiniMax-M3-NVFP4"

run_one "modular" \
  "https://minimax.api.modular.com" \
  "sk-mod-minimax-af8e3407d78f77a388fc95943eb3c5ba833a32605750bf98aeb0be567f38b72e" \
  "MiniMaxAI/MiniMax-M3"

echo "=== Done, summary at $SUMMARY ==="
