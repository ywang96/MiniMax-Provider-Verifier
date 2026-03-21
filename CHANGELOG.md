# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-03-19

### Added
- **Batch verification script** (`run_batch_sequential.sh`): New script for running pass@10 verification tests
  - Supports 10 iterations by default for statistical significance
  - Auto-generates metrics reports and baseline comparisons
  - Parameters: `--module`, `--url`, `--model`, `--api-key`, `--max-workers`, `--mm-model`, etc.
- Added `expected_tool_call_total_count` to summary statistics in `validator/tool_calls.py`
- Added batch testing documentation to README.md and README_CN.md
- New script dependencies:
  - `scripts/batch_verify.py`: Batch verification executor
  - `scripts/calculate_batch_metrics.py`: Metrics aggregation and calculation
  - `scripts/compare_with_baseline.py`: Baseline comparison tool
  - `scripts/calculate_toolcall_similarity.py`: Tool call similarity calculator

### Changed
- **ToolCalls-Match-Rate denominator**: Now uses `expected_tool_call` label count from `sample.jsonl` as the denominator for more accurate match rate calculation
- All Chinese comments and outputs converted to English for better internationalization
- `calculate_batch_metrics.py` now reads `expected_tool_call` statistics directly from `sample.jsonl`

### Fixed
- Fixed path consistency issues in shell script calling Python scripts

## [1.1.0] - 2026-03-18

### Changed
- **ToolCalls-Match-Rate redefined**: Changed from simple "proportion of triggered tool calls" to a match rate based on expected labels
  - New formula: `tool_calls_accuracy = (tool_calls_finish_tool_calls + stop_finish_stop) / success_count`
  - i.e., proportion of cases where actual result matches expected result

### Added
- Added `expected_tool_call` label field to test set `sample.jsonl`, indicating whether each case is expected to trigger a tool call
- Added confusion matrix statistics:
  - `tool_calls_finish_tool_calls`: expected tool_call, actual tool_call (True Positive)
  - `tool_calls_finish_stop`: expected tool_call, actual stop (False Negative)
  - `stop_finish_tool_calls`: expected stop, actual tool_call (False Positive)
  - `stop_finish_stop`: expected stop, actual stop (True Negative)
- Added `expected_tool_call` field to results, recording the expected label from the original request

### Fixed
- Backward compatible with historical data without `expected_tool_call` label (incremental mode)

## [1.0.0] - 2026-01-18

### Changed
- Stable release

## [0.1.0] - 2025-11-18

### Added
- Initial release of MiniMax Provider Verifier
- Support for multiple validators (ToolCalls, Russian Characters, Repeat N-Gram)
- Concurrent request processing
- Batch provider testing
- Incremental mode for rerunning failed requests
- Dynamic validator selection based on check_type
- Detailed test reports and statistical summaries
- Custom validator support

