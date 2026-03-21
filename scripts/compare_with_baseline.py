#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Comparison Script
Compare verification results with MiniMax-M2.5 baseline and calculate metrics

Usage:
    python3 scripts/compare_with_baseline.py \
        --result-dir output-dir \
        --baseline-dir MiniMax-M2.5 \
        --provider minimax \
        --output output-dir/comparison_report.json \
        --send-report
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add scripts directory to Python path
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calculate_toolcall_similarity import calculate_tool_call_f1


def find_summary_files(root_dir: str, provider: str = None) -> List[str]:
    """Recursively find all summary.json files"""
    if provider:
        pattern = os.path.join(root_dir, "**", f"{provider}_summary.json")
    else:
        pattern = os.path.join(root_dir, "**", "*_summary.json")
    
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def find_results_files(root_dir: str, provider: str = None) -> List[str]:
    """Recursively find all results.jsonl files"""
    if provider:
        pattern = os.path.join(root_dir, "**", f"{provider}_results.jsonl")
    else:
        pattern = os.path.join(root_dir, "**", "*_results.jsonl")
    
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def load_json(file_path: str) -> Optional[Dict]:
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Cannot read file {file_path}: {e}")
        return None


def load_jsonl(file_path: str) -> Optional[List[Dict]]:
    """Load JSONL file"""
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"❌ Cannot read file {file_path}: {e}")
        return None


def calculate_metrics_from_summary(data: Dict) -> Dict:
    """Calculate metrics from summary data"""
    metrics = {}
    
    all_count = data.get('all_count', 0)
    success_count = data.get('success_count', 0)
    metrics['Query-Success-Rate'] = success_count / all_count if all_count > 0 else 0
    
    tool_calls_finish_tool_calls = data.get('tool_calls_finish_tool_calls', 0)
    stop_finish_stop = data.get('stop_finish_stop', 0)
    expected_tool_call_total_count = data.get('expected_tool_call_total_count', 0)
    metrics['ToolCalls-Match-Rate'] = (tool_calls_finish_tool_calls + stop_finish_stop) / expected_tool_call_total_count if expected_tool_call_total_count > 0 else 0
    
    tool_calls_successful_count = data.get('tool_calls_successful_count', 0)
    metrics['ToolCalls-Accuracy'] = tool_calls_successful_count / tool_calls_finish_tool_calls if tool_calls_finish_tool_calls > 0 else 0
    
    error_only_reasoning_count = data.get('error_only_reasoning_count', 0)
    error_only_reasoning_checked_count = data.get('error_only_reasoning_checked_count', 0)
    metrics['Response-Success-Rate-Not-Only-Reasoning'] = (
        (error_only_reasoning_checked_count - error_only_reasoning_count) / error_only_reasoning_checked_count 
        if error_only_reasoning_checked_count > 0 else 0
    )
    
    language_following_valid_count = data.get('language_following_valid_count', 0)
    language_following_checked_count = data.get('language_following_checked_count', 0)
    metrics['Language-Following-Success-Rate'] = (
        language_following_valid_count / language_following_checked_count 
        if language_following_checked_count > 0 else 0
    )
    
    # Token Usage
    token_usage = data.get('token_usage', {})
    if token_usage:
        metrics['token_usage'] = token_usage
    
    return metrics


def match_loops(result_dir: str, baseline_dir: str, provider: str) -> List[Tuple[str, str, str, str]]:
    """
    Match loops between result directory and baseline directory
    
    Returns:
        List of (result_summary, result_jsonl, baseline_summary, baseline_jsonl)
    """
    matches = []
    
    # Check if multi-loop mode
    result_loops = sorted(glob.glob(os.path.join(result_dir, "loop_*")))
    baseline_loops = sorted(glob.glob(os.path.join(baseline_dir, "loop_*")))
    
    if result_loops and baseline_loops:
        # Multi-loop mode: match by loop
        print(f"📁 Multi-loop mode:")
        print(f"   Result directory loops: {len(result_loops)}")
        print(f"   Baseline directory loops: {len(baseline_loops)}")
        
        for result_loop in result_loops:
            loop_name = os.path.basename(result_loop)
            baseline_loop = os.path.join(baseline_dir, loop_name)
            
            if not os.path.exists(baseline_loop):
                print(f"   ⚠️  {loop_name}: No match in baseline directory")
                continue
            
            # Find files for this loop
            result_summary = find_summary_files(result_loop, provider)
            # Prefer minimax_summary.json, fallback to any *_summary.json
            baseline_summary = find_summary_files(baseline_loop, "minimax")
            if not baseline_summary:
                baseline_summary = find_summary_files(baseline_loop, None)  # Find any summary file
            
            if not result_summary:
                print(f"   ⚠️  {loop_name}: No summary file in result directory")
                continue
            if not baseline_summary:
                print(f"   ⚠️  {loop_name}: No summary file in baseline directory")
                continue
            
            # Find results.jsonl
            result_jsonl_path = result_summary[0].replace('_summary.json', '_results.jsonl')
            # Prefer minimax_results.jsonl, fallback to any *_results.jsonl
            baseline_jsonl = find_results_files(baseline_loop, "minimax")
            if not baseline_jsonl:
                baseline_jsonl = find_results_files(baseline_loop, None)  # Find any results file
            
            if not os.path.exists(result_jsonl_path):
                result_jsonl_path = None
            
            baseline_jsonl_path = baseline_jsonl[0] if baseline_jsonl else None
            
            matches.append((
                result_summary[0],
                result_jsonl_path,
                baseline_summary[0],
                baseline_jsonl_path
            ))
            print(f"   ✅ {loop_name}: Match successful")
    
    else:
        # Single run mode: direct match
        print(f"📁 Single run mode")
        
        result_summary = find_summary_files(result_dir, provider)
        # Prefer minimax_summary.json, fallback to any *_summary.json
        baseline_summary = find_summary_files(baseline_dir, "minimax")
        if not baseline_summary:
            baseline_summary = find_summary_files(baseline_dir, None)  # Find any summary file
        
        if result_summary and baseline_summary:
            result_jsonl_path = result_summary[0].replace('_summary.json', '_results.jsonl')
            baseline_jsonl = find_results_files(baseline_dir, "minimax")
            
            if not os.path.exists(result_jsonl_path):
                result_jsonl_path = None
            
            baseline_jsonl_path = baseline_jsonl[0] if baseline_jsonl else None
            
            matches.append((
                result_summary[0],
                result_jsonl_path,
                baseline_summary[0],
                baseline_jsonl_path
            ))
            print(f"   ✅ Match successful")
    
    return matches


def compare_single_loop(
    result_summary_path: str,
    result_jsonl_path: Optional[str],
    baseline_summary_path: str,
    baseline_jsonl_path: Optional[str]
) -> Dict:
    """Compare results for a single loop"""
    result = {
        'result_file': result_summary_path,
        'baseline_file': baseline_summary_path,
    }
    
    # Load summary data
    result_summary = load_json(result_summary_path)
    baseline_summary = load_json(baseline_summary_path)
    
    if not result_summary or not baseline_summary:
        result['status'] = 'error'
        result['error'] = 'Failed to load summary files'
        return result
    
    result['model'] = result_summary.get('model', 'Unknown')
    result['baseline_model'] = baseline_summary.get('model', 'Unknown')
    
    # Calculate base metrics
    result['metrics'] = calculate_metrics_from_summary(result_summary)
    result['baseline_metrics'] = calculate_metrics_from_summary(baseline_summary)
    
    # Calculate ToolCalls-Trigger-Similarity (if results.jsonl exists)
    if result_jsonl_path and baseline_jsonl_path:
        result_data = load_jsonl(result_jsonl_path)
        baseline_data = load_jsonl(baseline_jsonl_path)
        
        if result_data and baseline_data:
            try:
                # Align sample count
                min_len = min(len(result_data), len(baseline_data))
                if len(result_data) != len(baseline_data):
                    print(f"   ⚠️  Sample count mismatch (result: {len(result_data)}, baseline: {len(baseline_data)}), using first {min_len}")
                
                f1_metrics = calculate_tool_call_f1(baseline_data[:min_len], result_data[:min_len])
                result['toolcalls_trigger_similarity'] = {
                    'f1': f1_metrics['f1'],
                    'precision': f1_metrics['precision'],
                    'recall': f1_metrics['recall'],
                    'tp': f1_metrics['tp'],
                    'fp': f1_metrics['fp'],
                    'fn': f1_metrics['fn'],
                    'tn': f1_metrics['tn'],
                    'samples_used': min_len
                }
                result['metrics']['ToolCalls-Trigger-Similarity-F1'] = f1_metrics['f1']
            except Exception as e:
                print(f"   ⚠️  Failed to calculate ToolCalls-Trigger-Similarity: {e}")
                result['metrics']['ToolCalls-Trigger-Similarity-F1'] = None
        else:
            result['metrics']['ToolCalls-Trigger-Similarity-F1'] = None
    else:
        result['metrics']['ToolCalls-Trigger-Similarity-F1'] = None
        if not result_jsonl_path:
            print(f"   ⚠️  No results.jsonl in result directory, cannot calculate ToolCalls-Trigger-Similarity")
        if not baseline_jsonl_path:
            print(f"   ⚠️  No minimax_results.jsonl in baseline directory, cannot calculate ToolCalls-Trigger-Similarity")
    
    result['status'] = 'success'
    return result


def calculate_average_metrics(all_results: List[Dict]) -> Dict:
    """Calculate average metrics across all loops"""
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        return {}
    
    metric_keys = [
        'Query-Success-Rate',
        'ToolCalls-Match-Rate',
        'ToolCalls-Trigger-Similarity-F1',
        'ToolCalls-Accuracy',
        'Response-Success-Rate-Not-Only-Reasoning',
        'Language-Following-Success-Rate'
    ]
    
    avg_metrics = {}
    for key in metric_keys:
        values = []
        for r in successful_results:
            val = r['metrics'].get(key)
            if val is not None:
                values.append(val)
        if values:
            avg_metrics[key] = sum(values) / len(values)
    
    # Token Usage 平均
    token_usage_list = []
    for r in successful_results:
        if 'token_usage' in r['metrics']:
            token_usage_list.append(r['metrics']['token_usage'])
    
    if token_usage_list:
        avg_token_usage = {}
        token_types = ['prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens', 'cached_tokens']
        
        for token_type in token_types:
            averages = []
            totals = []
            for tu in token_usage_list:
                if token_type in tu and isinstance(tu[token_type], dict):
                    averages.append(tu[token_type].get('average', 0))
                    totals.append(tu[token_type].get('total', 0))
            
            if averages:
                avg_token_usage[token_type] = {
                    'average': sum(averages) / len(averages),
                    'total': sum(totals)
                }
        
        if avg_token_usage:
            avg_metrics['Token-Usage'] = avg_token_usage
    
    return avg_metrics


def print_comparison_table(all_results: List[Dict], avg_metrics: Dict):
    """Print comparison results table"""
    print("\n" + "=" * 120)
    print("📊 Comparison Results Summary")
    print("=" * 120)
    
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        print("❌ No successful comparison results")
        return
    
    has_similarity = any(r['metrics'].get('ToolCalls-Trigger-Similarity-F1') is not None for r in successful_results)
    
    # Header
    if has_similarity:
        print(f"\n{'Loop':<15} {'Model':<25} {'Q-Succ':<10} {'F-Tool':<10} {'TC-Sim-F1':<12} {'Tool-Acc':<10} {'R-Succ':<10} {'Lang-Succ':<10}")
        print("-" * 120)
    else:
        print(f"\n{'Loop':<15} {'Model':<25} {'Q-Succ':<10} {'F-Tool':<10} {'Tool-Acc':<10} {'R-Succ':<10} {'Lang-Succ':<10}")
        print("-" * 100)
    
    for i, result in enumerate(successful_results, 1):
        loop_name = f"Loop {i}"
        model = result.get('model', 'Unknown')
        if len(model) > 23:
            model = model[:20] + "..."
        
        metrics = result['metrics']
        
        if has_similarity:
            sim_f1 = metrics.get('ToolCalls-Trigger-Similarity-F1')
            sim_str = f"{sim_f1*100:>10.2f}%" if sim_f1 is not None else "N/A".rjust(12)
            print(f"{loop_name:<15} {model:<25} "
                  f"{metrics.get('Query-Success-Rate', 0)*100:>8.2f}% "
                  f"{metrics.get('ToolCalls-Match-Rate', 0)*100:>8.2f}% "
                  f"{sim_str} "
                  f"{metrics.get('ToolCalls-Accuracy', 0)*100:>8.2f}% "
                  f"{metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0)*100:>8.2f}% "
                  f"{metrics.get('Language-Following-Success-Rate', 0)*100:>8.2f}%")
        else:
            print(f"{loop_name:<15} {model:<25} "
                  f"{metrics.get('Query-Success-Rate', 0)*100:>8.2f}% "
                  f"{metrics.get('ToolCalls-Match-Rate', 0)*100:>8.2f}% "
                  f"{metrics.get('ToolCalls-Accuracy', 0)*100:>8.2f}% "
                  f"{metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0)*100:>8.2f}% "
                  f"{metrics.get('Language-Following-Success-Rate', 0)*100:>8.2f}%")
    
    print("=" * (120 if has_similarity else 100))
    
    # Print averages
    if avg_metrics:
        print(f"\n📈 Average Metrics ({len(successful_results)} comparisons):")
        print(f"  1. Query-Success-Rate: {avg_metrics.get('Query-Success-Rate', 0):.4f} ({avg_metrics.get('Query-Success-Rate', 0)*100:.2f}%)")
        print(f"  2. ToolCalls-Match-Rate: {avg_metrics.get('ToolCalls-Match-Rate', 0):.4f} ({avg_metrics.get('ToolCalls-Match-Rate', 0)*100:.2f}%)")
        if 'ToolCalls-Trigger-Similarity-F1' in avg_metrics:
            print(f"  3. ToolCalls-Trigger-Similarity-F1: {avg_metrics['ToolCalls-Trigger-Similarity-F1']:.4f} ({avg_metrics['ToolCalls-Trigger-Similarity-F1']*100:.2f}%)")
        print(f"  4. ToolCalls-Accuracy: {avg_metrics.get('ToolCalls-Accuracy', 0):.4f} ({avg_metrics.get('ToolCalls-Accuracy', 0)*100:.2f}%)")
        print(f"  5. Response-Success-Rate-Not-Only-Reasoning: {avg_metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0):.4f} ({avg_metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0)*100:.2f}%)")
        print(f"  6. Language-Following-Success-Rate: {avg_metrics.get('Language-Following-Success-Rate', 0):.4f} ({avg_metrics.get('Language-Following-Success-Rate', 0)*100:.2f}%)")
        
        # Token Usage
        if 'Token-Usage' in avg_metrics:
            print(f"\n  Token Usage Statistics Averages:")
            tu = avg_metrics['Token-Usage']
            if 'prompt_tokens' in tu:
                print(f"    - Avg Prompt Tokens: {tu['prompt_tokens']['average']:.2f}")
            if 'completion_tokens' in tu:
                print(f"    - Avg Completion Tokens: {tu['completion_tokens']['average']:.2f}")
            if 'reasoning_tokens' in tu:
                print(f"    - Avg Reasoning Tokens: {tu['reasoning_tokens']['average']:.2f}")
            if 'total_tokens' in tu:
                print(f"    - Avg Total Tokens: {tu['total_tokens']['average']:.2f}")


def send_report(summary: Dict, api_url: Optional[str] = None) -> bool:
    """Send report to API"""
    import requests
    
    if not api_url:
        api_url = os.environ.get(
            'METRICS_REPORT_API_URL',
            'https://swing.xaminim.com/minimax/provider/report/send'
        )
    
    print(f"\n📧 Sending report to: {api_url}")
    
    try:
        # Get config from environment variable
        provider_config_str = os.environ.get('PROVIDER_VERIFIER_CONFIG', '')
        provider_config = {}
        if provider_config_str:
            try:
                provider_config = json.loads(provider_config_str)
                print(f"  📦 Loaded PROVIDER_VERIFIER_CONFIG")
            except json.JSONDecodeError:
                pass
        
        # Build request data (consistent with send_metrics_report.py)
        payload = {
            'average_metrics': summary.get('average_metrics', {}),
            **provider_config
        }
        
        # Note: key is average_metrics, corresponds to aggregated_metrics in send_metrics_report.py
        # Both have the same content structure - average metrics
        
        print("\n📤 Request Body:")
        print("-" * 60)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("-" * 60)
        
        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            print("\n✅ Report sent successfully")
            try:
                result = response.json()
                print(f"  Response: {result}")
            except:
                pass
            return True
        else:
            print(f"\n❌ Failed to send report: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Failed to send report: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare with baseline data and calculate metrics'
    )
    
    parser.add_argument(
        '--result-dir',
        type=str,
        required=True,
        help='Verification result directory'
    )
    
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default='MiniMax-M2.5',
        help='Baseline data directory (default: MiniMax-M2.5)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='Provider name (auto-detected)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output report file path'
    )
    
    parser.add_argument(
        '--send-report',
        action='store_true',
        help='Send report to API'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='Report API URL'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 MiniMax Provider Verifier - Baseline Comparison")
    print("=" * 60)
    
    # Check directories
    if not os.path.exists(args.result_dir):
        print(f"❌ Result directory does not exist: {args.result_dir}")
        return 1
    
    # Parse baseline directory (supports relative paths)
    baseline_dir = args.baseline_dir
    if not os.path.isabs(baseline_dir):
        # Try multiple possible locations
        possible_paths = [
            baseline_dir,
            os.path.join(os.path.dirname(args.result_dir), baseline_dir),
            os.path.join(os.getcwd(), baseline_dir),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                baseline_dir = path
                break
    
    if not os.path.exists(baseline_dir):
        print(f"❌ Baseline directory does not exist: {baseline_dir}")
        return 1
    
    print(f"📂 Result directory: {args.result_dir}")
    print(f"📂 Baseline directory: {baseline_dir}")
    
    # Auto-detect provider
    provider = args.provider
    if not provider:
        summary_files = find_summary_files(args.result_dir)
        if summary_files:
            # Extract provider from filename
            basename = os.path.basename(summary_files[0])
            provider = basename.replace('_summary.json', '')
            print(f"📦 Auto-detected provider: {provider}")
    
    if not provider:
        print("❌ Cannot auto-detect provider, please use --provider argument")
        return 1
    
    # Match loops
    print("\n🔍 Matching results with baseline...")
    matches = match_loops(args.result_dir, baseline_dir, provider)
    
    if not matches:
        print("❌ No matching data found")
        return 1
    
    print(f"\n✅ Found {len(matches)} matches")
    
    # Execute comparison
    print("\n🔄 Starting comparison...")
    all_results = []
    
    for i, (result_summary, result_jsonl, baseline_summary, baseline_jsonl) in enumerate(matches, 1):
        print(f"\n--- Comparison {i}/{len(matches)} ---")
        print(f"   Result: {result_summary}")
        print(f"   Baseline: {baseline_summary}")
        
        result = compare_single_loop(
            result_summary, result_jsonl,
            baseline_summary, baseline_jsonl
        )
        all_results.append(result)
    
    # Calculate average metrics
    avg_metrics = calculate_average_metrics(all_results)
    
    # Print results
    print_comparison_table(all_results, avg_metrics)
    
    # Build final report
    report = {
        'provider': provider,
        'baseline_dir': baseline_dir,
        'result_dir': args.result_dir,
        'total_comparisons': len(all_results),
        'successful_comparisons': len([r for r in all_results if r.get('status') == 'success']),
        'generated_at': datetime.now().isoformat(),
        'average_metrics': avg_metrics,
        'comparisons': all_results
    }
    
    # Save report
    output_file = args.output
    if not output_file:
        output_file = os.path.join(args.result_dir, 'comparison_report.json')
    
    print(f"\n💾 Saving report to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("✅ Report saved successfully")
    
    # Send report
    if args.send_report:
        send_report(report, args.api_url)
    
    print("\n🎉 Comparison completed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())