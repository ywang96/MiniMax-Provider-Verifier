#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
与基准数据对比脚本
使用 MiniMax-M2.5 作为基准，对比验证结果并计算指标

使用方式:
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

# 添加 scripts 目录到 Python 路径
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calculate_toolcall_similarity import calculate_tool_call_f1


def find_summary_files(root_dir: str, provider: str = None) -> List[str]:
    """递归查找所有 summary.json 文件"""
    if provider:
        pattern = os.path.join(root_dir, "**", f"{provider}_summary.json")
    else:
        pattern = os.path.join(root_dir, "**", "*_summary.json")
    
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def find_results_files(root_dir: str, provider: str = None) -> List[str]:
    """递归查找所有 results.jsonl 文件"""
    if provider:
        pattern = os.path.join(root_dir, "**", f"{provider}_results.jsonl")
    else:
        pattern = os.path.join(root_dir, "**", "*_results.jsonl")
    
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def load_json(file_path: str) -> Optional[Dict]:
    """加载 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}: {e}")
        return None


def load_jsonl(file_path: str) -> Optional[List[Dict]]:
    """加载 JSONL 文件"""
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}: {e}")
        return None


def calculate_metrics_from_summary(data: Dict) -> Dict:
    """根据 summary 数据计算各项指标"""
    metrics = {}
    
    all_count = data.get('all_count', 0)
    success_count = data.get('success_count', 0)
    metrics['Query-Success-Rate'] = success_count / all_count if all_count > 0 else 0
    
    finish_tool_calls_count = data.get('tool_calls_finish_tool_calls', 0)
    metrics['Finish-ToolCalls-Rate'] = finish_tool_calls_count / all_count if all_count > 0 else 0
    
    tool_calls_successful_count = data.get('tool_calls_successful_count', 0)
    metrics['ToolCalls-Accuracy'] = tool_calls_successful_count / finish_tool_calls_count if finish_tool_calls_count > 0 else 0
    
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
    匹配结果目录和基准目录的循环
    
    Returns:
        List of (result_summary, result_jsonl, baseline_summary, baseline_jsonl)
    """
    matches = []
    
    # 检查是否是多循环模式
    result_loops = sorted(glob.glob(os.path.join(result_dir, "loop_*")))
    baseline_loops = sorted(glob.glob(os.path.join(baseline_dir, "loop_*")))
    
    if result_loops and baseline_loops:
        # 多循环模式：按循环匹配
        print(f"📁 多循环模式:")
        print(f"   结果目录循环数: {len(result_loops)}")
        print(f"   基准目录循环数: {len(baseline_loops)}")
        
        for result_loop in result_loops:
            loop_name = os.path.basename(result_loop)
            baseline_loop = os.path.join(baseline_dir, loop_name)
            
            if not os.path.exists(baseline_loop):
                print(f"   ⚠️  {loop_name}: 基准目录无匹配")
                continue
            
            # 查找该循环的文件
            result_summary = find_summary_files(result_loop, provider)
            # 优先查找 minimax_summary.json，如果找不到则查找任意 *_summary.json
            baseline_summary = find_summary_files(baseline_loop, "minimax")
            if not baseline_summary:
                baseline_summary = find_summary_files(baseline_loop, None)  # 查找任意 summary 文件
            
            if not result_summary:
                print(f"   ⚠️  {loop_name}: 结果目录无 summary 文件")
                continue
            if not baseline_summary:
                print(f"   ⚠️  {loop_name}: 基准目录无 summary 文件")
                continue
            
            # 查找 results.jsonl
            result_jsonl_path = result_summary[0].replace('_summary.json', '_results.jsonl')
            # 优先查找 minimax_results.jsonl，如果找不到则查找任意 *_results.jsonl
            baseline_jsonl = find_results_files(baseline_loop, "minimax")
            if not baseline_jsonl:
                baseline_jsonl = find_results_files(baseline_loop, None)  # 查找任意 results 文件
            
            if not os.path.exists(result_jsonl_path):
                result_jsonl_path = None
            
            baseline_jsonl_path = baseline_jsonl[0] if baseline_jsonl else None
            
            matches.append((
                result_summary[0],
                result_jsonl_path,
                baseline_summary[0],
                baseline_jsonl_path
            ))
            print(f"   ✅ {loop_name}: 匹配成功")
    
    else:
        # 单次运行模式：直接匹配
        print(f"📁 单次运行模式")
        
        result_summary = find_summary_files(result_dir, provider)
        # 优先查找 minimax_summary.json，如果找不到则查找任意 *_summary.json
        baseline_summary = find_summary_files(baseline_dir, "minimax")
        if not baseline_summary:
            baseline_summary = find_summary_files(baseline_dir, None)  # 查找任意 summary 文件
        
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
            print(f"   ✅ 匹配成功")
    
    return matches


def compare_single_loop(
    result_summary_path: str,
    result_jsonl_path: Optional[str],
    baseline_summary_path: str,
    baseline_jsonl_path: Optional[str]
) -> Dict:
    """对比单个循环的结果"""
    result = {
        'result_file': result_summary_path,
        'baseline_file': baseline_summary_path,
    }
    
    # 加载 summary 数据
    result_summary = load_json(result_summary_path)
    baseline_summary = load_json(baseline_summary_path)
    
    if not result_summary or not baseline_summary:
        result['status'] = 'error'
        result['error'] = 'Failed to load summary files'
        return result
    
    result['model'] = result_summary.get('model', 'Unknown')
    result['baseline_model'] = baseline_summary.get('model', 'Unknown')
    
    # 计算基础指标
    result['metrics'] = calculate_metrics_from_summary(result_summary)
    result['baseline_metrics'] = calculate_metrics_from_summary(baseline_summary)
    
    # 计算 ToolCalls-Trigger-Similarity（如果有 results.jsonl）
    if result_jsonl_path and baseline_jsonl_path:
        result_data = load_jsonl(result_jsonl_path)
        baseline_data = load_jsonl(baseline_jsonl_path)
        
        if result_data and baseline_data:
            try:
                # 样本数对齐
                min_len = min(len(result_data), len(baseline_data))
                if len(result_data) != len(baseline_data):
                    print(f"   ⚠️  样本数不匹配 (result: {len(result_data)}, baseline: {len(baseline_data)})，使用前 {min_len} 个")
                
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
                print(f"   ⚠️  计算 ToolCalls-Trigger-Similarity 失败: {e}")
                result['metrics']['ToolCalls-Trigger-Similarity-F1'] = None
        else:
            result['metrics']['ToolCalls-Trigger-Similarity-F1'] = None
    else:
        result['metrics']['ToolCalls-Trigger-Similarity-F1'] = None
        if not result_jsonl_path:
            print(f"   ⚠️  结果目录无 results.jsonl，无法计算 ToolCalls-Trigger-Similarity")
        if not baseline_jsonl_path:
            print(f"   ⚠️  基准目录无 minimax_results.jsonl，无法计算 ToolCalls-Trigger-Similarity")
    
    result['status'] = 'success'
    return result


def calculate_average_metrics(all_results: List[Dict]) -> Dict:
    """计算所有循环的平均指标"""
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        return {}
    
    metric_keys = [
        'Query-Success-Rate',
        'Finish-ToolCalls-Rate',
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
    """打印对比结果表格"""
    print("\n" + "=" * 120)
    print("📊 对比结果汇总")
    print("=" * 120)
    
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        print("❌ 没有成功的对比结果")
        return
    
    has_similarity = any(r['metrics'].get('ToolCalls-Trigger-Similarity-F1') is not None for r in successful_results)
    
    # 表头
    if has_similarity:
        print(f"\n{'循环':<15} {'模型':<25} {'Q-Succ':<10} {'F-Tool':<10} {'TC-Sim-F1':<12} {'Tool-Acc':<10} {'R-Succ':<10} {'Lang-Succ':<10}")
        print("-" * 120)
    else:
        print(f"\n{'循环':<15} {'模型':<25} {'Q-Succ':<10} {'F-Tool':<10} {'Tool-Acc':<10} {'R-Succ':<10} {'Lang-Succ':<10}")
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
                  f"{metrics.get('Finish-ToolCalls-Rate', 0)*100:>8.2f}% "
                  f"{sim_str} "
                  f"{metrics.get('ToolCalls-Accuracy', 0)*100:>8.2f}% "
                  f"{metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0)*100:>8.2f}% "
                  f"{metrics.get('Language-Following-Success-Rate', 0)*100:>8.2f}%")
        else:
            print(f"{loop_name:<15} {model:<25} "
                  f"{metrics.get('Query-Success-Rate', 0)*100:>8.2f}% "
                  f"{metrics.get('Finish-ToolCalls-Rate', 0)*100:>8.2f}% "
                  f"{metrics.get('ToolCalls-Accuracy', 0)*100:>8.2f}% "
                  f"{metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0)*100:>8.2f}% "
                  f"{metrics.get('Language-Following-Success-Rate', 0)*100:>8.2f}%")
    
    print("=" * (120 if has_similarity else 100))
    
    # 打印平均值
    if avg_metrics:
        print(f"\n📈 平均指标 (共 {len(successful_results)} 次对比):")
        print(f"  1. Query-Success-Rate: {avg_metrics.get('Query-Success-Rate', 0):.4f} ({avg_metrics.get('Query-Success-Rate', 0)*100:.2f}%)")
        print(f"  2. Finish-ToolCalls-Rate: {avg_metrics.get('Finish-ToolCalls-Rate', 0):.4f} ({avg_metrics.get('Finish-ToolCalls-Rate', 0)*100:.2f}%)")
        if 'ToolCalls-Trigger-Similarity-F1' in avg_metrics:
            print(f"  3. ToolCalls-Trigger-Similarity-F1: {avg_metrics['ToolCalls-Trigger-Similarity-F1']:.4f} ({avg_metrics['ToolCalls-Trigger-Similarity-F1']*100:.2f}%)")
        print(f"  4. ToolCalls-Accuracy: {avg_metrics.get('ToolCalls-Accuracy', 0):.4f} ({avg_metrics.get('ToolCalls-Accuracy', 0)*100:.2f}%)")
        print(f"  5. Response-Success-Rate-Not-Only-Reasoning: {avg_metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0):.4f} ({avg_metrics.get('Response-Success-Rate-Not-Only-Reasoning', 0)*100:.2f}%)")
        print(f"  6. Language-Following-Success-Rate: {avg_metrics.get('Language-Following-Success-Rate', 0):.4f} ({avg_metrics.get('Language-Following-Success-Rate', 0)*100:.2f}%)")
        
        # Token Usage
        if 'Token-Usage' in avg_metrics:
            print(f"\n  Token 使用统计平均值:")
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
    """发送报告到 API"""
    import requests
    
    if not api_url:
        api_url = os.environ.get(
            'METRICS_REPORT_API_URL',
            'https://swing.xaminim.com/minimax/provider/report/send'
        )
    
    print(f"\n📧 发送报告到: {api_url}")
    
    try:
        # 从环境变量获取配置
        provider_config_str = os.environ.get('PROVIDER_VERIFIER_CONFIG', '')
        provider_config = {}
        if provider_config_str:
            try:
                provider_config = json.loads(provider_config_str)
                print(f"  📦 已加载 PROVIDER_VERIFIER_CONFIG")
            except json.JSONDecodeError:
                pass
        
        # 构建请求数据（与 send_metrics_report.py 保持一致）
        payload = {
            'average_metrics': summary.get('average_metrics', {}),
            **provider_config
        }
        
        # 注意：这里的 key 是 average_metrics，与 send_metrics_report.py 的 aggregated_metrics 对应
        # 两者内容结构相同，都是平均指标
        
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
            print("\n✅ 报告发送成功")
            try:
                result = response.json()
                print(f"  响应: {result}")
            except:
                pass
            return True
        else:
            print(f"\n❌ 报告发送失败: HTTP {response.status_code}")
            print(f"  响应: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ 发送报告失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='与基准数据对比并计算指标'
    )
    
    parser.add_argument(
        '--result-dir',
        type=str,
        required=True,
        help='验证结果目录'
    )
    
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default='MiniMax-M2.5',
        help='基准数据目录 (默认: MiniMax-M2.5)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='Provider 名称（自动检测）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--send-report',
        action='store_true',
        help='发送报告到 API'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='报告 API URL'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 MiniMax Provider Verifier - 基准对比")
    print("=" * 60)
    
    # 检查目录
    if not os.path.exists(args.result_dir):
        print(f"❌ 结果目录不存在: {args.result_dir}")
        return 1
    
    # 解析基准目录（支持相对路径）
    baseline_dir = args.baseline_dir
    if not os.path.isabs(baseline_dir):
        # 尝试多个可能的位置
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
        print(f"❌ 基准目录不存在: {baseline_dir}")
        return 1
    
    print(f"📂 结果目录: {args.result_dir}")
    print(f"📂 基准目录: {baseline_dir}")
    
    # 自动检测 provider
    provider = args.provider
    if not provider:
        summary_files = find_summary_files(args.result_dir)
        if summary_files:
            # 从文件名提取 provider
            basename = os.path.basename(summary_files[0])
            provider = basename.replace('_summary.json', '')
            print(f"📦 自动检测 provider: {provider}")
    
    if not provider:
        print("❌ 无法自动检测 provider，请使用 --provider 参数指定")
        return 1
    
    # 匹配循环
    print("\n🔍 匹配结果与基准...")
    matches = match_loops(args.result_dir, baseline_dir, provider)
    
    if not matches:
        print("❌ 没有找到可匹配的数据")
        return 1
    
    print(f"\n✅ 找到 {len(matches)} 组匹配")
    
    # 执行对比
    print("\n🔄 开始对比...")
    all_results = []
    
    for i, (result_summary, result_jsonl, baseline_summary, baseline_jsonl) in enumerate(matches, 1):
        print(f"\n--- 对比 {i}/{len(matches)} ---")
        print(f"   结果: {result_summary}")
        print(f"   基准: {baseline_summary}")
        
        result = compare_single_loop(
            result_summary, result_jsonl,
            baseline_summary, baseline_jsonl
        )
        all_results.append(result)
    
    # 计算平均指标
    avg_metrics = calculate_average_metrics(all_results)
    
    # 打印结果
    print_comparison_table(all_results, avg_metrics)
    
    # 构建最终报告
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
    
    # 保存报告
    output_file = args.output
    if not output_file:
        output_file = os.path.join(args.result_dir, 'comparison_report.json')
    
    print(f"\n💾 保存报告到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("✅ 报告保存成功")
    
    # 发送报告
    if args.send_report:
        send_report(report, args.api_url)
    
    print("\n🎉 对比完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())