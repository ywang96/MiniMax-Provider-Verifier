import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import megfile
from loguru import logger
from verify import ValidatorRunner


# Models that need a large output budget. See full rationale below — pass@10 on
# Together fp4 with the default max_tokens (~2048) silently truncated 49/53
# failures and dragged ToolCalls-Match-Rate down by ~4.4pp. Re-running with
# max_tokens=40960 lifted all metrics to passing. Only M3 is known to exhibit
# this footgun today; add other models here as new evidence arrives.
M3_MODEL_REGEX = re.compile(r"(?<![A-Za-z0-9])m3(?![A-Za-z0-9])", re.IGNORECASE)
M3_DEFAULT_MAX_TOKENS = 40960


def _maybe_force_m3_max_tokens(model: str, extra_body: dict) -> dict:
    """If `model` looks like an M3 deployment and the caller didn't already set a
    large-enough max_tokens, force it to M3_DEFAULT_MAX_TOKENS. Matches the
    'm3' token bounded by non-alphanumerics so 'gm3'/'pm3'/'rm3' do NOT trigger,
    but 'MiniMax-M3', 'minimax-m3-0603-fp4', 'MiniMaxAI/MiniMax-M3-NVFP4' all do.

    Returns the (possibly mutated) extra_body. The previous value, if any, is
    logged so caller intent stays auditable.
    """
    if not M3_MODEL_REGEX.search(model or ""):
        return extra_body
    if extra_body is None:
        extra_body = {}
    prev_max = extra_body.get("max_tokens")
    extra_body["max_tokens"] = M3_DEFAULT_MAX_TOKENS
    if prev_max != M3_DEFAULT_MAX_TOKENS:
        logger.info(
            f"🔒 [M3 detected: {model}] Forced max_tokens={M3_DEFAULT_MAX_TOKENS} "
            f"(previous={prev_max!r}) to avoid length truncation"
        )
    return extra_body


async def run_provider_verification(
    provider_config: dict,
    file_path: str,
    concurrency: int,
    timeout: int,
    max_retries: int,
    output_dir: str,
    incremental: bool,
    stream: bool = False,
    debug: bool = False,
    openrouter_provider: str = None,
    api_format: str = "openai",
    extra_body: dict = None,
) -> dict:
    """Run verification for a single provider"""
    provider_name = provider_config["name"]
    logger.info(f"Starting verification for provider: {provider_name}")
    
    # Create separate output files for each provider
    output_file = os.path.join(output_dir, f"{provider_name}_results.jsonl")
    summary_file = os.path.join(output_dir, f"{provider_name}_summary.json")
    
    try:
        # Merge extra_body: command line arguments take precedence over provider_config
        merged_extra_body = provider_config.get("extra_body", {})
        if extra_body:
            merged_extra_body.update(extra_body)

        # M3 deployments need a generous max_tokens — see _maybe_force_m3_max_tokens
        merged_extra_body = _maybe_force_m3_max_tokens(
            provider_config.get("model", ""), merged_extra_body
        )

        if merged_extra_body:
            logger.info(f"📦 Using extra_body for provider {provider_name}: {merged_extra_body}")

        default_headers = provider_config.get("default_headers") or {}
        if default_headers:
            logger.info(f"📦 Using default_headers for provider {provider_name}: {list(default_headers.keys())}")

        runner = ValidatorRunner(
            model=provider_config["model"],
            base_url=provider_config["base_url"],
            api_key=provider_config["api_key"],
            concurrency=concurrency,
            output_file=output_file,
            summary_file=summary_file,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=merged_extra_body,
            incremental=incremental,
            stream=stream,
            debug=debug,
            openrouter_provider=openrouter_provider,
            api_format=api_format,
            default_headers=default_headers,
        )
        
        await runner.validate_file(file_path)
        
        return {
            "provider": provider_name,
            "status": "success",
            "summary": runner.summary,
            "output_file": output_file,
            "summary_file": summary_file,
            "completed_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Provider {provider_name} verification failed: {e}")
        return {
            "provider": provider_name,
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        }


async def batch_verify_providers(
    provider_file: str,
    test_file: str,
    concurrency: int,
    timeout: int,
    max_retries: int,
    output_dir: str,
    incremental: bool,
    parallel_providers: int = 1,
    stream: bool = False,
    debug: bool = False,
    openrouter_provider: str = None,
    api_format: str = "openai",
    extra_body: dict = None,
):
    """Batch verify all providers"""
    # Read provider configuration
    with megfile.smart_open(provider_file, "r", encoding="utf-8") as f:
        providers = json.load(f)
    
    logger.info(f"Loaded {len(providers)} provider configurations")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # If parallel_providers is 1, execute serially
    if parallel_providers == 1:
        results = []
        for provider in providers:
            result = await run_provider_verification(
                provider,
                test_file,
                concurrency,
                timeout,
                max_retries,
                output_dir,
                incremental,
                stream,
                debug,
                openrouter_provider,
                api_format,
                extra_body,
            )
            results.append(result)
    else:
        # Execute multiple providers in parallel
        semaphore = asyncio.Semaphore(parallel_providers)
        
        async def run_with_semaphore(provider):
            async with semaphore:
                return await run_provider_verification(
                    provider,
                    test_file,
                    concurrency,
                    timeout,
                    max_retries,
                    output_dir,
                    incremental,
                    stream,
                    debug,
                    openrouter_provider,
                    api_format,
                    extra_body,
                )
        
        tasks = [run_with_semaphore(provider) for provider in providers]
        results = await asyncio.gather(*tasks)
    
    # Generate overall report
    total_report = {
        "test_file": test_file,
        "total_providers": len(providers),
        "successful_providers": sum(1 for r in results if r["status"] == "success"),
        "failed_providers": sum(1 for r in results if r["status"] == "failed"),
        "provider_results": results,
        "generated_at": datetime.now().isoformat(),
    }
    
    # Save overall report
    report_file = os.path.join(output_dir, "batch_report.json")
    with megfile.smart_open(report_file, "w", encoding="utf-8") as f:
        json.dump(total_report, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 80)
    logger.info("Batch verification completed!")
    logger.info(f"Total providers: {total_report['total_providers']}")
    logger.info(f"Successful: {total_report['successful_providers']}")
    logger.info(f"Failed: {total_report['failed_providers']}")
    logger.info(f"Overall report saved to: {report_file}")
    logger.info("=" * 80)
    
    # Print summary for each provider
    for result in results:
        provider = result["provider"]
        status = result["status"]
        logger.info(f"\nProvider: {provider}")
        logger.info(f"  Status: {status}")
        
        if status == "success" and "summary" in result:
            summary = result["summary"]
            logger.info(f"  Success count: {summary.get('success_count', 0)}")
            logger.info(f"  Failure count: {summary.get('failure_count', 0)}")
            
            # Tool Calls Validator related statistics
            if "tool_calls_successful_count" in summary:
                logger.info(f"  Tool Calls successful: {summary.get('tool_calls_successful_count', 0)}")
                logger.info(f"  Tool Calls schema validation errors: {summary.get('tool_calls_schema_validation_error_count', 0)}")
                logger.info(f"  Tool Calls total count: {summary.get('tool_calls_total_count', 0)}")
            
        elif status == "failed":
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")


async def main():
    parser = argparse.ArgumentParser(
        description="Batch verify capabilities of multiple LLM providers\n\n"
        "Reads provider configurations from provider.json and runs verification tests for each provider.\n"
        "Supports multiple validators to simultaneously verify tool calls, content length, and other dimensions."
    )
    
    parser.add_argument(
        "test_file",
        help="Test data file path (JSONL format), e.g.: sample.jsonl",
    )
    
    parser.add_argument(
        "--provider-file",
        default="provider.json",
        help="Provider configuration file path (default: provider.json)",
    )
    
    parser.add_argument(
        "--output-dir",
        default="batch_results",
        help="Output directory (default: batch_results)",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        dest="max_workers",
        help="Maximum concurrent requests per provider (default: 10)",
    )
    
    parser.add_argument(
        "--parallel-providers",
        type=int,
        default=1,
        help="Number of providers to run simultaneously (default: 1, meaning serial execution)",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per request in seconds (default: 600)",
    )
    
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries on failure (default: 3)",
    )
    
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: only rerun failed requests",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode for API requests (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only run first 10 test cases (default: False)",
    )
    parser.add_argument(
        "--openrouter-provider",
        type=str,
        default=None,
        help="OpenRouter provider filter, e.g., 'fireworks'. Adds provider.only field to API requests.",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="API format to use: 'openai' (default) or 'anthropic'. Use 'anthropic' for Claude models via Anthropic API.",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="Extra body parameters as JSON string, e.g., '{\"safety\": {\"input_level\": \"none\"}}'",
    )
    
    args = parser.parse_args()
    
    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
            logger.info(f"Using extra_body: {extra_body}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for --extra-body: {e}")
            return
    
    await batch_verify_providers(
        provider_file=args.provider_file,
        test_file=args.test_file,
        concurrency=args.max_workers,
        timeout=args.timeout,
        max_retries=args.retries,
        output_dir=args.output_dir,
        incremental=args.incremental,
        parallel_providers=args.parallel_providers,
        stream=args.stream,
        debug=args.debug,
        openrouter_provider=args.openrouter_provider,
        api_format=args.api_format,
        extra_body=extra_body,
    )


if __name__ == "__main__":
    asyncio.run(main())

