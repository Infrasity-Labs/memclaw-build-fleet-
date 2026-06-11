"""
Pipeline Orchestrator — runs all 5 agents in sequence via MCP tool-use.

Each agent:
  1. Discovers MemClaw tools from the MCP server (tools/list)
    2. Sends its prompt to the configured LLM with those tools attached
    3. The model calls memclaw_recall / memclaw_write / memclaw_insights as needed
    4. The loop runs until the model stops issuing tool calls

Execution order:
  1. Frontend Agent    — write only  (nothing to recall, first in chain)
  2. Performance Agent — recall → write
  3. SEO Agent         — recall → write  (informed by Performance constraints)
  4. Code Review Agent — recall → insights → write
  5. Manager Tenant    — list + stats + insights  (read-only, no write)

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-manager
  python run_pipeline.py --dry-run
  python run_pipeline.py --json-output results.json
  python run_pipeline.py --log-level DEBUG
"""

import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding so UTF-8 chars print correctly
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load .env BEFORE any module that reads env vars at import time
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

import agent_frontend
import agent_performance
import agent_seo
import agent_codereview
import manager as agent_manager
import mcp_client as mcp
from config import AgentID


def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


log = logging.getLogger(__name__)

PIPELINE_STEPS = [
    ("Frontend Agent",     agent_frontend,    "recall:—  write:HTML5/CSS decisions"),
    ("Performance Agent",  agent_performance, "recall:frontend → write:CWV rules"),
    ("SEO Agent",          agent_seo,         "recall:all → write:SEO decisions"),
    ("Code Review Agent",  agent_codereview,  "recall:all + insights → write:verdict"),
    ("Manager Tenant",     agent_manager,     "list+stats+insights  (read-only audit)"),
]


def check_env() -> list[str]:
    return [k for k in ("LLM_GATEWAY_API_KEY", "MEMCLAW_API_KEY", "MEMCLAW_TENANT_ID")
            if not os.environ.get(k)]


def _mask(value: str, visible: int = 4) -> str:
    """Show only the last `visible` chars of a sensitive value."""
    if not value or value == "(not set)":
        return "(not set)"
    return f"{'*' * max(0, len(value) - visible)}{value[-visible:]}"


def print_banner():
    tenant_raw = os.environ.get("MEMCLAW_TENANT_ID", "(not set)")
    print("\n" + "=" * 65)
    print("  MemClaw 5-Fleet SaaS Build Pipeline  (MCP tool-use)")
    print("  Recall Before Acting · Write After Deciding")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"  MCP URL : {mcp.MCP_URL}")
    print(f"  Transport: {os.environ.get('MEMCLAW_TRANSPORT', 'mcp')}")
    print(f"  Fleet   : {os.environ.get('MEMCLAW_FLEET_ID', 'fleet')}")
    print(f"  Tenant  : {_mask(tenant_raw)}")   # masked — never log raw tenant/key values
    print(f"  Model   : {os.environ.get('LLM_GATEWAY_MODEL', 'llama-3.3-70b-versatile')}")
    print("=" * 65 + "\n")


def print_pipeline_table(steps):
    print("Execution plan:")
    print(f"  {'#':<3} {'Agent':<22} {'MCP Tool Usage'}")
    print(f"  {'-'*3} {'-'*22} {'-'*38}")
    for i, (name, _, desc) in enumerate(steps, 1):
        print(f"  {i:<3} {name:<22} {desc}")
    print()


def run_pipeline(steps) -> dict:
    results = {}
    for i, (name, module, _) in enumerate(steps, 1):
        log.info("[%d/%d] Starting %s", i, len(steps), name)
        t0 = time.time()
        try:
            data = module.run()
            elapsed = time.time() - t0
            results[name] = {"status": "ok", "elapsed_s": round(elapsed, 2), "data": data}
            calls = len(data.get("tool_calls", []))
            iters = data.get("iterations", 0)
            log.info("[%d/%d] ✓ %s  %.1fs  %d tool call(s)  %d iteration(s)",
                     i, len(steps), name, elapsed, calls, iters)
        except Exception as exc:
            elapsed = time.time() - t0
            results[name] = {"status": "error", "elapsed_s": round(elapsed, 2), "error": str(exc)}
            log.error("[%d/%d] ✗ %s  FAILED: %s", i, len(steps), name, exc, exc_info=True)
    return results


def _parse_verdict(text: str, keywords: list[str], fallback: str = "?") -> str:
    """Case-insensitive scan for first matching keyword in text."""
    upper = text.upper()
    for kw in keywords:
        if kw.upper() in upper:
            return kw
    return fallback


def print_summary(results: dict):
    print("\n" + "=" * 65)
    print("  PIPELINE SUMMARY")
    print("=" * 65)

    for name, r in results.items():
        icon = "✓" if r["status"] == "ok" else "✗"
        t = r["elapsed_s"]
        if r["status"] == "ok":
            data = r["data"]
            by_tool: dict[str, int] = {}
            for c in data.get("tool_calls", []):
                by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
            tool_str = "  ".join(f"{k}×{v}" for k, v in by_tool.items()) or "—"
            print(f"  {icon} {name:<22} {t:>5.1f}s  [{tool_str}]")
        else:
            print(f"  {icon} {name:<22} {t:>5.1f}s  ERROR: {r['error'][:40]}")

    # Code review verdict
    cr = results.get("Code Review Agent", {})
    if cr.get("status") == "ok":
        text = cr["data"].get("final_text", "")
        verdict = _parse_verdict(text, ["LGTM", "BLOCK"])
        icon = "✅" if verdict == "LGTM" else ("🚫" if verdict == "BLOCK" else "❓")
        print(f"\n  Code Review Verdict : {icon} {verdict}")

    # Manager health — look for structured keywords the prompt instructs the manager to emit
    mgr = results.get("Manager Tenant", {})
    if mgr.get("status") == "ok":
        text = mgr["data"].get("final_text", "")
        health = _parse_verdict(text, ["HEALTHY", "WARNINGS", "CRITICAL"])
        icon = "✅" if health == "HEALTHY" else ("⚠️" if health == "WARNINGS" else ("🚫" if health == "CRITICAL" else "❓"))
        print(f"  Pipeline Health     : {icon} {health}")
        if "zero write" in text.lower() or "no write" in text.lower():
            print("  Data Isolation      : ✅ VERIFIED")

    print(f"\n  View memories at: {mcp.MEMCLAW_BASE_URL.rstrip('/')}/prism")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MemClaw 5-Fleet SaaS Build Pipeline")
    parser.add_argument("--dry-run",      action="store_true", help="Check env + MCP connectivity, then exit")
    parser.add_argument("--skip-manager", action="store_true", help="Skip Manager Tenant audit")
    parser.add_argument("--json-output",  metavar="FILE",      help="Write full results to JSON file")
    parser.add_argument("--log-level",    default="INFO",       help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    print_banner()

    missing = check_env()
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        print("Copy .env.example → .env and fill in your keys.")
        sys.exit(1)

    steps = PIPELINE_STEPS[:-1] if args.skip_manager else PIPELINE_STEPS

    if args.dry_run:
        log.info("Checking MemClaw API connectivity...")
        try:
            ok = mcp.health_check()
            if ok:
                tools = mcp.list_tools()
                log.info("MemClaw API reachable — %d tools defined: %s",
                         len(tools), [t["name"] for t in tools])
            else:
                log.error("MemClaw API returned an error — check MEMCLAW_API_KEY and MEMCLAW_TENANT_ID")
                sys.exit(1)
        except Exception as exc:
            log.error("MemClaw connection failed: %s", exc, exc_info=True)
            sys.exit(1)
        log.info("Dry run complete — env OK, MemClaw OK.")
        print_pipeline_table(steps)
        sys.exit(0)

    print_pipeline_table(steps)
    results = run_pipeline(steps)
    print_summary(results)

    if args.json_output:
        safe = json.loads(json.dumps(results, default=str))
        Path(args.json_output).write_text(json.dumps(safe, indent=2), encoding="utf-8")
        log.info("Full results written to %s", args.json_output)


if __name__ == "__main__":
    main()
