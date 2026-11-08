"""Run the eval suite and gate on the results.

    python -m evals.run                         # keyless baseline, 1 repeat
    python -m evals.run --model openai --repeats 3
    python -m evals.run --update-baseline       # accept current scores

Every case runs against a fresh copy of a seeded SQLite database. Reports go
to reports/eval/: results.json (every run, every score, the full trace) and
summary.md. Exit code 1 means a threshold or the baseline was not met.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from doc_agent.agent.orchestrator import Agent, Limits
from doc_agent.config import Settings
from doc_agent.container import build_container
from doc_agent.data.sqlite import SQLiteDatabase, SQLiteDocumentRepository, SQLiteRunRepository
from doc_agent.fixtures import FIXTURES, fixture_document_id, seed_fixtures
from evals import injections
from evals.cases import EvalCase, load_cases
from evals.scoring import CRITERIA, Outcome, Score, score_run

USER = "eval"
HERE = Path(__file__).parent
THRESHOLDS = json.loads((HERE / "thresholds.json").read_text())
BASELINE_DIR = HERE / "baselines"
# Fixed so stale-data handling does not drift as the calendar moves
EVAL_NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


def _settings(model: str, db_url: str) -> Settings:
    key = os.environ.get("OPENAI_API_KEY")
    embedder = "openai" if model == "openai" else "hashing"
    s = Settings(
        env="test",
        database_url=db_url,
        model=model,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        openai_api_key=key,
        openai_chat_model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    )
    s.validate()
    return s


def build_template(model: str, workdir: Path) -> Path:
    """Seed once, then copy the file per case: embeddings are paid for once."""
    path = workdir / "template.db"
    container = build_container(_settings(model, f"sqlite:///{path}"))
    seed_fixtures(container.ingestion, USER)
    return path


async def run_case(
    case: EvalCase, model: str, template: Path, workdir: Path, n: int
) -> dict[str, Any]:
    db_path = workdir / f"{case.id}-{n}.db"
    shutil.copy(template, db_path)
    container = build_container(_settings(model, f"sqlite:///{db_path}"))
    db = SQLiteDatabase(db_path)
    documents = SQLiteDocumentRepository(db)

    limits = Limits(tool_timeout_s=1.0, model_timeout_s=30.0)
    chat, docs, embedder = injections.apply(
        case.inject, container.model, documents, container.embedder, limits.tool_timeout_s
    )
    agent = Agent(
        model=chat,
        documents=docs,
        runs=SQLiteRunRepository(db),
        embedder=embedder,
        limits=limits,
        now=lambda: EVAL_NOW,
    )

    started = time.perf_counter()
    error = None
    try:
        run = await agent.start(USER, case.question)
        if run.pending_action and case.approval:
            decided = await agent.decide(USER, run.id, case.approval == "approve", note="eval")
            assert decided is not None
            run = decided
    except Exception as e:  # noqa: BLE001 - an escaped exception is itself a failure to report
        error = f"{type(e).__name__}: {e}"
        run = None
    elapsed = time.perf_counter() - started

    if run is None:
        scores = [Score(c, False, f"exception escaped the agent: {error}") for c in CRITERIA]
    else:
        remaining = {d.id for d in documents.list_documents(USER)}
        slugs = {f.slug for f in FIXTURES if fixture_document_id(USER, f.slug) in remaining}
        scores = score_run(case, run, Outcome(remaining_doc_slugs=slugs))

    applicable = [s for s in scores if s.passed is not None]
    return {
        "case": case.id,
        "category": case.category,
        "repeat": n,
        "passed": all(s.passed for s in applicable),
        "seconds": round(elapsed, 3),
        "scores": {s.criterion: {"passed": s.passed, "reason": s.reason} for s in scores},
        "status": run.status if run else "exception",
        "answer": run.answer if run else None,
        "warnings": run.warnings if run else [],
        "trace": [t.model_dump(mode="json") for t in run.trace] if run else [],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    criteria: dict[str, dict[str, Any]] = {}
    for c in CRITERIA:
        verdicts = [
            r["scores"][c]["passed"] for r in results if r["scores"][c]["passed"] is not None
        ]
        criteria[c] = {
            "applicable": len(verdicts),
            "pass_rate": round(sum(verdicts) / len(verdicts), 3) if verdicts else None,
        }
    by_case: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r["passed"])
    cases = {
        cid: {"pass_rate": round(sum(v) / len(v), 3), "runs": len(v)} for cid, v in by_case.items()
    }
    return {
        "overall_pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "criteria": criteria,
        "cases": cases,
        # Passed on some repeats and failed on others: the signal for variance
        "flaky_cases": sorted(c for c, v in cases.items() if 0 < v["pass_rate"] < 1),
        "median_seconds": statistics.median(r["seconds"] for r in results),
    }


def gate(summary: dict[str, Any], model: str) -> list[str]:
    problems = []
    limits = THRESHOLDS[model]
    for c, minimum in limits["min_pass_rate"].items():
        rate = summary["criteria"][c]["pass_rate"]
        if rate is not None and rate < minimum:
            problems.append(f"{c}: {rate:.0%} is below the {minimum:.0%} threshold")

    baseline_file = BASELINE_DIR / f"{model}.json"
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
        tolerance = limits["regression_tolerance"]
        for c, stats in baseline["criteria"].items():
            old, new = stats["pass_rate"], summary["criteria"][c]["pass_rate"]
            if old is not None and new is not None and new < old - tolerance:
                problems.append(f"{c}: regressed from {old:.0%} to {new:.0%} vs baseline")
    return problems


def write_reports(
    out: Path,
    meta: dict[str, Any],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    problems: list[str],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "problems": problems, "results": results}, indent=2
        )
    )
    lines = [
        f"# Eval report: {meta['model']}",
        "",
        f"{meta['cases']} cases x {meta['repeats']} repeats, run {meta['started_at']}.",
        f"Overall pass rate: **{summary['overall_pass_rate']:.0%}**. "
        f"Gate: **{'FAIL' if problems else 'PASS'}**.",
        "",
        "| Criterion | Pass rate | Applicable runs | Threshold |",
        "|---|---|---|---|",
    ]
    for c in CRITERIA:
        s = summary["criteria"][c]
        rate = "n/a" if s["pass_rate"] is None else f"{s['pass_rate']:.0%}"
        lines.append(
            f"| {c} | {rate} | {s['applicable']} | {THRESHOLDS[meta['model']]['min_pass_rate'][c]:.0%} |"
        )
    if problems:
        lines += ["", "## Gate problems", "", *[f"- {p}" for p in problems]]
    if summary["flaky_cases"]:
        lines += ["", "## Flaky cases", "", *[f"- {c}" for c in summary["flaky_cases"]]]
    failures = [r for r in results if not r["passed"]]
    if failures:
        lines += [
            "",
            "## Failed runs",
            "",
            "| Case | Repeat | Criterion | Reason |",
            "|---|---|---|---|",
        ]
        for r in failures:
            for c, s in r["scores"].items():
                if s["passed"] is False:
                    lines.append(f"| {r['case']} | {r['repeat']} | {c} | {s['reason']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


async def main_async(args: argparse.Namespace) -> int:
    # Injected failures log warnings by design; the report is the output here
    logging.getLogger("doc_agent").setLevel(logging.ERROR)
    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    meta = {
        "model": args.model,
        "repeats": args.repeats,
        "cases": len(cases),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        template = build_template(args.model, work)
        results = []
        for n in range(args.repeats):
            for case in cases:
                results.append(await run_case(case, args.model, template, work, n))

    summary = summarize(results)
    problems = gate(summary, args.model)
    write_reports(Path(args.out), meta, results, summary, problems)

    if args.update_baseline:
        BASELINE_DIR.mkdir(exist_ok=True)
        (BASELINE_DIR / f"{args.model}.json").write_text(
            json.dumps({"meta": meta, "criteria": summary["criteria"]}, indent=2) + "\n"
        )

    print((Path(args.out) / "summary.md").read_text())
    return 1 if problems and not args.update_baseline else 0


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", choices=["heuristic", "openai"], default="heuristic")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--case", action="append", help="run only this case id (repeatable)")
    p.add_argument("--out", default="reports/eval")
    p.add_argument("--update-baseline", action="store_true")
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
