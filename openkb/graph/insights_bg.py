"""Background insight generation: cooldown-based auto-trigger after `openkb add`."""
from __future__ import annotations

import atexit
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import click

from openkb.config import DEFAULT_CONFIG, load_config
from openkb.graph.build import GraphLoadError, build_and_save_graph, load_graph
from openkb.graph.insights import generate_insights

STATE_FILENAME = "last_insights.json"
REPORT_FILENAME = "insights.md"

_bg_thread: threading.Thread | None = None


def _state_path(kb_dir: Path) -> Path:
    return kb_dir / ".openkb" / STATE_FILENAME


def _report_path(kb_dir: Path) -> Path:
    return kb_dir / ".openkb" / REPORT_FILENAME


def _utc_now_timestamp() -> float:
    return time.time()


def _utc_isoformat(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_cooldown(raw_value: object) -> int:
    default_cooldown = int(DEFAULT_CONFIG["insights_cooldown"])
    if isinstance(raw_value, bool):
        return default_cooldown
    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError, OverflowError):
        return default_cooldown
    if not math.isfinite(parsed_value) or parsed_value < 0:
        return default_cooldown
    return int(parsed_value)


def _get_cooldown_seconds(kb_dir: Path) -> int:
    config = load_config(kb_dir / ".openkb" / "config.yaml")
    raw_value = config.get(
        "insights_cooldown",
        config.get(
            "background_insights_cooldown_seconds",
            DEFAULT_CONFIG["insights_cooldown"],
        ),
    )
    return _normalize_cooldown(raw_value)


def _read_last_run(openkb_dir: Path) -> dict[str, Any]:
    """Read .openkb/last_insights.json; return empty dict if missing."""
    path = openkb_dir / STATE_FILENAME
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def load_background_insights_state(kb_dir: Path) -> dict[str, Any]:
    """Compatibility reader for persisted background-insights state."""
    return _read_last_run(kb_dir / ".openkb")


def inspect_background_insights_state(kb_dir: Path) -> dict[str, Any]:
    """Return an explicit present/missing/unreadable summary for persisted insights state."""
    state_path = _state_path(kb_dir)
    if not state_path.exists():
        return {
            "status": "missing",
            "summary": "Last insights run: missing",
            "last_run": None,
            "report_path": None,
        }

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "status": "unreadable",
            "summary": "Last insights run: unreadable persisted insights state",
            "last_run": None,
            "report_path": None,
        }

    last_run = state.get("last_run")
    report_path = _report_path(kb_dir)
    report_rel = report_path.relative_to(kb_dir).as_posix() if report_path.exists() else None
    if isinstance(last_run, str) and last_run:
        summary = f"Last insights run: {last_run}"
    else:
        summary = "Last insights run: unknown"
    return {
        "status": "ready",
        "summary": summary,
        "last_run": last_run if isinstance(last_run, str) and last_run else None,
        "report_path": report_rel,
    }


def _write_last_run(
    openkb_dir: Path,
    cooldown_seconds: int,
    *,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Write .openkb/last_insights.json with current timestamp."""
    actual_timestamp = _utc_now_timestamp() if timestamp is None else timestamp
    path = openkb_dir / STATE_FILENAME
    state = {
        "last_run": _utc_isoformat(actual_timestamp),
        "cooldown_seconds": cooldown_seconds,
    }
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _last_run_timestamp(state: dict[str, Any]) -> float | None:
    last_run_str = state.get("last_run")
    if not last_run_str:
        return None
    try:
        return datetime.fromisoformat(last_run_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _is_cooldown_expired(
    openkb_dir: Path,
    cooldown_seconds: int,
    *,
    now_timestamp: float | None = None,
) -> bool:
    """Return True if cooldown has expired since last run."""
    state = _read_last_run(openkb_dir)
    last_run_timestamp = _last_run_timestamp(state)
    if last_run_timestamp is None:
        return True
    actual_now = _utc_now_timestamp() if now_timestamp is None else now_timestamp
    return (actual_now - last_run_timestamp) >= cooldown_seconds


def _format_summary(result: dict[str, Any], n_nodes: int, n_edges: int) -> str:
    """Format a 3-5 line terminal summary."""
    n_comm = len(result["communities_summary"])
    n_orphans = len(result["orphans"])
    n_surprising = len(result["surprising_connections"])
    return (
        f"  Communities: {n_comm} | Orphans: {n_orphans} | Surprising: {n_surprising}\n"
        f"  Full report: .openkb/insights.md"
    )


def _write_insights_md(
    openkb_dir: Path,
    result: dict[str, Any],
    n_nodes: int,
    n_edges: int,
    *,
    timestamp: float | None = None,
) -> Path:
    """Write full insights report to .openkb/insights.md."""
    actual_timestamp = _utc_now_timestamp() if timestamp is None else timestamp
    path = openkb_dir / REPORT_FILENAME
    n_comm = len(result["communities_summary"])

    lines = [
        f"# Insights — {datetime.fromtimestamp(actual_timestamp).strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Graph: {n_nodes} nodes, {n_edges} edges, {n_comm} communities",
        "",
    ]

    lines.append("## Surprising Connections")
    for src, tgt, reason, score in result["surprising_connections"][:20]:
        lines.append(f"- {src} <-> {tgt}  [{reason}]  score={score:.2f}")
    if not result["surprising_connections"]:
        lines.append("(none)")

    lines += ["", "## Knowledge Gaps"]
    orphans = result["orphans"]
    if orphans:
        lines.append(f"Orphan nodes ({len(orphans)}):")
        for name, degree in orphans[:20]:
            lines.append(f"  - {name}  (degree={degree})")
    sparse = result["sparse_communities"]
    if sparse:
        lines.append(f"Sparse communities ({len(sparse)}):")
        for community_id, representative, cohesion in sparse[:20]:
            lines.append(
                f"  - community {community_id} (rep: {representative})  cohesion={cohesion:.3f}"
            )
    bridges = result["bridge_nodes"]
    if bridges:
        lines.append(f"Bridge nodes ({len(bridges)}):")
        for name, span in bridges[:20]:
            lines.append(f"  - {name}  (spans {span} communities)")
    if not orphans and not sparse and not bridges:
        lines.append("(none)")

    lines += ["", "## Communities"]
    for community_id, info in sorted(result["communities_summary"].items()):
        lines.append(
            f"- [{community_id}] {info['representative']}  size={info['size']}  cohesion={info['cohesion']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_or_build_graph(
    kb_dir: Path,
    *,
    build_graph_fn: Callable[..., tuple[Any, Path]] | None = None,
    load_graph_fn: Callable[[Path], Any] | None = None,
) -> Any:
    if build_graph_fn is None:
        build_graph_fn = build_and_save_graph
    if load_graph_fn is None:
        load_graph_fn = load_graph

    openkb_dir = kb_dir / ".openkb"
    wiki_dir = kb_dir / "wiki"
    graph_path = openkb_dir / "graph.json"

    if graph_path.exists():
        try:
            return load_graph_fn(graph_path)
        except GraphLoadError:
            graph, _ = build_graph_fn(wiki_dir, openkb_dir)
            return graph

    graph, _ = build_graph_fn(wiki_dir, openkb_dir)
    return graph


def _bg_insights(
    kb_dir: Path,
    *,
    cooldown_seconds: int | None = None,
    now_timestamp: float | None = None,
    build_graph_fn: Callable[..., tuple[Any, Path]] | None = None,
    load_graph_fn: Callable[[Path], Any] | None = None,
    generate_insights_fn: Callable[[Any], dict[str, Any]] | None = None,
    echo_fn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Run insight generation in background thread."""
    if build_graph_fn is None:
        build_graph_fn = build_and_save_graph
    if load_graph_fn is None:
        load_graph_fn = load_graph
    if generate_insights_fn is None:
        generate_insights_fn = generate_insights
    if echo_fn is None:
        echo_fn = click.echo

    openkb_dir = kb_dir / ".openkb"
    actual_timestamp = _utc_now_timestamp() if now_timestamp is None else now_timestamp
    actual_cooldown = cooldown_seconds if cooldown_seconds is not None else _get_cooldown_seconds(kb_dir)

    try:
        graph = _load_or_build_graph(
            kb_dir,
            build_graph_fn=build_graph_fn,
            load_graph_fn=load_graph_fn,
        )
        if graph.number_of_nodes() == 0:
            return None

        result = generate_insights_fn(graph)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        report_path = _write_insights_md(
            openkb_dir,
            result,
            n_nodes,
            n_edges,
            timestamp=actual_timestamp,
        )
        _write_last_run(
            openkb_dir,
            actual_cooldown,
            timestamp=actual_timestamp,
        )

        echo_fn(_format_summary(result, n_nodes, n_edges))
        return {
            "status": "completed",
            "report_path": report_path,
            "state_path": _state_path(kb_dir),
        }
    except Exception:
        return None


def refresh_background_insights(
    kb_dir: Path,
    *,
    cooldown_seconds: int,
    now: float | None = None,
    build_graph_fn: Callable[..., tuple[Any, Path]] | None = None,
    load_graph_fn: Callable[[Path], Any] | None = None,
    generate_insights_fn: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for synchronous insight refresh."""
    result = _bg_insights(
        kb_dir,
        cooldown_seconds=cooldown_seconds,
        now_timestamp=now,
        build_graph_fn=build_graph_fn,
        load_graph_fn=load_graph_fn,
        generate_insights_fn=generate_insights_fn,
        echo_fn=lambda _message: None,
    )
    return result or {
        "status": "failed",
        "report_path": _report_path(kb_dir),
        "state_path": _state_path(kb_dir),
    }


def _wait_for_insights() -> None:
    """atexit handler: wait for background insights thread (max 10s)."""
    if _bg_thread is not None and _bg_thread.is_alive():
        _bg_thread.join(timeout=10)


def maybe_trigger_insights(
    kb_dir: Path,
    *,
    now_timestamp: float | None = None,
    thread_factory: Callable[..., threading.Thread] | None = None,
    register_atexit: Callable[[Callable[[], None]], Any] | None = None,
    echo_fn: Callable[[str], None] | None = None,
) -> str:
    """Check cooldown and spawn background insight generation if expired."""
    global _bg_thread
    if thread_factory is None:
        thread_factory = threading.Thread
    if register_atexit is None:
        register_atexit = atexit.register
    if echo_fn is None:
        echo_fn = click.echo

    actual_now = _utc_now_timestamp() if now_timestamp is None else now_timestamp
    openkb_dir = kb_dir / ".openkb"
    cooldown_seconds = _get_cooldown_seconds(kb_dir)

    if not _is_cooldown_expired(
        openkb_dir,
        cooldown_seconds,
        now_timestamp=actual_now,
    ):
        state = _read_last_run(openkb_dir)
        last_run_timestamp = _last_run_timestamp(state)
        elapsed_minutes = 0
        if last_run_timestamp is not None:
            elapsed_minutes = int((actual_now - last_run_timestamp) // 60)
        echo_fn(f"  (Insights cached {elapsed_minutes}m ago — run `openkb insights` for fresh)")
        return "cached"

    _bg_thread = thread_factory(target=_bg_insights, args=(kb_dir,), daemon=True)
    _bg_thread.start()
    register_atexit(_wait_for_insights)
    echo_fn("  Insights refreshing in background...")
    return "triggered"


def trigger_background_insights(
    kb_dir: Path,
    *,
    cooldown_seconds: int,
    now: float | None = None,
    run_async: bool = True,
    build_graph_fn: Callable[..., tuple[Any, Path]] | None = None,
    load_graph_fn: Callable[[Path], Any] | None = None,
    generate_insights_fn: Callable[[Any], dict[str, Any]] | None = None,
    spawn_process_fn: Callable[..., Any] | None = None,
    thread_factory: Callable[..., threading.Thread] | None = None,
    register_atexit: Callable[[Callable[[], None]], Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper around the plan-aligned background insights flow."""
    if build_graph_fn is None:
        build_graph_fn = build_and_save_graph
    if load_graph_fn is None:
        load_graph_fn = load_graph
    if generate_insights_fn is None:
        generate_insights_fn = generate_insights
    if thread_factory is None:
        thread_factory = threading.Thread
    if register_atexit is None:
        register_atexit = atexit.register

    actual_now = _utc_now_timestamp() if now is None else now
    if not _is_cooldown_expired(
        kb_dir / ".openkb",
        cooldown_seconds,
        now_timestamp=actual_now,
    ):
        return {
            "status": "cached",
            "report_path": _report_path(kb_dir),
            "state_path": _state_path(kb_dir),
        }

    if run_async:
        global _bg_thread

        def _runner(target_kb_dir: Path) -> None:
            _bg_insights(
                target_kb_dir,
                cooldown_seconds=cooldown_seconds,
                now_timestamp=actual_now,
                build_graph_fn=build_graph_fn,
                load_graph_fn=load_graph_fn,
                generate_insights_fn=generate_insights_fn,
            )

        _bg_thread = thread_factory(target=_runner, args=(kb_dir,), daemon=True)
        _bg_thread.start()
        register_atexit(_wait_for_insights)
        return {
            "status": "triggered",
            "report_path": _report_path(kb_dir),
            "state_path": _state_path(kb_dir),
        }

    result = _bg_insights(
        kb_dir,
        cooldown_seconds=cooldown_seconds,
        now_timestamp=actual_now,
        build_graph_fn=build_graph_fn,
        load_graph_fn=load_graph_fn,
        generate_insights_fn=generate_insights_fn,
        echo_fn=lambda _message: None,
    )
    return {
        "status": "triggered",
        "report_path": (result or {}).get("report_path", _report_path(kb_dir)),
        "state_path": (result or {}).get("state_path", _state_path(kb_dir)),
    }
