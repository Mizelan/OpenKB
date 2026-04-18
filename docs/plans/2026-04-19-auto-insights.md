# Auto Insights Implementation Plan

> **For Claude Code:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `openkb add` 완료 후 쿨다운 기반으로 백그라운드에서 인사이트를 자동 생성한다.

**Architecture:** `add` 명령 완료 시 `maybe_trigger_insights(kb_dir)` 호출 → `.openkb/last_insights.json` 쿨다운 체크 → 만료 시 `threading.Thread(daemon=True)`로 백그라운드 인사이트 생성 → 결과를 `.openkb/insights.md`에 저장 + 터미널 요약 출력. `atexit` 핸들러로 프로세스 종료 시 thread 완료 대기(최대 10초).

**Tech Stack:** Python threading, atexit, networkx, 기존 `openkb.graph.insights.generate_insights`

---

### Task 1: config에 `insights_cooldown` 키 추가

**Files:**
- Modify: `openkb/config.py:8-14`

**Step 1: DEFAULT_CONFIG에 키 추가**

```python
DEFAULT_CONFIG: dict[str, Any] = {
    "model": "sonnet",
    "provider": "claude",
    "effort": "medium",
    "language": "en",
    "pageindex_threshold": 20,
    "insights_cooldown": 3600,
}
```

**Step 2: 기존 테스트로 회귀 없음 확인**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add openkb/config.py
git commit -m "feat: add insights_cooldown to DEFAULT_CONFIG (3600s)"
```

---

### Task 2: `insights_bg.py` — 백그라운드 인사이트 생성 로직

**Files:**
- Create: `openkb/graph/insights_bg.py`
- Test: `tests/test_insights_bg.py`

**Step 1: `tests/test_insights_bg.py` 작성 — 실패하는 테스트 먼저**

```python
"""Tests for openkb.graph.insights_bg — background insight generation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openkb.graph.insights_bg import maybe_trigger_insights, _bg_insights


class TestMaybeTriggerInsights:
    def test_first_run_triggers_insights(self, tmp_path):
        """쿨다운 파일이 없으면 인사이트를 생성해야 한다."""
        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)
        (kb_dir / "wiki").mkdir(parents=True)

        with patch("openkb.graph.insights_bg._bg_insights") as mock_bg:
            maybe_trigger_insights(kb_dir)
            mock_bg.assert_called_once_with(kb_dir)

    def test_cooldown_not_expired_skips(self, tmp_path):
        """쿨다운 미만료 시 인사이트를 건너뛰어야 한다."""
        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)
        (kb_dir / "wiki").mkdir(parents=True)

        state = {"last_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cooldown_seconds": 3600}
        (openkb_dir / "last_insights.json").write_text(json.dumps(state), encoding="utf-8")

        with patch("openkb.graph.insights_bg._bg_insights") as mock_bg:
            result = maybe_trigger_insights(kb_dir)
            mock_bg.assert_not_called()
            assert result == "cached"

    def test_cooldown_expired_triggers(self, tmp_path):
        """쿨다운 만료 시 인사이트를 생성해야 한다."""
        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)
        (kb_dir / "wiki").mkdir(parents=True)

        state = {"last_run": "2020-01-01T00:00:00Z", "cooldown_seconds": 3600}
        (openkb_dir / "last_insights.json").write_text(json.dumps(state), encoding="utf-8")

        with patch("openkb.graph.insights_bg._bg_insights") as mock_bg:
            result = maybe_trigger_insights(kb_dir)
            mock_bg.assert_called_once_with(kb_dir)
            assert result == "triggered"


class TestBgInsights:
    def test_generates_insights_md(self, tmp_path):
        """_bg_insights가 insights.md를 생성해야 한다."""
        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)
        wiki_dir = kb_dir / "wiki"
        wiki_dir.mkdir(parents=True)

        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 5
        mock_graph.number_of_edges.return_value = 3

        fake_result = {
            "orphans": [("doc1", 0)],
            "sparse_communities": [],
            "bridge_nodes": [],
            "surprising_connections": [("a", "b", "cross-community", 0.8)],
            "communities_summary": {0: {"size": 3, "cohesion": 0.5, "representative": "doc2"}},
        }

        with patch("openkb.graph.insights_bg.build_and_save_graph", return_value=(mock_graph, openkb_dir / "graph.json")), \
             patch("openkb.graph.insights_bg.generate_insights", return_value=fake_result):
            _bg_insights(kb_dir)

        insights_path = openkb_dir / "insights.md"
        assert insights_path.exists()
        content = insights_path.read_text(encoding="utf-8")
        assert "Communities" in content or "communities" in content.lower()

    def test_updates_last_insights_json(self, tmp_path):
        """_bg_insights가 last_insights.json을 갱신해야 한다."""
        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)
        wiki_dir = kb_dir / "wiki"
        wiki_dir.mkdir(parents=True)

        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 2
        mock_graph.number_of_edges.return_value = 1

        fake_result = {
            "orphans": [], "sparse_communities": [], "bridge_nodes": [],
            "surprising_connections": [], "communities_summary": {},
        }

        with patch("openkb.graph.insights_bg.build_and_save_graph", return_value=(mock_graph, openkb_dir / "graph.json")), \
             patch("openkb.graph.insights_bg.generate_insights", return_value=fake_result):
            _bg_insights(kb_dir)

        state_path = openkb_dir / "last_insights.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "last_run" in state
        assert "cooldown_seconds" in state

    def test_graph_load_error_handled(self, tmp_path):
        """GraphLoadError 발생 시 예외 없이 종료해야 한다."""
        from openkb.graph.build import GraphLoadError

        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)

        with patch("openkb.graph.insights_bg.build_and_save_graph", side_effect=GraphLoadError("corrupted")):
            _bg_insights(kb_dir)  # should not raise
```

**Step 2: 테스트 실행 — 실패 확인**

Run: `python3 -m pytest tests/test_insights_bg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'openkb.graph.insights_bg'`

**Step 3: `openkb/graph/insights_bg.py` 구현**

```python
"""Background insight generation: cooldown-based auto-trigger after `openkb add`."""
from __future__ import annotations

import atexit
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from openkb.config import DEFAULT_CONFIG, load_config
from openkb.graph.build import build_and_save_graph, GraphLoadError
from openkb.graph.insights import generate_insights

_bg_thread: threading.Thread | None = None


def _read_last_run(openkb_dir: Path) -> dict:
    """Read .openkb/last_insights.json; return empty dict if missing."""
    path = openkb_dir / "last_insights.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_last_run(openkb_dir: Path, cooldown_seconds: int) -> None:
    """Write .openkb/last_insights.json with current timestamp."""
    path = openkb_dir / "last_insights.json"
    state = {
        "last_run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cooldown_seconds": cooldown_seconds,
    }
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _is_cooldown_expired(openkb_dir: Path, cooldown_seconds: int) -> bool:
    """Return True if cooldown has expired since last run."""
    state = _read_last_run(openkb_dir)
    last_run_str = state.get("last_run")
    if not last_run_str:
        return True
    try:
        last_run = datetime.fromisoformat(last_run_str.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
        return elapsed >= cooldown_seconds
    except (ValueError, TypeError):
        return True


def _format_summary(result: dict, n_nodes: int, n_edges: int) -> str:
    """Format a 3-5 line terminal summary."""
    n_comm = len(result["communities_summary"])
    n_orphans = len(result["orphans"])
    n_surprising = len(result["surprising_connections"])
    return (
        f"  Communities: {n_comm} | Orphans: {n_orphans} | Surprising: {n_surprising}\n"
        f"  Full report: .openkb/insights.md"
    )


def _write_insights_md(openkb_dir: Path, result: dict, n_nodes: int, n_edges: int) -> None:
    """Write full insights report to .openkb/insights.md."""
    path = openkb_dir / "insights.md"
    n_comm = len(result["communities_summary"])

    lines = [
        f"# Insights — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
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
        for name, deg in orphans[:20]:
            lines.append(f"  - {name}  (degree={deg})")
    sparse = result["sparse_communities"]
    if sparse:
        lines.append(f"Sparse communities ({len(sparse)}):")
        for cid, rep, coh in sparse[:20]:
            lines.append(f"  - community {cid} (rep: {rep})  cohesion={coh:.3f}")
    bridges = result["bridge_nodes"]
    if bridges:
        lines.append(f"Bridge nodes ({len(bridges)}):")
        for name, span in bridges[:20]:
            lines.append(f"  - {name}  (spans {span} communities)")
    if not orphans and not sparse and not bridges:
        lines.append("(none)")

    lines += ["", "## Communities"]
    for cid, info in sorted(result["communities_summary"].items()):
        lines.append(f"- [{cid}] {info['representative']}  size={info['size']}  cohesion={info['cohesion']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bg_insights(kb_dir: Path) -> None:
    """Run insight generation in background thread."""
    openkb_dir = kb_dir / ".openkb"
    wiki_dir = kb_dir / "wiki"

    try:
        config = load_config(openkb_dir / "config.yaml")
        cooldown = config.get("insights_cooldown", DEFAULT_CONFIG["insights_cooldown"])

        if (openkb_dir / "graph.json").exists():
            from openkb.graph.build import load_graph
            try:
                graph = load_graph(openkb_dir / "graph.json")
            except GraphLoadError:
                graph, _ = build_and_save_graph(wiki_dir, openkb_dir)
        else:
            graph, _ = build_and_save_graph(wiki_dir, openkb_dir)

        if graph.number_of_nodes() == 0:
            return

        result = generate_insights(graph)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        _write_insights_md(openkb_dir, result, n_nodes, n_edges)
        _write_last_run(openkb_dir, cooldown)

        summary = _format_summary(result, n_nodes, n_edges)
        click.echo(summary)

    except Exception:
        pass


def _wait_for_insights() -> None:
    """atexit handler: wait for background insights thread (max 10s)."""
    if _bg_thread is not None and _bg_thread.is_alive():
        _bg_thread.join(timeout=10)


def maybe_trigger_insights(kb_dir: Path) -> str:
    """Check cooldown and spawn background insight generation if expired.

    Returns "triggered" if thread was spawned, "cached" if cooldown not expired.
    """
    global _bg_thread

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    cooldown = config.get("insights_cooldown", DEFAULT_CONFIG["insights_cooldown"])

    if not _is_cooldown_expired(openkb_dir, cooldown):
        state = _read_last_run(openkb_dir)
        last_run_str = state.get("last_run", "")
        elapsed = 0
        try:
            last_run = datetime.fromisoformat(last_run_str.replace("Z", "+00:00"))
            elapsed = int((datetime.now(timezone.utc) - last_run).total_seconds() // 60)
        except (ValueError, TypeError):
            pass
        click.echo(f"  (Insights cached {elapsed}m ago — run `openkb insights` for fresh)")
        return "cached"

    _bg_thread = threading.Thread(target=_bg_insights, args=(kb_dir,), daemon=True)
    _bg_thread.start()
    atexit.register(_wait_for_insights)
    click.echo("  Insights refreshing in background...")
    return "triggered"
```

**Step 4: 테스트 실행 — 통과 확인**

Run: `python3 -m pytest tests/test_insights_bg.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/graph/insights_bg.py tests/test_insights_bg.py
git commit -m "feat: add insights_bg — background insight generation with cooldown"
```

---

### Task 3: `cli.py` add 명령에 `maybe_trigger_insights` 연결

**Files:**
- Modify: `openkb/cli.py:214-216`
- Test: `tests/test_cli.py`

**Step 1: 테스트 추가**

```python
# tests/test_cli.py 에 추가

def test_add_triggers_insights_on_cooldown(tmp_path):
    """add 완료 후 쿨다운 만료 시 maybe_trigger_insights가 호출되어야 한다."""
    runner = CliRunner()
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / ".openkb").mkdir()
    (kb_dir / "raw").mkdir()
    (kb_dir / "wiki" / "sources").mkdir(parents=True)

    # minimal source file
    src = kb_dir / "raw" / "test.md"
    src.write_text("# Test\n\nHello world.", encoding="utf-8")

    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file"), \
         patch("openkb.graph.insights_bg.maybe_trigger_insights") as mock_trigger:
        mock_trigger.return_value = "triggered"
        result = runner.invoke(cli, ["add", str(src)])
        mock_trigger.assert_called_once_with(kb_dir)


def test_add_skips_insights_when_cached(tmp_path):
    """add 완료 후 쿨다운 미만료 시 cached 반환."""
    runner = CliRunner()
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / ".openkb").mkdir()
    (kb_dir / "raw").mkdir()
    (kb_dir / "wiki" / "sources").mkdir(parents=True)

    src = kb_dir / "raw" / "test.md"
    src.write_text("# Test\n\nHello world.", encoding="utf-8")

    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file"), \
         patch("openkb.graph.insights_bg.maybe_trigger_insights") as mock_trigger:
        mock_trigger.return_value = "cached"
        result = runner.invoke(cli, ["add", str(src)])
        assert "cached" in result.output.lower() or "Insights" in result.output
```

**Step 2: 테스트 실행 — 실패 확인**

Run: `python3 -m pytest tests/test_cli.py::test_add_triggers_insights_on_cooldown tests/test_cli.py::test_add_skips_insights_when_cached -v`
Expected: FAIL — `maybe_trigger_insights` not called from `add` command

**Step 3: `cli.py` add 명령에 트리거 추가**

`add_single_file` 함수 끝(`append_log` 이후, `click.echo(f"  [OK]...")` 이후)에 `maybe_trigger_insights` 호출 추가:

```python
# openkb/cli.py — add_single_file 함수 끝에 추가
    append_log(kb_dir / "wiki", "ingest", file_path.name)
    click.echo(f"  [OK] {file_path.name} added to knowledge base.")

    # Auto-trigger background insights (cooldown-based)
    from openkb.graph.insights_bg import maybe_trigger_insights
    maybe_trigger_insights(kb_dir)
```

**Step 4: 테스트 실행 — 통과 확인**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/cli.py tests/test_cli.py
git commit -m "feat: trigger background insights after openkb add"
```

---

### Task 4: `openkb config` CLI 명령 추가

**Files:**
- Modify: `openkb/cli.py`
- Test: `tests/test_cli.py`

**Step 1: config 명령 테스트 추가**

```python
# tests/test_cli.py 에 추가

def test_config_set_insights_cooldown(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        # init first
        runner.invoke(cli, ["init"])
        # set cooldown
        result = runner.invoke(cli, ["config", "set", "insights_cooldown", "60"])
        assert result.exit_code == 0
        # verify
        from pathlib import Path
        config = yaml.safe_load(Path(".openkb/config.yaml").read_text())
        assert config["insights_cooldown"] == 60


def test_config_get_insights_cooldown(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        runner.invoke(cli, ["init"])
        result = runner.invoke(cli, ["config", "get", "insights_cooldown"])
        assert result.exit_code == 0
        assert "3600" in result.output  # default value
```

**Step 2: 테스트 실행 — 실패 확인**

Run: `python3 -m pytest tests/test_cli.py::test_config_set_insights_cooldown tests/test_cli.py::test_config_get_insights_cooldown -v`
Expected: FAIL — `No such command: config`

**Step 3: config 명령 구현**

`cli.py`에 `config` 그룹과 `set`, `get` 서브명령 추가:

```python
import yaml as _yaml  # 이미 import 되어 있으면 생략

@cli.group()
@click.pass_context
def config(ctx):
    """Manage OpenKB configuration."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a config value: openkb config set KEY VALUE"""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")

    # Type coercion
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass

    config[key] = value
    save_config(openkb_dir / "config.yaml", config)
    click.echo(f"Set {key} = {value}")


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx, key):
    """Get a config value: openkb config get KEY"""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    value = config.get(key)
    if value is None:
        click.echo(f"{key}: (not set)")
    else:
        click.echo(f"{key}: {value}")
```

**Step 4: 테스트 실행 — 통과 확인**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: 전체 스위트 회귀 확인**

Run: `python3 -m pytest tests/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add openkb/cli.py tests/test_cli.py
git commit -m "feat: add openkb config set/get CLI command"
```

---

### Task 5: 통합 검증

**Step 1: 전체 테스트 스위트 실행**

Run: `python3 -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: 실제 KB에서 동작 확인 (수동)**

```bash
cd <existing-kb-dir>
openkb add raw/some-doc.md
# 쿨다운 만료 시:
#   Insights refreshing in background...
#   Communities: 5 | Orphans: 2 | Surprising: 1
#   Full report: .openkb/insights.md

# 쿨다운 미만료 시:
#   (Insights cached 5m ago — run `openkb insights` for fresh)

openkb config set insights_cooldown 60
# 1분 후 다시 add → 인사이트 재생성
```

**Step 3: `.openkb/insights.md` 파일 확인**

insights.md가 정상적으로 생성되었는지 확인.

**Step 4: 프로세스 종료 시 thread 정상 완료 확인**

add 후 즉시 프로세스 종료 시 insights.md가 손상 없이 저장되었는지 확인(atexit 핸들러 동작).