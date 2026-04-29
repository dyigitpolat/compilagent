from __future__ import annotations

import json
from pathlib import Path

from compilagent.observation.artifacts import build_default_registry


def test_json_renderer_pretty_prints(tmp_path: Path):
    registry = build_default_registry()
    p = tmp_path / "result.json"
    p.write_text(json.dumps({"speedup": 1.42, "ok": True}), encoding="utf-8")
    preview = registry.render(p)
    assert preview.kind == "json"
    assert preview.language == "json"
    assert '"speedup"' in preview.text
    assert "1.42" in preview.text


def test_markdown_renderer(tmp_path: Path):
    registry = build_default_registry()
    p = tmp_path / "report.md"
    p.write_text("# heading\n\nbody text\n", encoding="utf-8")
    preview = registry.render(p)
    assert preview.kind == "markdown"
    assert preview.language == "markdown"
    assert "heading" in preview.text


def test_python_renderer(tmp_path: Path):
    registry = build_default_registry()
    p = tmp_path / "kernel.py"
    p.write_text("def foo():\n    return 42\n", encoding="utf-8")
    preview = registry.render(p)
    assert preview.kind == "code"
    assert preview.language == "python"


def test_text_renderer(tmp_path: Path):
    registry = build_default_registry()
    for suffix in (".txt", ".log"):
        p = tmp_path / f"sample{suffix}"
        p.write_text("hello world\n", encoding="utf-8")
        preview = registry.render(p)
        assert preview.kind == "text"
        assert "hello world" in preview.text


def test_unknown_suffix_falls_back_to_binary(tmp_path: Path):
    registry = build_default_registry()
    p = tmp_path / "obj.bin"
    p.write_bytes(b"\x00\x01\x02")
    preview = registry.render(p)
    assert preview.kind == "binary"
    assert "no renderer" in preview.text


def test_truncation(tmp_path: Path):
    registry = build_default_registry()
    p = tmp_path / "big.txt"
    p.write_text("a" * 100, encoding="utf-8")
    preview = registry.render(p, max_chars=20)
    assert "truncated" in preview.text
