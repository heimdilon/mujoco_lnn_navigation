from pathlib import Path


def test_no_isaac_imports():
    root = Path("source")
    haystack = "\n".join(path.read_text(encoding="utf-8").lower() for path in root.rglob("*.py"))
    assert "isaac" not in haystack
    assert "kit.exe" not in haystack

