from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_is_dev_only():
    assert not (ROOT / "scripts/eval/transfer.py").exists()
    assert (ROOT / "scripts/dev/transfer.py").exists()
    assert (ROOT / "scripts/dev/parity/transfer_cli.py").exists()


def test_no_stale_eval_transfer_references():
    stale = []
    paths = [
        *ROOT.glob("scripts/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
        ROOT / "README.md",
        ROOT / "docs/parity.md",
    ]
    for path in paths:
        if path == Path(__file__).resolve():
            continue
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text()
        if "scripts/eval/transfer.py" in text or "scripts.eval.transfer" in text:
            stale.append(str(path.relative_to(ROOT)))
    assert stale == []
