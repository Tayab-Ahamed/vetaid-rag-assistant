"""Exercise file persistence without executing Streamlit's top-level UI/models."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def save_references(files):
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_uploaded_files")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["save_uploaded_files"](files)


@pytest.mark.parametrize("name", ["../escaped.txt", "..\\escaped.txt", "folder/file.pdf", "folder\\file.pdf", "C:escaped.txt", "", ".."])
def test_rejects_reference_filenames_with_directory_components(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    upload = SimpleNamespace(name=name, getvalue=lambda: b"untrusted content")
    with pytest.raises(ValueError, match="filename"):
        save_references([upload])
    assert not (tmp_path / "escaped.txt").exists()


def test_accepts_plain_reference_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = save_references([SimpleNamespace(name="reference.txt", getvalue=lambda: b"reference content")])
    assert result == [str(Path("uploads") / "reference.txt")]
    assert Path(result[0]).read_bytes() == b"reference content"


def test_validates_all_names_before_persisting_a_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = [SimpleNamespace(name=name, getvalue=lambda: b"content") for name in ["valid.txt", "../escaped.txt"]]
    with pytest.raises(ValueError, match="filename"):
        save_references(files)
    assert not Path("uploads/valid.txt").exists()
