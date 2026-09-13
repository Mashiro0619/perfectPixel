import importlib.metadata
import shutil
import subprocess
import sys
import tarfile

import pytest
from PIL import Image

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@pytest.fixture
def project(repository):
    return tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_runtime_and_distribution_versions_match():
    import perfect_pixel
    assert perfect_pixel.__version__ == importlib.metadata.version("perfect-pixel")


def test_readme_filename_matches_case_sensitive_filesystems(repository, project):
    assert project["readme"] in {file.name for file in repository.iterdir()}


def test_debug_extra_declares_plotting_dependency(project):
    assert any(requirement.startswith("matplotlib") for requirement in project["optional-dependencies"]["debug"])


def test_manual_install_directory_is_real(repository):
    path = "integrations/comfyui/PerfectPixelComfy"
    assert (repository / path / "__init__.py").is_file()
    text = (repository / "integrations/comfyui/README.md").read_text(encoding="utf-8")
    assert path in text
    assert "integrations/comfyui/perfectPixel-ComfyUI" not in text


def test_package_imports_without_cv2(tmp_path):
    code = """
import importlib.abc
import sys
class WithoutOpenCV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cv2':
            raise ModuleNotFoundError('No cv2', name='cv2')
sys.meta_path.insert(0, WithoutOpenCV())
import numpy as np
from perfect_pixel import get_perfect_pixel
assert get_perfect_pixel.__module__ == 'perfect_pixel.perfect_pixel_noCV2'
assert get_perfect_pixel(np.zeros((32,32,3), np.uint8), grid_size=(4,4))[:2] == (4,4)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_debug_without_matplotlib_has_installation_guidance(tmp_path):
    code = """
import importlib.abc
import sys
class WithoutMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'matplotlib':
            raise ModuleNotFoundError('No matplotlib', name='matplotlib')
sys.meta_path.insert(0, WithoutMatplotlib())
import numpy as np
from perfect_pixel import get_perfect_pixel
try:
    get_perfect_pixel(np.zeros((32,32,3), np.uint8), grid_size=(4,4), debug=True)
except ImportError as exc:
    assert 'perfect-pixel[debug]' in str(exc)
else:
    raise AssertionError('Expected helpful ImportError')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_example_runs_without_a_gui_from_another_directory(repository, tmp_path):
    pytest.importorskip("cv2")
    target = tmp_path / "result.png"
    result = subprocess.run(
        [sys.executable, "-B", str(repository / "example.py"), "--output", str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    with Image.open(target) as small, Image.open(tmp_path / "result_8x.png") as large:
        assert small.width > 0 and small.height > 0
        assert large.size == (small.width * 8, small.height * 8)



def test_sdist_includes_standalone_test_inputs(repository, tmp_path):
    """Build offline from a clean copy, not from stale egg-info or the checkout."""
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "readme.md", "MANIFEST.in", "example.py"):
        if (repository / name).is_file():
            shutil.copy2(repository / name, source / name)
    for name in ("src", "tests", "images", "integrations"):
        shutil.copytree(
            repository / name, source / name,
            ignore=shutil.ignore_patterns("*.egg-info", "__pycache__", ".pytest_cache"))

    destination = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "build", "--sdist", "--no-isolation",
         "--outdir", str(destination), str(source)],
        cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    archives = list(destination.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        members = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}

    required = {"example.py", "readme.md", "MANIFEST.in", "tests/conftest.py"}
    for directory, patterns in (
        ("tests", ("*.py",)), ("images", ("*.png", "*.jpg", "*.jpeg")),
        ("integrations/comfyui", ("*.py", "*.md")),
    ):
        for pattern in patterns:
            required.update(path.relative_to(repository).as_posix()
                            for path in (repository / directory).rglob(pattern))
    assert required <= members, "Missing sdist test inputs: " + ", ".join(sorted(required - members))
