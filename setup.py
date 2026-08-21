from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements():
    """Return non-comment entries from the repository requirements file."""
    return [
        line
        for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]

setup(
    name='SparseDPC',
    version='0.1.0',
    description='Source code for SparseDPC',
    author='Ali Reza Daneshvar Garmroodi',
    author_email='adanesh6@jh.edu',
    url='https://github.com/Alirezad126/SparseDPC',
    python_requires='>=3.10',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
)
