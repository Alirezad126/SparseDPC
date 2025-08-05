from setuptools import setup, find_packages

setup(
    name='SparseDPC',
    version='0.1.0',
    description='Source code for SparseDPC',
    author='Ali Reza Daneshvar Garmroodi',
    author_email='adanesh6@jh.edu',
    url='https://github.com/Alirezad126/SparseDPC',
    python_requires='>=3.7',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'neuromancer',
    ],
)
