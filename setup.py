#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Ler o conteúdo do README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Ler as dependências do requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="vector-tpb",
    version="0.1.0",
    author="Vitor Collos",
    author_email="vcollos@gmail.com",
    description="Banco de Dados Vetorial sobre Transtorno de Personalidade Borderline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vcollos/tpb",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vector-tpb=vector_tpb.main:main",
            "vector-tpb-search=vector_tpb.search_example:main",
            "vector-tpb-qa=vector_tpb.langchain_qa:main",
        ],
    },
    include_package_data=True,
)