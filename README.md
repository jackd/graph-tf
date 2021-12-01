# [graph-tf](https://github.com/jackd/graph-tf)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Tensorflow implementation of graph networks

## Installation

```bash
git clone https://github.com/jackd/graph-tf.git
pip install -e graph-tf
```

Benchmarks require [tfbm](https://github.com/jackd/tfbm)

```bash
git clone https://github.com/jackd/tfbm.git
pip install -e tfbm
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
