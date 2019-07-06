# HiC-rs
[![Build Status](https://travis-ci.com/fkarg/HiC-rs.svg?branch=master)](https://travis-ci.com/fkarg/HiC-rs)

This part is the python-rust interface for the iterative matrix correction

## Install

First: Make sure you have Python and Conda installed properly.

To install `smb` (short for stochastic matrix balancing) on a Ubuntu 18.04, simply execute `conda install -c kargf smb`.

**For using, not building, installation of Rust is not needed.**

## Build

To build it, execute the commands applicable to you:

```
# first, install rust:
curl https://sh.rustup.rs -sSf | sh -s -- -y

# alternatively install rust with conda:
conda install -c conda-forge rust

# confirm install:
cargo --version
rustc --version

# download repository and navigate in it
git clone https://github.com/fkarg/HiC-rs
cd HiC-rs

# navigate to the rust code and compile (optional)
cd smb
cargo build
cd ..

# install missing python dependencies
pip install -r requirements.txt

# execute the setup.py (will also compile rust if not done yet)
python setup.py build
pip install .
```

