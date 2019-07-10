from setuptools import setup, find_packages
from distutils.command.sdist import sdist
import shutil
import subprocess
import sys
import os

DEBUG_BUILD = os.environ.get("DEBUG") == "1"
EXT = sys.platform == "darwin" and ".dylib" or ".so"


def vendor_rust_deps():
    try:
        subprocess.run("cargo --version".split(" "))
    except (FileNotFoundError, subprocess.CalledProcessError):
        os.system("curl https://sh.rustup.rs -sSf | sh -s -- -y")
        os.system("source $HOME/.cargo/env")
        os.system(". $HOME/.cargo/env")
        os.system("export PATH=$HOME/.cargo/bin:$PATH")
        print(
            "Installing Rust might have failed. Make sure you have Rust installed, and in your PATH.",
            file=sys.stderr,
        )
        print("Check with 'cargo --version' and 'rustc --version'", file=sys.stderr)
        # ps = subprocess.Popen(
        #     "curl https://sh.rustup.rs -sSf".split(" "),
        #     stdout=subprocess.PIPE)
        # output = subprocess.check_output("sh -y".split(" "), stdin=ps.stdout, shell=True)
        # ps.wait()

        # subprocess.run('curl https://sh.rustup.rs -sSf | sh'.split(' '))


class CustomSDist(sdist):
    def run(self):
        vendor_rust_deps()
        sdist.run(self)


def add_dll_module():
    pass
    shutil.copy2(dylib, os.path.join(base_path, self.lib_filename))


def build_native(spec):
    vendor_rust_deps()
    cmd = "cargo build".split(" ")
    if not DEBUG_BUILD:
        cmd.append("--release")
        target = "release"
    else:
        target = "debug"

    rust_path = "smb"

    print("running %s (%s target)" % (" ".join(cmd), target))
    build = spec.add_external_build(cmd=cmd, path=rust_path)

    rtld_flags = ["NOW"]
    if sys.platform == "darwin":
        rtld_flags.append("NODELETE")
    # spec.add_cffi_module(
    #         module_path='smb._hicrs',
    #         dylib=lambda: build.find_dylib('hicrs', in_path='target/release'),
    #         header_filename='',
    #         rtld_flags=rtld_flags
    #         )


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="smb",
    version="0.1.2",
    description="Fast stochastic matrix balancing in Rust",
    url="http://github.com/fkarg/HiC-rs",
    author="Felix Karg",
    author_email="kargf@informatik.uni-freiburg.de",
    license="GPLv3",
    packages=find_packages(),
    package_dir={"smb": "smb"},
    include_package_data=True,
    package_data={
        "smb": ["target/release/libhicrs" + EXT, "target/debug/libhicrs" + EXT]
    },
    zip_safe=False,
    #       platforms='any',
    install_requires=requirements,
    setup_requires=["milksnake>=0.1.2"],
    milksnake_tasks=[build_native],
    cmdclass={"sdist": CustomSDist},
)
