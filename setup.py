from setuptools import setup
from skdiveMove import __license__, __version__


def readme():
    with open('README.rst') as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().splitlines()

    return(reqs)


REQUIREMENTS = get_requirements()
PACKAGES = ["skdiveMove"]

setup(
    name="scikit-diveMove",
    version=__version__,
    python_requires=">=3.6",
    packages=PACKAGES,
    # include_package_data=True,
    # package_data={'tests': ["*.txt", "*.csv", "*.BIN"]},
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": ["ipython", "jupyter", "jupyter-sphinx"],
        "docs": ["jupyter-sphinx"]
      },
    # metadata for upload to PyPI
    author="Sebastian Luque",
    author_email="spluque@gmail.com",
    description="Python interface to R package diveMove",
    long_description=readme(),
    license=__license__,
    download_url="https://github.com/spluque/scikit-diveMove",
    keywords=["animal behaviour", "biology", "behavioural ecology",
              "diving", "diving behaviour"],
    url="https://https://github.com/spluque/scikit-diveMove",
    classifiers=["Development Status :: 2 - Pre-Alpha",
                 "Programming Language :: Python :: 3",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 ("License :: OSI Approved :: "
                  "GNU Affero General Public License v3"),
                 "Topic :: Scientific/Engineering"]
)
