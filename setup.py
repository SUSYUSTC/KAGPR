from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Self-descriptive entries which should always be present
    name='GPR',
    author='Jiace Sun',
    author_email='jsun3@caltech.edu',
    description="Kernel-Addition Gaussian Process Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SUSYUSTC/KAGPR",
    license='Open Source',

    # What packages are required for install
    install_requires=[],
    extras_require={
        'tests': [
            'unittest',
        ],
    },
    packages=["GPR",
              "GPR.kern",
              "GPR.regression"],
)
