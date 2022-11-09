import setuptools

with open("README.md") as fh:
    long_desc = fh.read()

setuptools.setup(
    name="cosmoSR",
    version="0.2.dev0",
    author_email="ivkos@mpa-garching.mpg.de",
    description="Train a network to super resolve cosmological simulations",
    authors=["Ivan Kostyuk"],
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="GPL-3.0",
    packages=["cosmoSR"],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "numpy",
    ],
)
