import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="HaPi",
    version="0.0.1",
    author="Jorge Garcia Condado",
    author_email="jorgeschool@gmail.com",
    description="A pipeline to determine the hand of cryoEM determined maps.",
    long_description=long_description,
    url="https://github.com/JGarciaCondado/EMapHandedness",
    packages=["hapi"],
    python_requires="==3.8.5",
    install_requires=[requirements]
)
