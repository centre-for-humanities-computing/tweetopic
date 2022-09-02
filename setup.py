from distutils.core import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="Tweetopic",
    version="0.0.1",
    description="Efficient Numba based implementation of Gibbs Sampling Dirichlet Mixed Model.",
    author="Márton Kardos",
    author_email="power.up1163@gmail.com",
    packages=["tweetopic"],
    license="MIT",
    install_requires=required,
)
