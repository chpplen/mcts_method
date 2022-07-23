import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MCTS",
    version="0.0.1",
    author="Chinwei.Chang",
    author_email="chinwei.chang@deltaww.com",
    description="mcts package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://10.136.154.2:9900/analytics_optimization/mcts_method",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)