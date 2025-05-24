from setuptools import setup, find_packages

setup(
    name="llm_entropy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy<2.0",
        "matplotlib",
        "pandas",
        "pyyaml"
    ],
    author="Your Name",
    author_email="finchrmatthew@gmail.com",
    description="Analysis of LLM internal behavior through information-theoretic metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FinchMF/llm_entropy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
