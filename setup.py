from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="llm_entropy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
    author="Matthew Finch",
    author_email="finchrmatthew@gmail.com",
    description="Analysis of LLM internal behavior through information-theoretic metrics",
    url="https://github.com/FinchMF/llm_entropy",
    python_requires=">=3.7",
)
