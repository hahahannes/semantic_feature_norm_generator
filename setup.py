from setuptools import setup, find_packages
setup(
    name='semantic_norm_generator',
    version='1.0',
    author='Hannes Hansen',
    author_email='',
    packages=find_packages(),
    install_requires=[
        "transformers==4.31.0",
        "openai==0.27.8",
        "spacy==3.6.0",
        "mlconjug==3.4.0",
        "anthropic==0.3.6",
        "inflect",
        "mlconjug3",
        "accelerate",
        "bitsandbytes==0.41.0"
    ],
)