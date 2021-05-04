import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='transformer_deid',
    version='0.1.0',
    description='Deidentification with transformers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alistairewj/transformer-deid',
    author='Alistair Johnson',
    author_email='aewj@mit.edu',
    packages=setuptools.find_packages(),
    python_requires='>=3.7'
)
