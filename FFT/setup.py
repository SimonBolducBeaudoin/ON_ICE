import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fft",
    version="1.0",
    author="Alexandre Dumont",
    author_email="Alexandre.Dumont3@usherbrooke.ca",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pybind11'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=2.7'
)
