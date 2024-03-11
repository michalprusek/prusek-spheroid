from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="prusek_spheroid",
    version="2.6",
    description="Spheroid segmentation package",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Michal Prusek",
    author_email="prusemic@cvut.cz",
    url="https://github.com/michalprusek/Spheroid-segmentation",
    packages=["prusek_spheroid"],
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-image',
        'scikit-learn',
        'shapely',
        'threadpoolctl',
        'matplotlib',
        'rasterio',
        'pandas',
        'openpyxl',
        'torch'
    ],
)