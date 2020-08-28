from io import open
from setuptools import find_packages, setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name="ocrd-gbn",
    version="1.0.0",
    author="Lucas Sulzbach",
    author_email="lucas@sulzbach.org",
    description="Collection of OCR-D compliant tools for layout analysis and segmentation of historical german-language documents published in Brazil",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords=['OCR', 'OCR-D'],
    license='Apache',
    url="https://github.com/sulzbals/gbn",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=install_requires,
    package_data={
        '': ['*.json'],
    },
    entry_points={
      'console_scripts': [
        "ocrd-gbn-sbb-predict=gbn:ocrd_gbn_sbb_predict",
        "ocrd-gbn-sbb-crop=gbn:ocrd_gbn_sbb_crop",
        "ocrd-gbn-sbb-binarize=gbn:ocrd_gbn_sbb_binarize",
        "ocrd-gbn-sbb-segment=gbn:ocrd_gbn_sbb_segment",
      ]
    },
    python_requires='>=3.6.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
