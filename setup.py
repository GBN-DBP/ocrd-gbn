from io import open
from setuptools import find_packages, setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name="gbn",
    version="0.0.1",
    author="Lucas Sulzbach",
    author_email="lucas@sulzbach.org",
    description="Collection of OCR-D compliant tools for layout analysis and segmentation of historical documents from the German-Brazilian Newspapers dataset",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='gbn',
    license='Apache',
    url="https://dokumente.ufpr.br",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=install_requires,
    package_data={
        '': ['*.json'],
    },
    entry_points={
      'console_scripts': [
        "ocrd-gbn-sbb-predict=gbn.sbb:ocrd_gbn_sbb_predict",
        "ocrd-gbn-sbb-page-segment=gbn.sbb:ocrd_gbn_sbb_page_segment",
        "ocrd-gbn-sbb-region-segment=gbn.sbb:ocrd_gbn_sbb_region_segment",
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
