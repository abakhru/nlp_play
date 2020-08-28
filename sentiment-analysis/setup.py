#!/usr/bin/env python

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages

setup(author="Amit Bakhru",
      author_email='bakhru@me.com',
      python_requires='>=3.5',
      description="Sentiment Analysis related experiments",
      install_requires=Path('requirements.txt').read_text().rsplit(),
      license="MIT license",
      include_package_data=True,
      keywords='nlp_play',
      name='sentiment_analysis',
      packages=find_packages(),
      test_suite='tests',
      zip_safe=False,
      )
