"""
Setup configuration for LiquidUI Quant Trading Pipeline
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='liquidui-quant',
    version='1.0.0',
    author='Quant Team',
    author_email='quant@example.com',
    description='Production-grade quantitative trading pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourorg/LiquidUI',
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.10.0',
            'flake8>=6.1.0',
            'mypy>=1.6.0',
        ],
        'ml': [
            'mlflow>=2.8.0',
            'optuna>=3.4.0',
        ],
        'advanced': [
            'vectorbt>=0.25.0',
            'QuantLib-Python>=1.31',
        ]
    },
    entry_points={
        'console_scripts': [
            'liquidui=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json'],
    },
    zip_safe=False,
)
