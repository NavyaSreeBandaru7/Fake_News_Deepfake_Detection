from setuptools import setup, find_packages
from pathlib import Path
import re

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from __init__.py
def get_version():
    version_file = this_directory / "src" / "fake_news_detector" / "__init__.py"
    if version_file.exists():
        version_text = version_file.read_text()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_text, re.M)
        if version_match:
            return version_match.group(1)
    return "2.0.0"

# Read requirements
def get_requirements():
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            requirements = [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#') and not line.startswith('http')
            ]
        return requirements
    return []

setup(
    name="fake-news-deepfake-detector",
    version=get_version(),
    author="Professional AI Engineer",
    author_email="ai.engineer@example.com",
    description="Advanced AI-powered system for detecting fake news and deepfakes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fake-news-deepfake-detector",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/fake-news-deepfake-detector/issues",
        "Documentation": "https://github.com/yourusername/fake-news-deepfake-detector/wiki",
        "Source Code": "https://github.com/yourusername/fake-news-deepfake-detector",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
        ],
        "web": [
            "streamlit>=1.25.0",
            "gradio>=3.35.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "nvidia-ml-py>=12.535.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "streamlit>=1.25.0",
            "gradio>=3.35.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fake-news-detector=fake_news_detector.cli:main",
            "deepfake-detector=fake_news_detector.cli:deepfake_main",
            "fake-news-web=fake_news_detector.web:streamlit_main",
            "fake-news-gradio=fake_news_detector.web:gradio_main",
        ],
    },
    include_package_data=True,
    package_data={
        "fake_news_detector": [
            "models/*.json",
            "models/*.pth",
            "data/*.csv",
            "config/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "fake news detection",
        "deepfake detection", 
        "misinformation",
        "machine learning",
        "natural language processing",
        "computer vision",
        "artificial intelligence",
        "transformers",
        "BERT",
        "RoBERTa",
        "CNN",
        "deep learning",
    ],
)
