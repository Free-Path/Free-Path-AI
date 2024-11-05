from setuptools import setup, find_packages

setup(
    name="recommend_server",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'scikit-surprise>=1.1.1',
        'python-dotenv>=0.19.0',
        'pydantic>=2.0.0',
        'pydantic-settings>=2.0.0',
    ],
)