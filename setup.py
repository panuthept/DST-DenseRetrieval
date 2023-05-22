from setuptools import setup, find_packages

setup(
    name='tevatron_dst',
    version='1.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache 2.0',
    author='Panuthep Tasawong',
    author_email='panuthep.t_s20@vistec.ac.th',
    description='Typo-Robust Representation Learning for Dense Retrieval',
    python_requires='>=3.7',
    install_requires=[
        "torch",
        "transformers",
        "textattack",
        "datasets",
        "faiss",
        "numpy"
    ]
)
