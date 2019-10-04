import setuptools
 
setuptools.setup(
  name="scitsr",
  version="0.0.1",
  author="Zewen Chi",
  author_email="czw@bit.edu.cn",
  description="code for paper: complicated table structure recognition",
  url="https://github.com/Academic-Hammer/SciTSR",
  packages=setuptools.find_packages(),
  install_requires=[
    #"ujson",
    #"tqdm",
    #"torch",
    #"torchtext"
  ],
  classifiers=(
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  )
)