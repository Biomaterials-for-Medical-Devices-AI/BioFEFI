# BioFEFI: Python Toolkit for Machine Learning, Feature Importance, and Fuzzy Interpretation

![License][license-badge]
![Python][python-badge]
![Poetry][poetry-badge]
![scikit-learn][sklearn-badge]
![Matplotlib][plt-badge]
![Linux][linux-badge]
![macOS][macos-badge]
![Windows][windows-badge]

![GitHub Issues or Pull Requests][issues-badge]
![Build docs status][build-docs-badge]
![Publish docs status][publish-docs-badge]
![Code quality status][code-quality-badge]
![PyPI downloads][downloads-badge]

## Overview

BioFEFI is an open-source, extensible tool for reproducible Machine Learning Modelling and results interpretation. It was originally designed for QSAR/QSPR modelling in biomaterials discovery, but can be applied to any tabular data classification or regression tasks. Version 1.0.0 contains tools for data visualisation and basic pre-processing, it has a collection of machine learning models and interpretation approaches, including fuzzy fusion. The theoretical work underpinning the development of the tool can be found in:

D. Rengasamy, Jimiama M. Mase, Aayush Kumar, Benjamin Rothwell, Mercedes Torres Torres, Morgan R. Alexander, David A. Winkler, Grazziela P. Figueredo,
Feature importance in machine learning models: A fuzzy information fusion approach,
Neurocomputing, Volume 511,2022, Pages 163-174,ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2022.09.053 [LINK](https://www.sciencedirect.com/science/article/pii/S0925231222011584)

D. Rengasamy, B. C. Rothwell; G. P. Figueredo, Towards a More Reliable Interpretation of Machine Learning Outputs for Safety-Critical Systems Using Feature Importance Fusion. Appl. Sci. 2021, 11, 11854. https://doi.org/10.3390/app112411854 [Link](https://www.mdpi.com/2076-3417/11/24/11854)

To cite BioFEFI, please use the following DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14721954.svg)](https://doi.org/10.5281/zenodo.14721954)


## Install and run BioFEFI

You will need to install **Python 3.11** or **3.12** to use BioFEFI. Make sure you also install `pip` (The Python package installer). If you don't already have it installed, [get Python.](https://www.python.org/downloads/)

You may need to make sure you have OpenMP installed on your machine before you can install BioFEFI. In the terminal use the following commands for your OS:

On Mac:
```shell
brew install libomp
```

You may need to try `brew3` if `brew` does not work. Make sure you [install Homebrew](https://brew.sh/) on your Mac to use the `brew`/`brew3` command.

On Linux (Ubuntu)
```shell
sudo apt install libomp-dev
```

On Windows, this doesn't seem to be a problem. You should be able to proceed with installation.

For information on how to install and run BioFEFI, check the [instructions](https://biomaterials-for-medical-devices-ai.github.io/BioFEFI/users/installation.html).

## Usage

BioFEFI will open in your internet browser when you run it. The main screen will appear giving a brief introduction to the app. To the left of the screen you will see a list of pages with the different functionalities of the app. Explanations can be found in the [instructions](https://biomaterials-for-medical-devices-ai.github.io/BioFEFI/index.html).


## Team
- [Daniel Lea](https://github.com/dcl10) (Lead Research Software Engineer)
- [Eduardo Aguilar](https://edaguilarb.github.io./) (Chemist, Data Scientist, Research Software Engineer)
- Karthikeyan Sivakumar (Data Scientist, Software Engineer)
- Jimiama M Mase (Data Scientist and Engineer)
- Reza Omidvar (Data Scientist, Research Software Engineer)
- James Mitchel-White (Data Scientist, Research Software Engineer)
- [Grazziela Figueredo](https://scholar.google.com/citations?user=DXNNUcUAAAAJ&hl=en) (Associate Professor, Principal Investigator)

[poetry-badge]: https://img.shields.io/badge/Poetry-%233B82F6.svg?style=for-the-badge&logo=poetry&logoColor=0B3D8D
[sklearn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[plt-badge]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[linux-badge]: https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black
[macos-badge]: https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0
[windows-badge]: https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white
[python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[issues-badge]: https://img.shields.io/github/issues/Biomaterials-for-Medical-Devices-AI/BioFEFI?style=for-the-badge
[build-docs-badge]: https://img.shields.io/github/actions/workflow/status/Biomaterials-for-Medical-Devices-AI/BioFEFI/build-dcos.yml?style=for-the-badge&label=Build%20docs
[publish-docs-badge]: https://img.shields.io/github/actions/workflow/status/Biomaterials-for-Medical-Devices-AI/BioFEFI/publish-docs.yml?style=for-the-badge&label=Publish%20docs
[code-quality-badge]: https://img.shields.io/github/actions/workflow/status/Biomaterials-for-Medical-Devices-AI/BioFEFI/format-code.yml?style=for-the-badge&label=Code%20quality
[license-badge]: https://img.shields.io/github/license/Biomaterials-for-Medical-Devices-AI/BioFEFI?style=for-the-badge&label=License
[downloads-badge]: https://img.shields.io/pypi/dm/biofefi?style=for-the-badge

## Contact

For bugs, questions, suggestions and collaborations, please [contact us](mailto:g.figueredo@gmail.com)
