## Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited

This repository contains a PyTorch implementation of our NeruIPS2022 paper "Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited."


## Environment Settings    
- pytorch 1.11.0
- numpy 1.23.4
- torch-geometric 1.7.2
- tqdm 4.64.1
- scipy 1.9.3
- seaborn 0.12.0
- scikit-learn 1.1.3
- ogb 1.3.5

### Installation

```
pip install -r requirements.txt
```

## Datasets
We provide the small datasets in the folder '/main/data'. The ogb datasets (ogbn-arxiv and ogbn-papers100M) and non-homophilous datasets (from [LINKX](https://arxiv.org/abs/2110.14446) ) can be downloaded automatically.


## Code Structure
The folder "main" is the code for the main results from Tables 1-3 and 6-7 in the paper; The folder "non-homophilous" is the code for the results from Table 8; The folder "ogb" is the code for the results from Table 9.

## Citation
```sh
@inproceedings{he2022chebnetii,
  title={Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited},
  author={He, Mingguo and Wei, Zhewei and Wen, Ji-Rong},
  booktitle={NeurIPS},
  year={2022}
}
```

## Contact

If you have any questions, please feel free to contact me with mingguo@ruc.edu.cn
