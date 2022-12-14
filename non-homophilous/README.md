### Reproduce the results of ChebNetII in Table 8

#### For medium datasets, you can run the following script:

For Penn94:

```sh
python main.py --dataset fb100 --sub_dataset Penn94 --net ChebNetII --hidden_channels 256 --dprate 0.3 --dropout 0.5 --lr 0.001 --prop_lr 0.01 --prop_wd 0.0005 --weight_decay 0.005
```
For arXiv-year:

```sh
python main.py --dataset arxiv-year --net ChebNetII --hidden_channels 512 --num_layers 3  --dprate 0.6 --dropout 0.1 --lr 0.0005 --prop_lr 0.01 --prop_wd 5e-4 --weight_decay 1e-3 --is_bns True
```
For genius:

```sh
python main.py --dataset genius --net ChebNetII --hidden_channels 128 --K 3 --num_layers 3  --dprate 0.5 --dropout 0.3 --lr 0.005  --prop_lr 0.0005 --prop_wd 0.0001 --weight_decay 0.0 --is_bns True
```
For twitch-gamers:
```sh
python main.py --dataset twitch-gamer --net ChebNetII --hidden_channels 128 --K 5 --num_layers 2 --dprate 0.8 --dropout 0.0 --lr 0.002 --prop_lr 0.002 --prop_wd 0.01  --weight_decay 0.005  --is_bns True
```

#### For large dataset, you need to run the preprocessing code first.

For pokec:
```sh
python processing.py --data_name pokec
```

For wiki:
```sh
python processing.py --data_name wiki
```

#### Second, you can run the following script.

For pokec:
```sh
python large_train.py --dataset pokec --num_layers 4 --lr 0.00005 --dropout 0.5 --hidden 2048 --pro_lr 0.01 --pro_wd 0.00005 --is_bns True
```

For wiki:
```sh
python large_train.py --dataset wiki --num_layers 3 --lr 0.00005 --dropout 0.5 --hidden 2048 --pro_lr 0.01 --pro_wd 0.00005
```

## Acknowledgement

This repository benefits a lot from [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale).
