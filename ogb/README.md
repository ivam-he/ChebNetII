### Reproduce the results of ChebNetII in Table 9

#### First, you need to run the preprocessing code.

For ogbn-arxiv:

```sh
python processing.py --data_name arxiv
```
For ogbn-papers100M:

```sh
python processing.py --data_name papers100m
```

#### Second, you can run the following script.

For ogbn-arxiv:

```sh
python large_train.py --data arxiv --lr 0.0005 --dropout 0.5 --hidden 512 --pro_lr 0.01 --pro_wd 0.0
```

For ogbn-papers100M:

```sh
python large_train.py --data papers100m --lr 0.001 --dropout 0.5 --hidden 2048 --pro_lr 0.01 --pro_wd 0.00005
```

