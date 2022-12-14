## Reproduce the results in Tables 1-3 and 6-7

### In Table 1 for ChebNet and GCN
You can run the following commands directly.

ChebNet(K=2)
```sh
python training.py --net ChebNet --K 2 --hidden 32 --dataset Cora
python training.py --net ChebNet --K 2 --hidden 32 --dataset Citeseer
python training.py --net ChebNet --K 2 --hidden 32 --dataset Pubmed --dropout 0.8
```
ChebNet(K=10)
```sh
python training.py --net ChebNet --K 10 --hidden 16 --dataset Cora
python training.py --net ChebNet --K 10 --hidden 16 --dataset Citeseer
python training.py --net ChebNet --K 10 --hidden 16 --dataset Pubmed --dropout 0.8
```
GCN
```sh
python training.py --net GCN --dataset Cora
python training.py --net GCN --dataset Citeseer
python training.py --net GCN --dataset Pubmed 
```

### In Table 2 and Table 3
You can run the following commands directly. 

ChebBase
```sh
python training.py --net ChebBase --K 10 --dataset Cora --dprate 0.5 --q 0
python training.py --net ChebBase --K 10 --dataset Citeseer --dprate 0.5 --q 0
python training.py --net ChebBase --K 10 --dataset Pubmed --dprate 0.8 --q 0
```

ChebBase/k
```sh
python training.py --net ChebBase --K 10 --dataset Cora --dprate 0.5 --q 1
python training.py --net ChebBase --K 10 --dataset Citeseer --dprate 0.5 --q 1
python training.py --net ChebBase --K 10 --dataset Pubmed --dprate 0.8 --q 1
```

### The results of ChebNetII in Table 6

You can run the following script. 
```sh
sh semi-rnd.sh
```

### The results of ChebNetII in Table 7

You can run the following script. 
```sh
sh full.sh
```
