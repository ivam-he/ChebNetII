python training.py --net ChebNetII --dataset Chameleon --K 10 --prop_lr 0.0005 --prop_wd 0.0 --dprate 0.5 --semi_fix True
python training.py --net ChebNetII --dataset Squirrel --K 10 --prop_lr 0.0005 --prop_wd 0.0 --dprate 0.5 --semi_fix True
python training.py --net ChebNetII --dataset Actor --K 10 --prop_lr 0.01 --prop_wd 0.0005 --dprate 0.9 --dropout 0.5 --semi_fix True 
python training.py --net ChebNetII --dataset Texas --K 10 --prop_lr 0.001 --prop_wd 0.0 --dprate 0.7 --dropout 0.6 --semi_fix True  
python training.py --net ChebNetII --dataset Cornell --K 10 --prop_lr 0.001 --prop_wd 0.0 --dprate 0.7 --dropout 0.6 --semi_fix True
python training.py --net ChebNetII --dataset Cora --K 10 --prop_lr 0.001 --prop_wd 0.05 --dprate 0.8 --semi_fix True
python training.py --net ChebNetII --dataset Citeseer --K 10  --prop_lr 0.001 --prop_wd 0.05 --dprate 0.5 --semi_fix True
python training.py --net ChebNetII --dataset Pubmed --K 10 --prop_lr 0.005 --prop_wd 0.05 --dprate 0.5 --semi_fix True   