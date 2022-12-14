python training.py --net ChebNetII --dataset Chameleon  --full True --lr 0.05  --weight_decay 0.0 --prop_wd 0.0 --dprate 0.6
python training.py --net ChebNetII --dataset Squirrel  --full True --lr 0.05  --weight_decay 0.0 --prop_wd 0.0005 --dprate 0.5
python training.py --net ChebNetII --dataset Actor  --full True --lr 0.05  --weight_decay 0.0  --prop_wd 0.0 --dprate 0.9 --prop_lr 0.01
python training.py --net ChebNetII --dataset Texas  --full True --lr 0.05 --weight_decay 0.0005 --prop_wd 0.0 --dprate 0.6 --prop_lr 0.001
python training.py --net ChebNetII --dataset Cornell  --full True --lr 0.05 --weight_decay 0.0005 --prop_wd 0.0005 --dprate 0.5 --prop_lr 0.001
python training.py --net ChebNetII --dataset Cora  --full True --lr 0.01  --weight_decay 0.0005 --dprate 0.0
python training.py --net ChebNetII --dataset Citeseer  --full True --lr 0.01  --weight_decay 0.0005 --prop_wd 0.0
python training.py --net ChebNetII --dataset Pubmed  --full True --lr 0.01  --weight_decay 0.0 --prop_wd 0.0 --dprate 0.0
