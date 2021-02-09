# Our experiments run every experiment 10 times with seeds in [0, 9].
DEVICE=0 # represents the GPU index.
SEED=0   # represents the random seed.

python main.py --graph cora --seed $SEED --gpu $DEVICE --print-every 100
python main.py --graph citeseer --seed $SEED --gpu $DEVICE --print-every 100
python main.py --graph pubmed --seed $SEED --gpu $DEVICE --print-every 100
python main.py --graph cora-ml --seed $SEED --gpu $DEVICE --print-every 100
python main.py --graph dblp --seed $SEED --gpu $DEVICE --print-every 100
python main.py --graph amazon --seed $SEED --gpu $DEVICE --print-every 100
