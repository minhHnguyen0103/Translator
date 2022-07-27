import argparse

p = argparse.ArgumentParser()
p.add_argument('-l', type=int, default=12, dest='lr')
p = p.parse_args()
print(p.lr)