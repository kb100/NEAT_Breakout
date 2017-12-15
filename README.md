# NEAT Breakout

In order to run the demo, setup a vitual environment with all the dependencies:

```
virtualenv NEAT --python=python3
cd NEAT
source bin/activate
pip install neat-python
pip install pygame
pip install numpy

git clone "https://github.com/kb100/NEAT_Breakout.git"
cd NEAT_Breakout
./demo.sh
```

You can play the game with `python3 breakout.py`.

You can watch a checkpoint with e.g. `python3 watch-checkpoint.py neat-checkpoint-49`.

You can play a genome with the same command e.g. `python3 watch-checkpoint.py winner.genome`.

