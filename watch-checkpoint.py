import neat
from neat.six_util import iteritems, itervalues
from breakout import Breakout
import sys
from train import eval_genomes

def playGenome(genome, config):
    return Breakout(genome=genome, config=config, gui=True).play()

def playBestGenomeFromCheckpoint(filename):
    p = neat.checkpoint.Checkpointer.restore_checkpoint(filename)
    config = p.config
    best = p.run(eval_genomes,1)
    print(best)
    return playGenome(best, config)

if __name__ == "__main__":
    filename = sys.argv[1]
    print("Playing from", filename)
    score = playBestGenomeFromCheckpoint(filename)
    print("Final score was", score)

