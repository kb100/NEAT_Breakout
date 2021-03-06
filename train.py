import neat
from breakout import Breakout
import pickle
import numpy as np
import multiprocessing
import itertools
import breakout


def eval_genome(genome_info, config):
    genome_id, genome = genome_info
    score = Breakout(genome=genome, config=config, gui=False).play()
    return score

def eval_genomes(genomes, config):
    #print(set(itertools.starmap(eval_genome, zip(genomes, itertools.repeat(config)))))
    with multiprocessing.Pool(processes=6) as pool:
        fitnesses = pool.starmap(eval_genome, zip(genomes, itertools.repeat(config)))
        for (genome_id, genome), fitness in zip(genomes, fitnesses):
            genome.fitness = fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    winner = p.run(eval_genomes, 1000)

    with open("population.data","wb") as f:
        pickle.dump(p,f)

    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open("winner.genome","wb") as f:
        pickle.dump((winner,config), f)

if __name__ == "__main__":
    run("train_config")
