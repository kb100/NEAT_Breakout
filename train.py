import neat
from breakout import Breakout
import pickle
import numpy as np
import random
import multiprocessing
import itertools, functools
import breakout
import visualize


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
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 1000)
    visualize.draw_net(config, winner, view=False, filename="winner_net_graph.pdf")
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)

    with open("population.data","wb") as f:
        pickle.dump(p,f)

    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open("winner.net","wb") as f:
        pickle.dump(winner_net, f)

if __name__ == "__main__":
    run("train_config")
