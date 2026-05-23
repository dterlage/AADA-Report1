import sys
import time
from dataset import *
from data_generation import *
from kmeans import *
from visualizer import plot_one, plot_two

def run(path_in, path_out):
    """
    Reads the input set, clusters the points and writes to output

    :param path_in:     location of the input set
    :param path_out:    location to print the output
    """

    # read input from file
    try:
        with open(path_in, "r") as f:
            input_obj = Dataset.read_input(f)
    except IOError:
        print("Could not read input file: " + path_in, file=sys.stderr)
        return

    methods = {"Gonzalez": init_Gonzalez(input_obj), "kmeans++":init_kmeans_plusplus(input_obj)}
    results = {}

    print("Cluster counter:      ", input_obj.k, file=sys.stderr)

    for name, method in methods.items():
        start = time.time()

        # find the best centroids using k_means
        centroids = cluster(input_obj, method)

        # simple check if the correct number of centroids has been given
        assert len(centroids) == input_obj.k

        score = input_obj.avg_score(centroids)
        end = time.time()
        time_taken = end - start

        # print result to file
        try:
            input_obj.write_output(centroids, path_out + f"_{name}", 1)      # I've set the assignment number to 1
        except IOError:
            print("Could not write output to file: " + path_out, file=sys.stderr)

        print("Initialization method: {}".format(name), file=sys.stderr)
        print("Mean squared distance: {:.3f}".format(input_obj.avg_score(centroids)), file=sys.stderr)
        print("Time taken:            {:.3f}s".format(time_taken), file=sys.stderr)
        print("")

        results[name] = [score, time_taken]

    return results


def run_experiment(name, param_list, sample, k, n_outliers, d, std, range, min_dist, seed, param_type='dist'):
    results = {}

    for param in param_list:
        n_out    = round(sample * 0.01 * param) if param_type == 'pct' else n_outliers
        filename = "data_in/{0}_in/{0}_{1}.in".format(name, param)
        outfile  = "data_out/{0}_out/{0}_{1}.out".format(name, param)

        data = generate_data(sample, k, n_out, d, std, range, min_dist, seed)
        write_input_file(filename, data, k)
        results[param] = run(filename, outfile)

    return results


def plot_experiment(name, param, results):
    gonzalez = './data_out/{0}_out/{0}_{1}.out_Gonzalez'.format(name, param)
    kmeans   = './data_out/{0}_out/{0}_{1}.out_kmeans++'.format(name, param)
    plot_two(gonzalez, kmeans, results[param])




### Experimental setup
# Base case: n=2500, k=5, outliers=500, d=2, std=1.5, range=15.0, dist=3
k = 5  # Cluster count


# Exp 1: Vary over distance
distances  = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
results_e1 = run_experiment('exp1', distances, 2500, k, 500, 2, 1.5, 15.0, 3.0, 421, param_type='dist')

plot_experiment('exp1', 1.0, results_e1) # Change param to see diff graph



# Exp 2: Vary over proportion of outliers
proportions = [5, 10, 25, 50, 75, 90, 95]
results_e2  = run_experiment('exp2', proportions, 2500, k, None, 2, 1.5, 15.0, 3.0, 422, param_type='pct')

plot_experiment('exp2', 5, results_e2) # Change param to see diff graph



# Exp 3: Repeat Experiment 1 and 2 in 3D
results_e3_dist = run_experiment('exp3_dist', distances,   2500, k, 500,  3, 3, 25.0, 3.0, 423, param_type='dist')
results_e3_pct  = run_experiment('exp3_pct',  proportions, 2500, k, None, 3, 3, 25.0, 3.0, 424, param_type='pct')

# Plot examples
plot_one("data_out/exp3_dist_out/exp3_dist_1.0.out_Gonzalez") # Change path to see diff graph
