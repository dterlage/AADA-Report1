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




### Experimental setup
# Base case: n=2500, k=5, outliers=500, d=2, std=1.5, clusterbox=15.0, dist=3
k = 5


## Exp 1: Vary over distance outliers
distances = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]      # * sd of created clusters
results_e1 = {}

for i in distances:
    e1_data = generate_data(2500, k, 500, 2, 1.5, 15.0, i, 421)
    write_input_file("data_in/exp1_in/E1_{:.1f}dist.in".format(i), e1_data, k)
    results_e1[i] = run("data_in/exp1_in/E1_{:.1f}dist.in".format(i), "data_out/exp1_out/E1_{}dist.out".format(i))

# Request a plot
req_plot = 1.0
plot_two('./data_out/exp1_out/E1_{:.1f}dist.out_Gonzalez'.format(req_plot),
         './data_out/exp1_out/E1_{:.1f}dist.out_kmeans++'.format(req_plot),
          results_e1[req_plot])




## Exp 2: Vary over percentage of outliers
proportion = [5, 10, 25, 50, 75, 90, 95]     # in %
results_e2 = {}

for i in proportion:
    sample = 2500
    outliers = round(sample * 0.01 * i)
    e2_data = generate_data(sample, k, outliers, 2, 1.5, 15.0, 3, 422)
    write_input_file("data_in/exp2_in/E2_{}%.in".format(i), e2_data, k)
    results_e2[i] = run("data_in/exp2_in/E2_{}%.in".format(i), "data_out/exp2_out/E2_{}%.out".format(i))

# Request a plot
req_plot2 = 5
plot_two('./data_out/exp2_out/E2_{}%.out_Gonzalez'.format(req_plot2),
         './data_out/exp2_out/E2_{}%.out_kmeans++'.format(req_plot2),
          results_e2[req_plot2])



## Exp 3: impact of 3d (larger cluster box and std used to spread out points more
e3_data = generate_data(2500, k, 500, 3, 3, 25.0, 3, 423)
write_input_file("data_in/exp3_in/E3.in", e3_data, k)
results_e3 = run("data_in/exp3_in/E3.in", "data_out/exp3_out/E3.out")


plot_one('./data_out/exp3_out/E3.out_Gonzalez')


# Optional experiments: Repeat 1 and 2 in 3d, sparse clusters