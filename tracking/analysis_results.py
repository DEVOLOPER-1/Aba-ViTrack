import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'mtc_aic4'

# 1. Load the protected Kalman run
trackers.extend(trackerlist(
    name='abavitrack',
    parameter_name='abavit_kalman',
    dataset_name=dataset_name,
    run_ids=None,
    display_name='AbaViTrack (Kalman + Gamma)'
))

# 2. Load the new Baseline run
trackers.extend(trackerlist(
    name='abavitrack',
    parameter_name='abavit_patch16_224',
    dataset_name=dataset_name,
    run_ids=None,
    display_name='AbaViTrack (Baseline)'
))

dataset = get_dataset(dataset_name)

# 3. Print the comparison table and plot the curves!
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))