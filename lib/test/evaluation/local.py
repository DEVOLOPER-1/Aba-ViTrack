from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/got10k_lmdb'
    settings.got10k_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/itb'
    settings.lasot_extension_subset_path_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/lasot_lmdb'
    settings.lasot_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/lasot'
    settings.network_path = '/home/maro/final-projects/Aba-ViTrack/outputs/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/nfs'
    settings.otb_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/otb'
    settings.prj_dir = '/home/maro/final-projects/Aba-ViTrack'
    settings.result_plot_path = '/home/maro/final-projects/Aba-ViTrack/outputs/test/result_plots'
    settings.results_path = '/home/maro/final-projects/Aba-ViTrack/outputs/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/maro/final-projects/Aba-ViTrack/outputs'
    settings.segmentation_path = '/home/maro/final-projects/Aba-ViTrack/outputs/test/segmentation_results'
    settings.tc128_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/trackingnet'
    settings.uav_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/uav'
    settings.vot18_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/vot2018'
    settings.vot22_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/vot2022'
    settings.vot_path = '/media/maro/Mom0-0/Datasets/MTC-AIC/raw/VOT2019'
    settings.youtubevos_dir = ''

    return settings

