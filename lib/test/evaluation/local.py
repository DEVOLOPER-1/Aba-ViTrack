from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    prj_path = '/home/youssef_mohammad/projects/Aba-ViTrack'

    settings.davis_dir = ''
    settings.got10k_lmdb_path = f'{prj_path}/data/got10k_lmdb'
    settings.got10k_path = f'{prj_path}/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = f'{prj_path}/data/itb'
    settings.lasot_extension_subset_path_path = f'{prj_path}/data/lasot_extension_subset'
    settings.lasot_lmdb_path = f'{prj_path}/data/lasot_lmdb'
    settings.lasot_path = f'{prj_path}/data/lasot'
    settings.network_path = f'{prj_path}/output/test/networks'  # Where tracking networks are stored.
    settings.nfs_path = f'{prj_path}/data/nfs'
    settings.otb_path = f'{prj_path}/data/otb'
    settings.prj_dir = prj_path
    settings.result_plot_path = f'{prj_path}/output/test/result_plots'
    settings.results_path = f'{prj_path}/output/test/tracking_results'  # Where to store tracking results
    settings.save_dir = f'{prj_path}/output'
    settings.segmentation_path = f'{prj_path}/output/test/segmentation_results'
    settings.tc128_path = f'{prj_path}/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = f'{prj_path}/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = f'{prj_path}/data/trackingnet'
    settings.uav_path = f'{prj_path}/data/uav'
    settings.vot18_path = f'{prj_path}/data/vot2018'
    settings.vot22_path = f'{prj_path}/data/vot2022'
    settings.vot_path = f'{prj_path}/data/VOT2019'
    settings.youtubevos_dir = ''

    # Competition Data path
    settings.mtc_aic4_dir = '/mnt/contest_release_data'

    # test datasets
    settings.dtb70_path = f'{prj_path}/dataset/DTB70'
    settings.uavdt_path = f'{prj_path}/dataset/UAVDT'
    settings.visdrone_path = f'{prj_path}/dataset/VisDrone2018'
    settings.uav123_10fps_path = f'{prj_path}/dataset/UAV123@10FPS'
    settings.uav123_path = f'{prj_path}/dataset/UAV123'
    settings.uavtrack_path = f'{prj_path}/dataset/UAVTrack112'

    return settings