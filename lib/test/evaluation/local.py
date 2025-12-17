from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '{{PROJECT PATH}}/data/got10k_lmdb'
    settings.got10k_path = '{{PROJECT PATH}}/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '{{PROJECT PATH}}/data/itb'
    settings.lasot_extension_subset_path_path = '{{PROJECT PATH}}/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '{{PROJECT PATH}}/data/lasot_lmdb'
    settings.lasot_path = '{{PROJECT PATH}}/data/lasot'
    settings.network_path = '{{PROJECT PATH}}/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '{{PROJECT PATH}}/data/nfs'
    settings.otb_path = '{{PROJECT PATH}}/data/otb'
    settings.prj_dir = '{{PROJECT PATH}}'
    settings.result_plot_path = '{{PROJECT PATH}}/output/test/result_plots'
    settings.results_path = '{{PROJECT PATH}}/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '{{PROJECT PATH}}/output'
    settings.segmentation_path = '{{PROJECT PATH}}/output/test/segmentation_results'
    settings.tc128_path = '{{PROJECT PATH}}/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '{{PROJECT PATH}}/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '{{PROJECT PATH}}/data/trackingnet'
    settings.uav_path = '{{PROJECT PATH}}/data/uav'
    settings.vot18_path = '{{PROJECT PATH}}/data/vot2018'
    settings.vot22_path = '{{PROJECT PATH}}/data/vot2022'
    settings.vot_path = '{{PROJECT PATH}}/data/VOT2019'
    settings.youtubevos_dir = ''

    # test datasets
    settings.dtb70_path = '{{PROJECT PATH}}/dataset/DTB70'
    settings.uavdt_path = '{{PROJECT PATH}}/dataset/UAVDT'
    settings.visdrone_path = '{{PROJECT PATH}}/dataset/VisDrone2018'
    settings.uav123_10fps_path = '{{PROJECT PATH}}/dataset/UAV123@10FPS'
    settings.uav123_path = '{{PROJECT PATH}}/dataset/UAV123'
    settings.uavtrack_path = '{{PROJECT PATH}}/dataset/UAVTrack112'

    return settings

