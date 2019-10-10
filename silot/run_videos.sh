# carnival
# python run.py carnival atari_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/atari/run/carnival/exp_alg=atari-train-silot_2019_09_02_11_55_34_seed=579349643/weights/checkpoint_stage_1 --env-name=long_video --alg-name=carnival
# 
# # wizard of wor
# python run.py wizard_of_wor atari_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/atari/run/wizard_of_wor/exp_alg=atari-train-silot_2019_08_06_08_36_47_seed=886551525/weights/checkpoint_stage_2 --env-name=long_video --alg-name=wizard_of_wor
# 
# # space invaders
# python run.py space_invaders atari_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/atari/run/space_invaders/exp_alg=atari-train-silot_2019_08_29_22_49_02_seed=1380144060/weights/checkpoint_stage_2 --env-name=long_video --alg-name=space_invaders
# 
# # mnist
# python run.py moving_mnist mnist_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/mnist/run/run_env=moving-mnist_max-digits=12_alg=conv-silot_duration=long_2019_07_08_04_06_03_seed=0/experiments/exp_idx=0_repeat=0_2019_07_08_06_30_44_seed=2083385566/weights/best_of_stage_2 --env-name=long_video --alg-name=mnist
# 
# # big, 30
# python run.py big_shapes shapes_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/shapes/run/run_env=big-shapes_max-shapes=30_alg=shapes-silot_duration=restart_2019_08_07_12_39_32_seed=0/experiments/exp_idx=0_repeat=0_2019_08_09_01_39_27_seed=724209602/weights/best_of_stage_1 --env-name=long_video --alg-name=shapes,big,30

# small, 30
python run.py big_shapes shapes_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/shapes/run/run_env=big-shapes-small_max-shapes=30_alg=shapes-silot_duration=long_2019_08_05_10_25_53_seed=0/experiments/exp_idx=0_repeat=0_2019_08_08_16_32_02_seed=1264917965/weights/best_of_stage_2 --env-name=long_video --alg-name=shapes,small,30

# big, 10
python run.py big_shapes shapes_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/shapes/run/run_env=big-shapes_max-shapes=10_alg=shapes-silot_duration=restart10_2019_08_19_17_39_20_seed=0/experiments/exp_idx=0_repeat=0_2019_08_19_20_12_48_seed=186422792/weights/best_of_stage_1 --env-name=long_video --alg-name=shapes,big,10

# small, 10
python run.py big_shapes shapes_long_video_silot --load-path=/media/data/Dropbox/experiment_data/active/aaai_2020/shapes/run/run_env=big-shapes-small_max-shapes=10_alg=shapes-silot_duration=long_2019_08_14_22_21_50_seed=0/experiments/exp_idx=0_repeat=0_2019_08_15_14_12_10_seed=1604628506/weights/best_of_stage_1 --env-name=long_video --alg-name=shapes,small,10