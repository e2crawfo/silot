python run.py moving_mnist baseline --min-digits=2 --max-digits=2 --n-train=100 --n-val=256
python run.py moving_mnist baseline --min-digits=4 --max-digits=4 --n-train=100 --n-val=256
python run.py moving_mnist baseline --min-digits=6 --max-digits=6 --n-train=100 --n-val=256
python run.py moving_mnist baseline --min-digits=8 --max-digits=8 --n-train=100 --n-val=256
python run.py big_shapes shapes_baseline_AP --render-step=100000000000
python run.py big_shapes shapes_baseline_count_1norm --render-step=100000000000
python run.py big_shapes shapes_baseline_mota --render-step=100000000000