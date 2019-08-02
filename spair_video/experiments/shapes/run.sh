# python silot_run.py long --max-shapes=10
# python silot_run.py long --small --max-shapes=10
# python silot_run.py long --max-shapes=20
# python silot_run.py long --small --max-shapes=20
# python silot_run.py long --max-shapes=30
# python silot_run.py long --small --max-shapes=30
# python silot_run.py short --max-shapes=10
# python silot_run.py short --small --max-shapes=10
# python silot_run.py short --max-shapes=20
# python silot_run.py short --small --max-shapes=20
# python silot_run.py short --max-shapes=30
# python silot_run.py short --small --max-shapes=30
# python silot_run.py local --small --max-shapes=20 --render-step=5000 --final-count-prior-log-odds=0.00625
python silot_run.py local --small --max-shapes=20 --render-step=5000 --final-count-prior-log-odds=0.003125
# python silot_run.py local --small --max-shapes=20 --render-step=5000 --final-count-prior-log-odds=0.0015625 --max-steps=80000