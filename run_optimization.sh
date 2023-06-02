export PYTHONPATH=.

# python optimization/run_optimization.py\
#     --mode=free_generation

python optimization/run_optimization.py\
    --mode="edit"\
    --description="a woman with red hair"\
    --save_intermediate_image_every=30

