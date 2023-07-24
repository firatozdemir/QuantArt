# # Stage-1.1
python -u main.py \
--base configs/vqgan_oadat_sim.yaml \
--train True --gpus 0,1

# Stage-1.2
python -u main.py \
--base configs/vqgan_oadat_swfd.yaml \
--train True --gpus 0,1

# Stage-2
# python -u main.py \
# --base configs/oadat_sim2invivo.yaml \
# --train True --gpus 0,1