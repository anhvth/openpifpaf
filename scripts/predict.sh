CKPT="outputs/resnet50-210514-045004-cocokp.pkl.epoch1000"
IMAGE="data/out/images/000000008211.jpg"
python openpifpaf/predict.py $IMAGE --checkpoint $CKPT  --show --debug --long-edge=385 --save-all $@