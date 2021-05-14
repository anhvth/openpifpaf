# CKPT="outputs/resnet50-210514-045004-cocokp.pkl.epoch1000"
basenet=resnet50
epoch=1000
CKPT=outputs/$(ls outputs | grep $basenet | grep $epoch)

IMAGE="data/out/images/000000008211.jpg"
rm -r all-images

python openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 --show --debug  --save-all $@
#--- Summary

echo "Full ckpt path:" $CKPT