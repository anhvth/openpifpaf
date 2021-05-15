# CKPT="outputs/resnet50-210514-045004-cocokp.pkl.epoch1000"
basenet=resnet50
epoch=1000
dataset='park'
CKPT=outputs/$(ls outputs | grep $basenet | grep $epoch |grep $dataset)

IMAGE="/data/fisheye-parking/1k8_12Mar/train/image/20210226_152537-stitch-00045.png"
# IMAGE="./data-mscoco/images/val2017/000000008211.jpg"
rm -r all-images

dpython openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 --show --debug  --save-all
#--- Summary

echo "Full ckpt path:" $CKPT