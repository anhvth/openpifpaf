basenet=resnet50
epoch=1000
CKPT=outputs/$(ls outputs | grep $basenet | grep $epoch |grep line)
echo "Full ckpt path:" $CKPT

IMAGE="/data/fisheye-parking/1k8_12Mar/val/image/20210226_153123-stitch-00000.png"
rm -r all-images

dpython openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 --show --debug  --save-all $@