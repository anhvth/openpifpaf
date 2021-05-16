basenet=resnet50
epoch=1000
# CKPT=outputs/$(ls outputs | grep $basenet | grep $epoch |grep line)
CKPT=outputs/mobilenetv2-210516-070123-parking_line_kp.pkl.epoch020
echo "Full ckpt path:" $CKPT

# IMAGE="/data/fisheye-parking/1k8_12Mar/train/image/20210226_152537-stitch-00045.png"
# IMAGE="/data/fisheye-parking/1k8_12Mar/val/image/20210*.png"
IMAGE="/data/fisheye-parking/1k8_12Mar/train/image/*.png"
python openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 \
    --show --debug  --save-all --debug-indices cif:1 cif:0  caf:1 caf:0  --ablation-cifseeds-no-rescore --ablation-caf-no-rescore --disable-cuda