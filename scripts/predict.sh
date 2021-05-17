
CKPT=outputs/mobilenetv2-210516-074942-parking_line_kp.pkl.epoch090
echo "Full ckpt path:" $CKPT

IMAGE=/data/fisheye-parking/1k8_12Mar/train/image/20210226_152537-stitch-00045.png
IMAGE=/data/fisheye-parking/1k8_12Mar/train/image/*.png
rm -r all-images
python openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 \
    --show --debug   $@

#  --save-all --debug-indices cif:1