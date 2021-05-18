
CKPT=outputs/mobilenetv2-210516-074942-parking_line_kp.pkl.epoch090
echo "Full ckpt path:" $CKPT

IMAGE="/data/fisheye-parking/1k8_12Mar/train/image/*.png"
# IMAGE=/data/fisheye-parking/1k8_12Mar/train/image/20210226_153123-stitch-00000.png

rm -r all-images

# python openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 \
#     --show --debug  --show-decoding-order $@ 


IMAGE=/data/fisheye-parking/1k8_12Mar/train/image/20210226_153123-stitch-00000.png

echo "\n---DEBUG ---"
x="dpython openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 \
    --show --debug  --show-decoding-order $@ "
echo $x

#  --save-all --debug-indices cif:1