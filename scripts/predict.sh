basenet=resnet50
epoch=1000
CKPT=outputs/$(ls outputs | grep $basenet | grep $epoch |grep line)
echo "Full ckpt path:" $CKPT

IMAGE="/data/fisheye-parking/1k8_12Mar/train/image/20210226_152537-stitch-00045.png"
python openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 \
    --show --debug  --save-all --debug-indices cif:1 cif:0  caf:1 caf:0  --ablation-cifseeds-no-rescore --ablation-caf-no-rescore