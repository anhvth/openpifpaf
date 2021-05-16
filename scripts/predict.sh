basenet=resnet50
epoch=1000
# CKPT=outputs/$(ls outputs | grep $basenet | grep $epoch |grep line)
CKPT=outputs/mobilenetv2-210516-074942-parking_line_kp.pkl.epoch030
echo "Full ckpt path:" $CKPT

# IMAGE="/data/fisheye-parking/1k8_12Mar/train/image/20210226_152537-stitch-00045.png"
# IMAGE="/data/fisheye-parking/1k8_12Mar/val/image/20210*.png"
# IMAGE="/data/fisheye-parking/1k8_12Mar/val/image/*.png"
IMAGE="/data/fisheye-parking/all_data/image/*.png"
# python openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 \
#     --show --debug  --save-all --debug-indices cif:1  --ablation-cifseeds-no-rescore --ablation-caf-no-rescore 
rm -r ./all-images
python openpifpaf/predict.py $IMAGE --checkpoint $CKPT --long-edge=385 \
    --show --debug  --save-all --debug-indices cif:0