# Use avcv to make 1 image coco dataset 
# av debug_make_mini_dataset -a mini-coco/annotations/mini_json.json mini-coco/images/ out 1 000000008211


# ln -s /root/openpifpaf/data/out data-mscoco
# ln -s /root/openpifpaf/data/out/annotations/mini_json.json data-mscoco/annotations/person_keypoints_train2017.json
# ln -s /root/openpifpaf/data/out/annotations/mini_json.json data-mscoco/annotations/person_keypoints_val2017.json
# ln -s /root/openpifpaf/data/out/images data-mscoco/images/train2017
# ln -s /root/openpifpaf/data/out/images data-mscoco/images/val2017
# python openpifpaf/train.py \
#   --lr=10e-4 --momentum=0.9 --b-scale=5.0 \
#   --epochs=1000 --lr-warm-up-epochs=100 \
#   --batch-size=1 --train-batches=1 --val-batches=1 --val-interval=100 \
#   --weight-decay=1e-5 \
#   --dataset=cocokp --cocokp-upsample=2 --cocokp-no-augmentation \
#   --basenet=resnet50 --debug --debug-indices cif:1 --save-all


#--------------- Overfit -lot
# python openpifpaf/train.py \
#   --lr=10e-4 --momentum=0.9 --b-scale=5.0 \
#   --epochs=1000 --lr-warm-up-epochs=100 \
#   --batch-size=1 --train-batches=1 --val-batches=1 --val-interval=100 \
#   --weight-decay=1e-5 \
#   --dataset=parking_line_kp --parking_line_kp-upsample=2 --parking_line_kp-no-augmentation \
#   --basenet=resnet50 


#--------------- Overfit - line
rm -r ./outputs
python openpifpaf/train.py \
  --lr=10e-4 --momentum=0.9 --b-scale=5.0 \
  --epochs=1000 --lr-warm-up-epochs=100 \
  --batch-size=1 --train-batches=1 --val-batches=1 --val-interval=1000 \
  --weight-decay=1e-5 \
  --dataset=parking_line_kp --parking_line_kp-upsample=2 --parking_line_kp-no-augmentation \
  --basenet=resnet50 $@


  # dpython openpifpaf/train.py \
  # --lr=10e-4 --momentum=0.9 --b-scale=5.0 \
  # --epochs=1000 --lr-warm-up-epochs=100 \
  # --batch-size=1 --train-batches=1 --val-batches=1 --val-interval=1000 \
  # --weight-decay=1e-5 \
  # --dataset=parking_line_kp --parking_line_kp-upsample=2 --parking_line_kp-no-augmentation \
  # --basenet=resnet50 --debug --debug-indices caf:1 cif:1 --save-all