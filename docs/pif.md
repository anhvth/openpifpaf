# Pif
model output shape: b,n,5,h,w
targets: same with model output shape


To create targets for training:
    file: openpifpaf/transforms/encoders.py
    -> openpifpaf/encoder/cif.py
    use class CirGenerator

    bg_mask_shape = [h/s,w/s]
    
