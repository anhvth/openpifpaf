from avcv import *
import openpifpaf
import numpy as np
import mmcv
import torch
import timm
import logging


LOG = logging.getLogger(__name__)

LOG.setLevel(logging.DEBUG)


def get_cif_field_generator(vis_indices=[0, 1], bmin=0.1, backbone_num_out_channels=2048):
    cif_head_meta = openpifpaf.headmeta.Cif('cif', 'parking_line_kp',
                                            keypoints=['p1', 'p2'],
                                            sigmas=[0.0025, 0.0025],
                                            pose=np.array([[0., 0., 2.],
                                                           [10., 0., 10.]]),
                                            draw_skeleton=[[1, 2], [2, 1]],
                                            score_weights=[3, 3])
    cif_head_meta.base_stride = 16
    cif_head_meta.upsample_stride = 2
    pif_encoder = openpifpaf.encoder.Cif(cif_head_meta, bmin=bmin)
    visualizer = openpifpaf.visualizer.Cif(pif_encoder.meta)
    visualizer.all_indices = [('cif', i, 'all') for i in vis_indices]

    pif_head = openpifpaf.network.heads.CompositeField3(cif_head_meta, backbone_num_out_channels)
    loss = openpifpaf.network.losses.Factory().factory([pif_head])

    return pif_encoder, pif_head, loss, visualizer, cif_head_meta


if __name__ == '__main__':
    # pass
    print(timm.list_models())
    backbone = timm.create_model('tf_efficientnet_lite0')
    #     
    # img = mmcv.imread("/data/fisheye-parking/1k8_12Mar/val/image/20210226_153123-stitch-00000.png")
    # img = img /255
    # img = torch.from_numpy(img)[None].permute([0,3,1,2])


    pif_encoder, pif_head, loss_fn, pif_visualizer, cif_head_meta = get_cif_field_generator()
    optim = torch.optim.Adam(pif_head.parameters(), 0.00005)

    image, anns, img_meta = mmcv.load('pif_input.pkl')
    # img_meta['valid_area'] = np.array([0,0, 768,598])
    img_meta['scale'] = np.array([1,1])
    # img_meta['offset']=np.array([0.,   0.]),
    target_cif = pif_encoder(image, anns, img_meta)

    # pif_visualizer.targets(target_cif, annotation_dicts=anns)
    pif_visualizer.predicted(target_cif)

    # import ipdb; ipdb.set_trace()
    cifhr_instance = openpifpaf.decoder.utils.CifHr()
    cif_head_meta.head_index = 0
    cifhr = cifhr_instance.fill(target_cif[None].cpu().numpy(), [cif_head_meta])
    import matplotlib.pyplot as plt; plt.imshow(cifhr.accumulated[0]); plt.savefig('test.jpg'); plt.close()
    # x = mmcv.load("self.basenet(image_path).pkl")# backbone features



    # for i in range(1):
    #     pif_outputs = pif_head(x)
    #     optim.zero_grad()
    #     loss = loss_fn([pif_outputs], [target_cif[None]])

    #     total_loss = loss[0] + sum(loss[1])
    #     print(i, loss)
    #     total_loss.backward()
    #     optim.step()

    #     pif_visualizer.predicted(pif_outputs[0].detach().cpu().numpy(), name_signature=f'cif-{i}')