import ipdb
from avcv import *


ROOT_DIR = "/workdir/data/fisheye-parking/1k8_12Mar"


def process(box):
    out = []
    for i in range(len(box)):
        if i % 2 == 0:  # x
            out.append(int(box[i]))
        else:
            out += [int(box[i]), 2]
    return out

def _load_annotation(im_path):
    filename = im_path.replace('/image/', '/label/')[:-4] + '.txt'

    boxes = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        objects = content.split('\n')
        for obj in objects:
            if len(obj.split()) == 9:
                class_name = obj.split()[0]
                # if class_name != '0':
                # continue
                box = obj.split()[1:9]
                boxes.append(box)
    boxes = [process(_) for _ in boxes]
    return boxes

# dataset = 'train'
for dataset in ['train', 'val']:
    out_json_path = osp.join(ROOT_DIR, f'{dataset}_keypoints.json')
    img_dir = osp.join(ROOT_DIR, dataset, 'image')
    image_paths = get_paths(img_dir)
    assert len(image_paths), img_dir
    # sample = read_json('data-mscoco/annotations/mini_json.json')
    # import ipdb; ipdb.set_trace()
    # _ann = sample['annotations'][0]
    # _img = sample['images'][0]
    ann_id = 0
    images = []
    annotations = []
    for image_id, image_path in enumerate(image_paths):
        img = dict()
        img['file_name'] = osp.basename(image_path)
        img['id'] = image_id
        images += [img]
        for kp in _load_annotation(image_path):
            ann = dict()
            ann['num_keypoints'] = len(kp)//3
            ann['keypoints'] = kp
            ann['iscrowd'] = False
            ann['image_id'] = image_id
            point = np.array(kp).reshape([-1, 3])[:, :2]
            x, y, w, h = cv2.boundingRect(point)
            ann['bbox'] = [x, y, w, h]

            ann['id'] = ann_id
            ann['category_id'] = 1
            ann_id += 1
            annotations += [ann]

    categories = [
        {'supercategory': 'parking_lot', 'id': 1, 'name': 'parking_lot',
         'keypoints': ['p1', 'p2', 'p3', 'p4'],
         'skeleton': [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
         'id':1,
         }
    ]
    assert len(images) and len(annotations)
    out = dict(
        images=images, annotations=annotations, categories=categories,
    )
    with open(out_json_path, 'w') as f:
        json.dump(out, f)
    print('dump:', out_json_path)
