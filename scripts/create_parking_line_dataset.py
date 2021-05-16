import ipdb
from avcv import *


# ROOT_DIR = "/workdir/data/fisheye-parking/1k8_12Mar"
ROOT_DIR = "/data/fisheye-parking/"

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

    # boxes = []
    lines = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        objects = content.split('\n')
        for obj in objects:
            if len(obj.split()) == 9:
                class_name = obj.split()[0]
                points = np.array(obj.split()[1:9]).astype(int).reshape([-1,2])
                for i,j in zip([0,1], [2,3]):
                    p1 = points[i].tolist()
                    p2 = points[j].tolist()
                    line = [*p1, 2, *p2, 2]
                    lines.append(line)
    
    return lines


def _load_annotation_v2(im_path):
    filename = im_path.replace('/image/', '/label/')[:-4] + '.txt'

    # boxes = []
    lines = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        objects = content.split('\n')
        for obj in objects:
            if len(obj.split()) == 9:
                class_name = obj.split()[0]
                points = np.array(obj.split()[1:9]).astype(int).reshape([-1,2])
                cx = int(points[:,0].mean())
                cy = int(points[:,1].mean())
                # import ipdb; ipdb.set_trace()
                for i in range(4):
                    p1 = points[i].tolist()
                    line = [*p1, 2, cx,cy,2]
                    lines.append(line)
    
    return lines


# dataset = 'train'
# for dataset in ['train', 'val']:
for dataset in ["all_data"]:
    out_json_path = osp.join(ROOT_DIR, f'{dataset}_keypoints.json')
    img_dir = osp.join(ROOT_DIR, dataset, 'image')
    image_paths = get_paths(img_dir)
    assert len(image_paths), img_dir
    ann_id = 0
    images = []
    annotations = []
    for image_id, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        img = dict()
        img['file_name'] = osp.basename(image_path)
        img['id'] = image_id
        images += [img]
        for kp in _load_annotation_v2(image_path):
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
        {'supercategory': 'parking_lot', 'id': 1, 'name': 'line',
         'keypoints': ['p1', 'p2'],
         'skeleton': [[1, 2]],
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
