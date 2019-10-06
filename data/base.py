import os


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def find_images_and_annotation(root_dir, attr_anno_file, base_dir=None):
    images = {}
    attr_file = None
    assert os.path.isdir(root_dir), f"{root_dir} does not exist"
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images[os.path.splitext(fname)[0]] = path
            elif fname.lower() == attr_anno_file:
                attr_file = os.path.join(root, fname)

    assert attr_file is not None, f"Failed to find {attr_anno_file}"

    # parse all image
    print("Parsing all images and their attributes...")
    datas = []
    with open(attr_file, "r") as f:
        attr_names = []
        lines = f.readlines()
        attr_names = lines[0].split(" ")
        for idx, line in enumerate(lines[1:]):
            line = line.strip()
            line = line.split("  ")
            fname = os.path.splitext(line[0])[0]
            font = idx // 52
            char = int(fname.split('_')[1].split('.')[0])
            attr_vals = [(float(v)/100.0) for v in line[1:]]
            assert len(attr_vals) == len(attr_names), f"{fname} has only {len(attr_vals)} attributes"
            datas.append({
                "image": images[fname],
                "font": font,
                "char": char,
                "attr": attr_vals
            })
    print(f"Found {len(datas)} images with font, char and attributes label.")

    return datas


def make_dataset_base_image_attr(base_dir, image_dir, attr_anno_file):
    base_imgs = sorted(make_dataset(base_dir))
    images = {}
    attr_file = None
    assert os.path.isdir(base_dir), f"{base_dir} does not exist"
    for root, _, fnames in sorted(os.walk(base_dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images[os.path.splitext(fname)[0]] = path
            elif fname.lower() == attr_anno_file:
                attr_file = os.path.join(root, fname)

        assert attr_file is not None, f"Failed to find {attr_anno_file}"

    # parse all image
    print("Parsing all images and their attributes...")
    datas = []
    with open(attr_file, "r") as f:
        attr_names = []
        lines = f.readlines()
        attr_names = lines[0].split(" ")
        for idx, line in enumerate(lines[1:]):
            line = line.strip()
            line = line.split("  ")
            fname = os.path.splitext(line[0])[0]
            font = idx // 52
            char = int(fname.split('_')[1].split('.')[0])
            base_img = base_imgs[char]
            attr_vals = [(float(v)/100.0) for v in line[1:]]
            assert len(attr_vals) == len(attr_names), f"{fname} has only {len(attr_vals)} attributes"
            datas.append({
                "base": base_img,
                "image": images[fname],
                "font": font,
                "char": char,
                "attr": attr_vals
            })
    print(f"Found {len(datas)} images with font, char and attributes label.")

    return datas
