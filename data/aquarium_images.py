import os.path

from .dataset import Dataset


class AquariumImages(Dataset):
    # def __init__(self, root, train=True, transform=None, benchmark=True):
    def __init__(self, root, train=True, transform=None, benchmark=False):
        self.benchmark = benchmark
        super(AquariumImages, self).__init__(root, train, transform)
        print(("Loaded {} samples for dataset {},  {} instances".format(len(self), self.name, self.num_instance)))

    @property
    def name(self):
        return 'aquarium_images_{}_{}'.format('benchmark' if self.benchmark else 'random', 'train' if self.train else 'val')

    @property
    def image_root_dir(self):
        return self.root

    @property
    def num_instance(self):
        return len(self.instance_map)

    def _load(self):
        self.instance_map = {} # instance id to instance index
        self.class_map = {} # class name to class index
        self.instance_label_to_id = {} # instance index to instance id
        for im_fname in os.listdir(self.image_root_dir):
            im_path = os.path.join(self.image_root_dir, im_fname)
            im_id = os.path.splitext(im_fname)[0]
            self.image_paths.append(im_path)
            self.instance_map[im_id] = len(self.instance_map)
            self.instance_labels.append(self.instance_map[im_id])
            self.instance_label_to_id[self.instance_map[im_id]] = im_id
            self.class_labels.append(0) # no labels

