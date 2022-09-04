import os

from .dataset import Dataset


class Aquarium(Dataset):
    # def __init__(self, root, train=True, transform=None, benchmark=True):
    def __init__(self, root, train=True, transform=None, benchmark=False):
        self.benchmark = benchmark
        super(Aquarium, self).__init__(root, train, transform)
        print(("Loaded {} samples for dataset {},  {} classes, {} instances".format(len(self), self.name, self.num_cls, self.num_instance)))

    @property
    def name(self):
        return 'aquarium_{}_{}'.format('benchmark' if self.benchmark else 'random', 'train' if self.train else 'val')

    @property
    def image_root_dir(self):
        return self.root

    @property
    def num_cls(self):
        return len(self.class_map)

    @property
    def num_instance(self):
        return len(self.instance_map)

    def _load(self):
        self.instance_map = {} # instance id to instance index
        self.class_map = {} # class name to class index

        class_folders = os.listdir(self.image_root_dir)
        for cls_name in class_folders:
            im_filenames = os.listdir(os.path.join(self.image_root_dir, cls_name))
            if len(im_filenames) == 0:
                continue
            assert(cls_name not in self.class_map)
            if cls_name not in self.class_map:
                self.class_map[cls_name] = len(self.class_map)
            for entry in im_filenames:
                self.class_labels.append(self.class_map[cls_name])
                im_path = os.path.join(self.image_root_dir, cls_name, entry)
                if not im_path.endswith('png'):
                    raise Exception('im_path {} is not a png'.format(im_path))
                self.image_paths.append(im_path)

                # assert(entry not in self.instance_map)
                # self.instance_map[entry] = len(self.instance_map)
                # self.instance_labels.append(self.instance_map[entry])

                if cls_name not in self.instance_map:
                    self.instance_map[cls_name] = len(self.instance_map)
                self.instance_labels.append(self.instance_map[cls_name])

