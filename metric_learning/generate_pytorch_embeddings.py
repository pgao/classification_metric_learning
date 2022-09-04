import aquariumlearning as al
import os
import pickle
import sys
import time
import tqdm
import PIL.Image
from itertools import chain

import torch
from pretrainedmodels.utils import ToRange255
from pretrainedmodels.utils import ToSpaceBGR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from data.aquarium_images import AquariumImages

from metric_learning.util import SimpleLogger
from metric_learning.sampler import ClassBalancedBatchSampler

import metric_learning.modules.featurizer as featurizer
import metric_learning.modules.losses as losses

from extract_features import extract_feature

def generate_embeddings_on_images(image_input_path, image_ids, model, transform, gpu_device):
    model.eval()

    dataset = AquariumImages(
        image_input_path,
        train=False,
        transform=transform
    )
    loader = DataLoader(dataset,
                        batch_size=256,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=4)

    log_every_n_step = 10
    emb_map = {}
    with torch.no_grad():
        for i, (im, class_label, instance_label, index) in enumerate(loader):
            im = im.to(device=gpu_device)
            embedding = model(im)

            instance_label_list = instance_label.cpu().tolist()
            emb_list = embedding.cpu().tolist()
            for inst_label, emb in zip(instance_label_list, emb_list):
                im_id = dataset.instance_label_to_id[inst_label]
                emb_map[im_id] = emb

            if (i + 1) % log_every_n_step == 0:
                print(('Process Iteration {} / {}:'.format(i, len(loader))))

    for im_id in image_ids:
        if im_id not in emb_map:
            print('image id {} is missing'.format(im_id))
            emb_map[im_id] = [-100.0 for _ in range(2048)]

    return emb_map


def main():
    # AL_PROJECT = 'demo_kitti_det_2d'
    # AL_DATASET = 'labeled'

    AL_PROJECT = 'Rareplanes_Wingtype_Project'
    AL_DATASET = 'initial_train_labels'

    hackathon_path = '/home/pgao/aquarium_hackathon_2022/'
    model_path = '/home/pgao/classification_metric_learning/model_outputs_aquarium_sgd_classes_as_instances_rareplanes/Aquarium/2048/resnet50_16/epoch_30.pth'
    images_path = os.path.join(hackathon_path, 'data/images')
    label_crops_path = os.path.join(hackathon_path, 'data/crop_cache')

    torch.cuda.set_device(0)
    gpu_device = torch.device('cuda')

    # Setup model
    model_factory = getattr(featurizer, 'resnet50')
    model = model_factory(2048)
    model.load_state_dict(torch.load(model_path))
    model.to(device=gpu_device)
    model.eval()

    # Setup eval transformations
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(max(model.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(model.input_space == 'BGR'),
        ToRange255(max(model.input_range) == 255),
        transforms.Normalize(mean=model.mean, std=model.std)
    ])

    al_client = al.Client()
    al_client.set_credentials(api_key=os.getenv('AQUARIUM_KEY'))
    al_project = al_client.get_project(AL_PROJECT)
    class_names = [cl['category'] for cl in al_project['label_class_map'] if not cl['ignore_in_eval']]

    print("getting frame ids")
    frame_ids = al_client.get_frame_ids(AL_PROJECT, AL_DATASET)
    # frame_ids = frame_ids[:100]

    print("getting frames")
    frames = al_client.get_frames(AL_PROJECT, AL_DATASET, frame_ids)

    print('generating frame embeddings')
    frame_emb_map = generate_embeddings_on_images(images_path, frame_ids, model, eval_transform, gpu_device)
    print('pickling frame embeddings')
    with open(os.path.join(hackathon_path, "frame_embeddings.pickle"), "wb") as f:
        pickle.dump(frame_emb_map, f)

    print('generating label crop embeddings')
    label_crop_ids = []
    for frame_id, f in frames.items():
        for label_data in f._labels.label_data:
            label_crop_ids.append(label_data['uuid'])

    label_crop_emb_map = generate_embeddings_on_images(label_crops_path, label_crop_ids, model, eval_transform, gpu_device)
    print('pickling crop embeddings')
    with open(os.path.join(hackathon_path, "label_crop_embeddings.pickle"), "wb") as f:
        pickle.dump(label_crop_emb_map, f)


if __name__ == '__main__':
    main()
