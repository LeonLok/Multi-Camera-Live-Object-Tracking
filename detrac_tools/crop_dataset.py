import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import os
import random
import argparse


class CreateDataset:

        def __init__(self,
                     DETRAC_images,
                     DETRAC_annots,
                     output_train,
                     occlusion_threshold,
                     truncation_threshold,
                     occurrences):
            self.root_images = DETRAC_images
            self.root_annots = DETRAC_annots
            self.output_folder = output_train
            self.occ_thresh = occlusion_threshold
            self.trunc_thresh = truncation_threshold
            self.no_of_occurrences = occurrences
            self.resize = (100, 100)

        def get_sequences(self):
            sequences = [x[1] for x in os.walk(self.root_images)]
            sequences = sequences[0]
            return sequences

        def calc_dict(self, frames):
            target_id_dict = {}
            for frame in frames:
                frame_num = int(frame.attrib['num'])
                target_list = frame.find('target_list')
                targets = target_list.findall('target')
                for target in targets:
                    target_id = target.attrib['id']
                    attribute = target.find('attribute')
                    occlusion = target.find('occlusion')

                    box = target.find('box')
                    width = round(float(box.attrib['width']))
                    height = round(float(box.attrib['height']))

                    truncation_ratio = float(attribute.attrib['truncation_ratio'])

                    if occlusion is not None:
                        region_overlap = occlusion.find('region_overlap')
                        overlap_width = round(float(region_overlap.attrib['width']))
                        overlap_height = round(float(region_overlap.attrib['height']))
                        occlusion_ratio = (overlap_width * overlap_height) / (width * height)
                    else:
                        occlusion_ratio = 0

                    if target_id not in list(target_id_dict):
                        target_id_dict[target_id] = []

                    if occlusion_ratio < self.occ_thresh and truncation_ratio < self.trunc_thresh:
                        target_id_dict[target_id].append(frame_num)

            for target_id in list(target_id_dict):
                no_of_occurrences = len(target_id_dict[target_id])
                if no_of_occurrences >= self.no_of_occurrences:
                    min_frame = min(target_id_dict[target_id])
                    max_frame = max(target_id_dict[target_id])
                    sample = random.sample(range(min_frame, max_frame), min(self.no_of_occurrences, len(range(min_frame, max_frame))))
                    target_id_dict[target_id] = sample

                elif no_of_occurrences < self.no_of_occurrences:
                    target_id_dict.pop(target_id)
            return target_id_dict

        def crop_sequence_images(self, sequences):
            max_target_id = 0
            for sequence in sequences:
                tree = ET.parse(self.root_annots + sequence + '_v3.xml')
                root = tree.getroot()
                frames = root.findall('frame')
                target_id_dict = self.calc_dict(frames)
                target_id_list = list(target_id_dict)

                for frame in frames:
                    target_list = frame.find('target_list')
                    targets = target_list.findall('target')
                    frame_num = int(frame.attrib['num'])
                    for target in targets:
                        box = target.find('box')
                        target_id = target.attrib['id']
                        if target_id in target_id_list:
                            frame_list = target_id_dict[target_id]
                            if frame_num in frame_list:
                                left = round(float(box.attrib['left']))
                                top = round(float(box.attrib['top']))
                                width = round(float(box.attrib['width']))
                                height = round(float(box.attrib['height']))

                                right = left + width
                                bottom = top + height

                                image_frame = "img" + str(frame_num).zfill(5) + '.jpg'

                                rectangle = (left, top, right, bottom)

                                # vehicle_type = attribute.attrib['vehicle_type']
                                # truncation_ratio = attribute.attrib['truncation_ratio']
                                image = Image.open(self.root_images + sequence + '/' + image_frame)
                                image = image.crop(rectangle)
                                image = image.resize(self.resize)
                                market1501_id = str(int(target_id) + int(max_target_id)).zfill(5) + '_c' + sequence[-5:] \
                                                + 's1_' + str(frame_num).zfill(5) + '_01'
                                image.save(self.output_folder + market1501_id + '.jpg')

                max_target_id += int(target_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create cropped sequences of vehicles from DETRAC dataset.")
    parser.add_argument("--DETRAC_images",
                        help="Relative location of DETRAC training images.",
                        default="./Insight-MVT_Annotation_Train/")
    parser.add_argument("--DETRAC_annots",
                        help="Relative location of DETRAC annotation files.",
                        default="./DETRAC-Train-Annotations-XML-v3/")
    parser.add_argument("--output_train",
                        help="Relative output location of cropped training images.",
                        default="./DETRAC_cropped/")
    parser.add_argument("--occlusion_threshold",
                        help='Ignore images with an occlusion ratio higher than the threshold.',
                        default=0.5, type=float)
    parser.add_argument("--truncation_threshold",
                        help='Ignore images with an truncation ratio higher than the threshold.',
                        default=0.5, type=float)
    parser.add_argument("--occurrences",
                        help='Number of occurrences of each sequence of vehicles.',
                        default=100, type=int)
    args = parser.parse_args()
    
    DETRAC_images = args.DETRAC_images
    if not os.path.exists(DETRAC_images):
        print('Cannot find path to DETRAC images.')
        sys.exit()

    DETRAC_annots = args.DETRAC_annots
    if not os.path.exists(DETRAC_annots):
        print('Cannot find path to DETRAC annotations.')
        sys.exit()

    output_train = args.output_train
    if not os.path.exists(output_train):
        os.makedirs(output_train)

    if not os.access(output_train, os.W_OK):
        print('{} folder is not writeable.'.format(output_train))

    occlusion_threshold = args.occlusion_threshold
    truncation_threshold = args.truncation_threshold
    occurrences = args.occurrences

    create_dataset = CreateDataset(DETRAC_images,
                                   DETRAC_annots,
                                   output_train,
                                   occlusion_threshold,
                                   truncation_threshold,
                                   occurrences)

    sequences = create_dataset.get_sequences()

    create_dataset.crop_sequence_images(sequences)

