import xml.etree.ElementTree as ET
import os
import shutil
import argparse
import sys
from collections import Counter
from PIL import Image


class CreateDataset:
        def __init__(self, DETRAC_images, DETRAC_annots, output_train, output_annots, occlusion_ratio, truncation_ratio):
            self.root_images = DETRAC_images
            self.root_annots = DETRAC_annots
            self.output_train = output_train
            self.output_annots = output_annots
            self.occ_thresh = occlusion_ratio
            self.trunc_thresh = truncation_ratio

        def get_sequences(self):
            sequences = [x[1] for x in os.walk(self.root_images)]
            sequences = sequences[0]
            return sequences

        def get_classes(self, sequences):
            '''
            Iterates through every annotation to get used class labels and output to class text file.
            '''
            file = open('detrac_classes.txt', 'w')
            class_list = []
            for sequence in sequences:
                tree = ET.parse(self.root_annots + sequence + '_v3' + '.xml')
                root = tree.getroot()
                frames = root.findall('frame')
                for frame in frames:
                    target_list = frame.find('target_list')
                    for target in target_list:
                        attribute = target.find('attribute')
                        cls = attribute.attrib['vehicle_type']
                        if cls not in class_list:
                            class_list.append(cls)
                            file.write(cls + '\n')
            file.close()
            return class_list

        def cls_occurrences(self, sequences):
            '''
            Calculates the number of training occurrences of each class that satisfy the threshold requirements.
            '''
            cls_count = Counter()
            for sequence in sequences:
                tree = ET.parse(self.root_annots + sequence + '_v3' + '.xml')
                root = tree.getroot()
                frames = root.findall('frame')
                for frame in frames:
                    target_list = frame.find('target_list')
                    targets = target_list.findall('target')
                    for target in targets:
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

                        if occlusion_ratio <= self.occ_thresh and truncation_ratio <= self.trunc_thresh:
                            cls = attribute.attrib['vehicle_type']
                            cls_count[cls] += 1
            count_file = open('class_occurrences.txt', 'w')
            for cls in cls_count:
                count_file.write("{}, {}\n".format(cls, cls_count[cls]))
            count_file.close()
            return cls_count

        def calc_dict(self, frames):
            '''
            Iterates through frames in the parsed XML file to create a dict of frame numbers based
            on the set truncation and occlusion ratio.
            '''
            frame_dict = Counter()
            for frame in frames:
                frame_num = int(frame.attrib['num'])
                target_list = frame.find('target_list')
                targets = target_list.findall('target')
                for target in targets:
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

                    if occlusion_ratio <= self.occ_thresh and truncation_ratio <= self.trunc_thresh:
                        frame_dict[frame_num] += 1
            return dict(frame_dict)

        def recreate_dataset(self, sequences):
            '''
            Recreate dataset by going through each sequence folder and copying + renaming each image file
            to a new directory based on the calculated frame dict.

            Frames that do not contain any vehicles based on the truncation and occlusion ratio will be
            excluded.
            '''
            for sequence in sequences:
                tree = ET.parse(self.root_annots + sequence + '_v3' + '.xml')
                root = tree.getroot()
                frames = root.findall('frame')
                frame_dict = self.calc_dict(frames)
                frame_list = list(frame_dict)
                sequence_folder = self.root_images + sequence
                for img_file in os.listdir(sequence_folder):
                    base = os.path.splitext(img_file)[0]
                    frame_num = int(base[3:].lstrip('0'))
                    if frame_num in frame_list:
                        copy_path = sequence_folder + '/' + img_file
                        rename_img = sequence + '_' + img_file
                        dst_path = self.output_train + rename_img
                        shutil.copy(copy_path, dst_path)

        def generate_annotations(self, class_list, sequences):
            '''
            Create YOLO annotations by iterating through the corresponding annotations for each image in
            the recreated dataset.
            '''
            for sequence in sequences:
                tree = ET.parse(self.root_annots + sequence + '_v3' + '.xml')
                root = tree.getroot()
                frames = root.findall('frame')
                frame_dict = self.calc_dict(frames)
                frame_list = list(frame_dict)

                for frame in frames:
                    frame_num = int(frame.attrib['num'])
                    # Find frame with same number as image and check if it's in the frame dict
                    if frame_num in frame_list:
                        img_name = 'img' + str(frame_num).zfill(5)
                        txt_file_name = sequence + '_' + img_name + '.txt'
                        txt_file = open(self.output_annots + txt_file_name, 'w')

                        img = Image.open(self.root_images + sequence + '/' + img_name+'.jpg')
                        img_width, img_height = img.size

                        target_list = frame.find('target_list')
                        for target in target_list:
                            box = target.find('box')
                            attribute = target.find('attribute')
                            occlusion = target.find('occlusion')

                            cls = attribute.attrib['vehicle_type']

                            cls_index = class_list.index(cls)

                            left = float(box.attrib['left'])
                            top = float(box.attrib['top'])
                            width = float(box.attrib['width'])
                            height = float(box.attrib['height'])

                            yolo_x = (left + width/2) / img_width
                            yolo_y = (top + height/2) / img_height
                            yolo_width = width / img_width
                            yolo_height = height / img_height

                            truncation_ratio = float(attribute.attrib['truncation_ratio'])

                            if occlusion is not None:
                                region_overlap = occlusion.find('region_overlap')
                                overlap_width = round(float(region_overlap.attrib['width']))
                                overlap_height = round(float(region_overlap.attrib['height']))
                                occlusion_ratio = (overlap_width * overlap_height) / (width * height)
                            else:
                                occlusion_ratio = 0

                            if occlusion_ratio <= self.occ_thresh and truncation_ratio <= self.trunc_thresh:
                                txt_file.write("{} {} {} {} {}\n".format(cls_index, yolo_x, yolo_y, yolo_width, yolo_height))

                        txt_file.close()

        def check_train_annots(self):
            annots_list = []
            for annots_file in os.listdir(self.output_annots):
                base = os.path.splitext(annots_file)[0]
                annots_list.append(base)
            for img_file in os.listdir(self.output_train):
                base2 = os.path.splitext(img_file)[0]
                if base2 not in annots_list:
                    print(base2 + ' not found in annotation files.')


def main():
    parser = argparse.ArgumentParser(description="Create YOLO training set with annotations from DETRAC dataset.")
    parser.add_argument("--DETRAC_images",
                        help="Relative location of DETRAC training images.",
                        default="./Insight-MVT_Annotation_Train/")
    parser.add_argument("--DETRAC_annots",
                        help="Relative location of DETRAC annotation files.",
                        default="./DETRAC-Train-Annotations-XML-v3/")
    parser.add_argument("--output_train",
                        help="Relative output location of training images.",
                        default="./DETRAC_YOLO_training/")
    parser.add_argument("--output_annots",
                        help="Relative output location of YOLO annotation files.",
                        default="./DETRAC_YOLO_annotations/")
    parser.add_argument("--occlusion_threshold",
                        help='Ignore images with an occlusion ratio higher than the threshold.',
                        default=0.5, type=float)
    parser.add_argument("--truncation_threshold",
                        help='Ignore images with an truncation ratio higher than the threshold.',
                        default=0.5, type=float)
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

    output_annots = args.output_annots
    if not os.path.exists(output_annots):
        os.makedirs(output_annots)

    if not os.access(output_annots, os.W_OK):
        print('{} folder is not writeable.'.format(output_annots))

    occlusion_threshold = args.occlusion_threshold
    truncation_threshold = args.truncation_threshold

    create_dataset = CreateDataset(DETRAC_images,
                                   DETRAC_annots,
                                   output_train,
                                   output_annots,
                                   occlusion_threshold,
                                   truncation_threshold)

    # Get DETRAC sequences
    sequences = create_dataset.get_sequences()

    # Calculate occurrences for each class.
    print('Calculating number of occurrences for each class...')
    cls_count = dict(create_dataset.cls_occurrences(sequences))
    print('Number of occurrences of each class: \n{}'.format(cls_count))

    # Get class list and also output to text file.
    class_list = create_dataset.get_classes(sequences)
    print('Class list text file created.')

    # Recreate DETRAC dataset for YOLO training.
    print('Copying and recreating the DETRAC dataset...')
    create_dataset.recreate_dataset(sequences)

    # Generate annotations for each image in the recreated dataset.
    print('Converting DETRAC v3 annotations into YOLO format...')
    create_dataset.generate_annotations(class_list, sequences)

    # Check if any annotation files don't match with an image file.
    print('Checking for mismatches...')
    create_dataset.check_train_annots()


if __name__ == '__main__':
    main()
