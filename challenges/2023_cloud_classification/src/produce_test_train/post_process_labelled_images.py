"""
Takes output from extract_labelled_images.sbatch and separates into test/train images, and produces test/train labels
"""

import os
import glob
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

class PPLabelledImages:

    def __init__(self, parent_dir):
        self.parent_dir = parent_dir

        self.fps = None
        self.labels = None
        self.image_names = None
        self.train_images = None
        self.test_images = None

    def extract_labels(self):
        """Read label files"""
        self.fps = glob.glob(os.path.join(self.parent_dir, '*.csv'))

        dfs = [pd.read_csv(fp, index_col=0) for fp in self.fps]
        self.labels = pd.concat(dfs, ignore_index=True)

    def move_test_train_files(self):
        if self.train_images is None or self.test_images is None:
            ValueError("No images in self.train_images or self.test_images to move")
            return

        os.makedirs(os.path.join(self.parent_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.parent_dir, 'test'), exist_ok=True)

        train_fps = [glob.glob(os.path.join(self.parent_dir, img + '*')) for img in self.train_images]
        test_fps = [glob.glob(os.path.join(self.parent_dir, img + '*')) for img in self.test_images]

        # join lists together
        train_fps = sum(train_fps, [])
        test_fps = sum(test_fps, [])

        # move files
        for fp in train_fps:
            new_fp = os.path.join(self.parent_dir, 'train', fp.split('/')[-1])
            shutil.move(fp, new_fp)
        for fp in test_fps:
            new_fp = os.path.join(self.parent_dir, 'test', fp.split('/')[-1])
            shutil.move(fp, new_fp)

    def delete_files(self):
        pass

    def process_images(self):
        # read all label files into pd.DataFrame
        self.extract_labels()

        # separate entire images into test/train
        image_names = [img.split('.')[0].split('_')[0] for img in self.labels['Image']]
        self.image_names = list(set(image_names))

        self.train_images, self.test_images = train_test_split(self.image_names, test_size=.2, random_state=11)

        # move files into new directories with train/test specific label files
        self.move_test_train_files()
        train_labels = self.labels[[x in self.train_images for x in image_names]]
        test_labels = self.labels[[x in self.test_images for x in image_names]]

        train_labels.to_csv(os.path.join(self.parent_dir, 'train', 'train_labels.csv'))
        test_labels.to_csv(os.path.join(self.parent_dir, 'test', 'test_labels.csv'))

        # delete old label files
        for fn in glob.glob(os.path.join(self.parent_dir, '*.csv')):
            os.remove(fn)


if __name__ == '__main__':
    lab_img_processor = PPLabelledImages(
        parent_dir='/data/users/meastman/understanding_clouds_kaggle/input/single_labels/224s')
    lab_img_processor.process_images()
