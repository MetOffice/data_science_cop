"""
Raw data is stored as a series of images and run-length-encoded labels.
This script converts run-length encoded labels to a 2d array that matches the corresponding image. This image is then
queried for rectangles that satisfy a certain criteria to produce a series of images that have a single label
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image

class Label_Images:
    def __init__(self, input_fp, output_fp, labels_fn='train.csv', images_fp='train_images', image_shape=None,
                 labels_suffix=None):
        self.input_fp = input_fp
        self.output_fp = output_fp
        self.labels_fn = labels_fn
        self.images_fp = images_fp
        self.image_shape = image_shape
        self.labels_suffix = labels_suffix

        self.images_fn = os.listdir(os.path.join(input_fp, images_fp))

        self.label_codes = None
        self.labels_rle = None
        self.labels_2d = {}

    def read_labels(self):
        """Read and process labels rle file"""
        labels_rle = pd.read_csv(os.path.join(self.input_fp, self.labels_fn))

        # labels stored with image name. Separate and remove original column
        labels_rle['Image'] = labels_rle['Image_Label'].apply(lambda img_lbl: self.split_img_label(img_lbl)[0])
        labels_rle['Label'] = labels_rle['Image_Label'].apply(lambda img_lbl: self.split_img_label(img_lbl)[1])
        del labels_rle['Image_Label']

        # set label codes
        self.label_codes = {k: v for v, k in enumerate(set(labels_rle['Label']))}

        self.labels_rle = labels_rle

    def split_img_label(self, img_lbl):
        """Return image and label from file name like '0011165.jpg_Flower'"""
        s = img_lbl.split("_")
        assert len(s) == 2
        return s[0], s[1]

    def read_image(self, fn):
        """read image into numpy array"""
        return Image.open(os.path.join(self.input_fp, self.images_fp, fn))

    def rle_decode(self, rle, shape, value=1):
        """
        Decodes an RLE-encoded string.

        Parameters
        ----------
        encoded
            RLE mask.
        shape
            Mask shape in (height, width) format.
        value
            Value to fill in the mask.

        Returns
        -------
        mask
            The decoded mask as 2D image of shape (height, width).
        """

        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.empty(shape[0] * shape[1])
        img[:] = np.nan
        for lo, hi in zip(starts, ends):
            img[lo:hi] = value
        return img.reshape(shape)

    def produce_2d_labels(self):
        """converts rle labels into 2d np arrays of labels"""

        if self.labels_rle is None:
            self.read_labels()

        images = self.labels_rle['Image']

        for image_i in set(images):
            print(f"Decoding {image_i}...")
            shape = self.read_image(fn=image_i).size
            self.labels_2d[image_i] = np.zeros(shape)
            image_l_rle = self.labels_rle[self.labels_rle['Image'] == image_i].copy()
            image_l_rle.dropna(inplace=True)

            for label_i in set(image_l_rle['Label']):
                v = image_l_rle['EncodedPixels'][image_l_rle['Label'] == label_i].values[0]
                self.labels_2d[image_i] += self.rle_decode(v, shape=shape, value=self.label_codes[label_i])

    def sample_rectangles_idx(self, arr, value=np.nan, rectangle_size=(224, 224), num_samples=np.Inf):
        arr = arr.copy()
        if arr.dtype != 'float64':
            Warning('Converting array to float64')
            arr = arr.astype(float)

        sample_rectangles = []
        failed_to_sample = False
        while (~failed_to_sample) & (len(sample_rectangles) < num_samples):
            print(f"Sampling rectangles: {len(sample_rectangles) + 1} of {num_samples}")

            valid_indices = np.argwhere(arr == value)
            np.random.shuffle(valid_indices)

            # iterate through random valid indices until criteria are satisfied, or loop ends
            for idx in valid_indices:
                x0, y0 = idx
                x1, y1 = x0 + rectangle_size[0], y0 + rectangle_size[1]

                sample_arr = arr[x0:x1, y0:y1]

                # if criteria is satisfied save indices, set values to na to sample without replacement,
                # and start next cycle
                if ((sample_arr != value).sum() == 0) & (sample_arr.shape == rectangle_size):
                    sample_rectangles.append((x0, y0))

                    arr[x0: x1,
                        y0: y1] = np.nan

                    break

                # if reach end of loop and haven't found suitable rectangle, end while loop
                if all(idx == valid_indices[-1]):
                    failed_to_sample = True

        return sample_rectangles

    def plot_2d_labels(self, ):
        pass

    def uniquify(self, path):
        """
        return path with suffixed numbers if path already exists
        """
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + " (" + str(counter) + ")" + extension
            counter += 1

        return path

    def extract_labelled_image(self, img_idx_to_load=None):
        """saves images and labels where images are subsets of those provided that satisfy certain criteria
        :param img_idx_to_load: list of images to load. Used for parallelisation. If None, does all
        """
        if self.labels_rle is None:
            self.read_labels()

        images = self.labels_rle['Image']
        labels = []

        if img_idx_to_load is None:
            img_idx_to_load = [x for x in range(len(images.unique()))]

        # remove those outside of range (cases come bash/sbatch scripts)
        img_idx_to_load = [x for x in img_idx_to_load if x < len(images.unique())]

        for image_i in images.unique()[img_idx_to_load]:
            print(f"Analysing {image_i}...")

            img = self.read_image(fn=image_i)

            # labels and rle pixels
            image_l_rle = self.labels_rle[self.labels_rle['Image'] == image_i].copy()
            image_l_rle.dropna(inplace=True)

            for label_i in image_l_rle['Label']:
                # first decode the rle into a 2d array
                v = image_l_rle['EncodedPixels'][image_l_rle['Label'] == label_i].values[0]
                labels_2d = self.rle_decode(v, shape=img.size, value=self.label_codes[label_i])

                # exhaustively search the rle for rectangles of predefined shape that satisfy criteria
                idxs = self.sample_rectangles_idx(arr=labels_2d, value=self.label_codes[label_i], rectangle_size=(224, 224))

                # output images and labels
                for i, idx in enumerate(idxs):
                    cropped_img = img.crop((idx[0], idx[1],
                                            idx[0] + 224, idx[1] + 224))

                    fp = os.path.join(self.output_fp, 'single_labels', '224s')
                    os.makedirs(fp, exist_ok=True)

                    fn = f"{image_i.split('.')[0]}_{i}.jpg"
                    fp_fn = self.uniquify(os.path.join(fp, fn))

                    cropped_img.save(fp_fn)
                    labels.append(pd.DataFrame({
                        "Image": [fp_fn.split('/')[-1]],
                        "Label": [label_i]
                    }))

        labels = pd.concat(labels)
        labels.to_csv(self.uniquify(os.path.join(fp, f"labels_{self.labels_suffix}.csv")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', nargs='?', help='If running in steps, which step this iteration is', type=int)
    parser.add_argument('--step_len', nargs='?', help='If running in steps, length of step', type=int)

    args = parser.parse_args()
    step = args.step
    step_len = args.step_len

    labels_suffix = None  # default is None, change if processing slices so file is not overwritten/parallel written
    if step is not None:
        idxs = [x for x in range(step * step_len, step * step_len + step_len)]
        labels_suffix = f"{str(min(idxs))}-{str(max(idxs))}"

    label_images_class = Label_Images(input_fp="/data/users/meastman/understanding_clouds_kaggle/input",
                                      output_fp="/data/users/meastman/understanding_clouds_kaggle/input",
                                      labels_suffix=labels_suffix)
    label_images_class.extract_labelled_image(idxs)
