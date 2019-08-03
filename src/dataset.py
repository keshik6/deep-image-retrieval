import os, glob
import numpy as np
from random import shuffle
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import itertools


class QueryExtractor():

    def __init__(self, labels_dir, image_dir, subset, identifiers=['good', 'ok', 'junk']):
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.identifiers = identifiers
        self.query_list = self.create_query_list()
        self.query_names = []
        if subset == "train":
            self.query_list = self.query_list[:-15]
        else:
             self.query_list = self.query_list[-15:]
        self.query_map = dict()
        self.create_query_maps()


    def create_query_list(self):
        all_file_list = os.listdir(self.labels_dir)
        query_list = [file for file in all_file_list if file.endswith('query.txt')]
        return query_list


    def create_query_maps(self):
        for query in self.query_list:
            query_image_name = self._get_query_image_name(query)
            tmp = dict()
            good_file, ok_file, junk_file = self._get_query_image_files(query)
            tmp['good'], tmp['ok'], tmp['junk'] = self._read_txt_file(good_file), self._read_txt_file(ok_file), self._read_txt_file(junk_file)
            tmp_set = set(tmp['good'] + tmp['ok'] + tmp['junk'])
            tmp['bad'] = self._get_bad_image_files(tmp_set)
            self.query_map[query_image_name] = tmp
            self.query_names.append(query_image_name)


    def _get_bad_image_files(self, tmp_set):
        all_set = set(self._get_all_image_files())
        bad_list = list(all_set - tmp_set)
        return bad_list


    def _get_query_image_files(self, query_file):
        good_file, ok_file, junk_file = query_file.replace('query', self.identifiers[0]), query_file.replace('query', self.identifiers[1]),\
            query_file.replace('query', self.identifiers[2])
        return good_file, ok_file, junk_file


    def _read_txt_file(self, txt_file_name):
        file_path = os.path.join(self.labels_dir, txt_file_name)
        line_list = ["{}.jpg".format(line.rstrip('\n')) for line in open(file_path)]
        return line_list

    
    def _get_all_image_files(self):
        all_file_list = [file for file in os.listdir(self.image_dir)]
        return all_file_list


    def get_query_list(self):
        return self.query_list

    
    def get_query_map(self):
        return self.query_map


    def _get_query_image_name(self, query_file):
        file_path = os.path.join(self.labels_dir, query_file)
        line_list = ["{}.jpg".format(line.rstrip('\n').split()[0].replace("oxc1_", "")) for line in open(file_path)][0]
        return line_list

    def get_query_names(self):
        return self.query_names


class OxfordDataset(Dataset):

    def __init__(self, labels_dir, image_dir, query_names, query_map, transforms=None):
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.query_names = query_names
        self.query_map = query_map
        self.transforms = transforms
        self.query_pairs, self.pair_weights = dict(), dict()
        self.duplicated_query_names = query_names*10
        self.bad_thres, self.junk_thres, self.ok_thres = 2.0, 0.5, 0.1
        self.junk_idx, self.ok_idx = [], []
        self._generate_pairs_and_weights()
    

    def _generate_pairs_and_weights(self):
        for query_img in self.query_names:
            pairs, sampler_weights = [], []
            gb_pairs = list(itertools.product(self.query_map[query_img]['good'], self.query_map[query_img]['bad']))
            gj_pairs = list(itertools.product(self.query_map[query_img]['good'], self.query_map[query_img]['junk']))
            go_pairs = list(itertools.product(self.query_map[query_img]['good'], self.query_map[query_img]['ok']))

            # Update index pointers for junk and ok categories
            self.junk_idx.append(len(gb_pairs))
            self.ok_idx.append(len(gb_pairs) + len(gj_pairs))

            # Create sampler weights
            sampler_weights.extend([self.bad_thres]*len(gb_pairs))
            sampler_weights.extend([self.junk_thres]*len(gj_pairs))
            sampler_weights.extend([self.ok_thres]*len(go_pairs))
            sampler_weights = np.asarray(sampler_weights)

            # Create pairs
            pairs.extend(gb_pairs)
            pairs.extend(gj_pairs)
            pairs.extend(go_pairs)

            # Update the dictionaries
            self.query_pairs[query_img] = pairs
            self.pair_weights[query_img] = sampler_weights/sampler_weights.sum()
            

    def increase_difficulty(self):
        print("Existing difficulty -> ok_thres={}, junk_thres={}, bad_thres={}".format(self.ok_thres, self.junk_thres, self.bad_thres))
        
        self.junk_thres += 0.5
        self.ok_thres += 0.3

        for idx, query_img in enumerate(self.query_names):
            # Create a copy of the weights for pairs
            weights = np.copy(self.pair_weights[query_img])

            # Update the sampler weights
            weights[self.junk_idx[idx]:self.ok_idx[idx]] = self.junk_thres
            weights[self.ok_idx[idx]:] = self.ok_thres

            # Normalize for probability
            self.pair_weights[query_img] = weights/weights.sum()

        print("Difficulty increased successfully -> ok_thres={}, junk_thres={}, bad_thres={}".format(self.ok_thres, self.junk_thres, self.bad_thres))


    def __getitem__(self, index):
        # Get query image name
        query_image_name = self.duplicated_query_names[index]
        #query_image_name = self._get_query_image_name(query_text_file)

        # Get image paths
        anchor_path = os.path.join(self.image_dir, query_image_name)
        idx = np.random.choice(len(self.query_pairs[query_image_name]), p=self.pair_weights[query_image_name])
        positive_image, negative_image = self.query_pairs[query_image_name][idx]

        positive_path = os.path.join(self.image_dir, positive_image)
        negative_path = os.path.join(self.image_dir, negative_image)
        
        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        neg_img = Image.open(negative_path).convert('RGB')

        # Transform the images
        if self.transforms is not None:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            neg_img = self.transforms(neg_img)
            return anchor_img, positive_img, neg_img
        
        return anchor_img, positive_img, neg_img


    # def _get_query_image_name(self, query_file):
    #     file_path = os.path.join(self.labels_dir, query_file)
    #     line_list = ["{}.jpg".format(line.rstrip('\n').split()[0].replace("oxc1_", "")) for line in open(file_path)][0]
    #     return line_list

    
    def __len__(self):
        return len(self.duplicated_query_names)



class EvalDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.filenames = os.listdir(image_dir)
    

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.filenames[index])
        image = Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)
            return image, self.filenames[index]
        
        return image, self.filenames[index]


    def __len__(self):
        return len(self.filenames)

# # Define directories
# labels_dir, image_dir = "./data/oxbuild/gt_files/", "./data/oxbuild/images/"

# # Create Query extractor object
# q = QueryExtractor(labels_dir, image_dir, "train")

# # Get query list and query map
# query_list, query_map = q.get_query_names(), q.get_query_map()
# print(query_list)

# # Create dataset
# ox = OxfordDataset(labels_dir, image_dir, query_list, query_map)
# a, p, n = ox.__getitem__(0)
# plt.imshow(a)
# plt.show()

# plt.imshow(p)
# plt.show()

# plt.imshow(n)
# plt.show()










