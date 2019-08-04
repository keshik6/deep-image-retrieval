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
        """
        Initialize the Query Extractor class
        """
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.identifiers = identifiers
        self.query_list = self.create_query_list()
        self.query_names = []
        if subset == "train":
            self.query_list = self.query_list[:-5]
        else:
            self.query_list = self.query_list[-5:]
        self.query_map = dict()
        self.create_query_maps()

        # Create triplet map
        self.triplet_pairs = []
        self._generate_triplets()


    def create_query_list(self):
        """
        Returns the list of all query txt files
        """
        all_file_list = sorted(os.listdir(self.labels_dir))
        query_list = [file for file in all_file_list if file.endswith('query.txt')]
        return query_list


    def create_query_maps(self):
        """
        This creates a dictionary of dictionary that contains the postive and negative examples for every query
        """
        for query in self.query_list:
            query_image_name = self._get_query_image_name(query)
            tmp = dict()
            good_file, ok_file, junk_file = self._get_query_image_files(query)
            tmp['positive'], tmp['negative'] = self._read_txt_file(good_file), self._read_txt_file(ok_file) + self._read_txt_file(junk_file)
            self.query_map[query_image_name] = tmp
            self.query_names.append(query_image_name)


    def _get_query_image_files(self, query_file):
        """
        This returns the txt file names of good, ok and junk files corresponding to the query file name
        """
        good_file, ok_file, junk_file = query_file.replace('query', self.identifiers[0]), query_file.replace('query', self.identifiers[1]),\
            query_file.replace('query', self.identifiers[2])
        return good_file, ok_file, junk_file


    def _read_txt_file(self, txt_file_name):
        """
        Given a text file, this function returns the lines as a list
        """
        file_path = os.path.join(self.labels_dir, txt_file_name)
        line_list = ["{}.jpg".format(line.rstrip('\n')) for line in open(file_path)]
        return line_list

    
    def _get_all_image_files(self):
        all_file_list = [file for file in os.listdir(self.image_dir)]
        return all_file_list


    def get_query_list(self):
        """
        Returns the query image list. ["all_souls_query_1.txt", "all_souls_query_2.txt"]
        """
        return self.query_list

    
    def get_query_map(self):
        """
        Returns the query map
        """
        return self.query_map


    def _get_query_image_name(self, query_file):
        """
        Given a query file name (all_souls_query_1.txt) returns the actual query image inside the file (all_souls_00001.jpg)
        """
        file_path = os.path.join(self.labels_dir, query_file)
        line_list = ["{}.jpg".format(line.rstrip('\n').split()[0].replace("oxc1_", "")) for line in open(file_path)][0]
        return line_list


    def get_query_names(self):
        """
        Returns the query image name list. ["all_souls_0001.jpg", "all_souls_query_002.jpg"]
        """
        return self.query_names


    def _generate_triplets(self):
        """
        This function generates (anchor, positive), (anchor, negative) pairs for all the queries
        """
        for anchor in self.query_names:
            anchor_positive_pairs = [(anchor, positive) for positive in self.query_map[anchor]['positive']]
            anchor_negative_pairs = [(anchor, negative) for negative in self.query_map[anchor]['negative']]
            
            # Define a low bound and filter based
            low_bound = min(len(anchor_positive_pairs), len(anchor_negative_pairs))

            # Now get the low bound elements to create triplet pairs
            anchor_positive_pairs = anchor_positive_pairs[:low_bound]
            shuffle(anchor_negative_pairs)
            anchor_negative_pairs = anchor_negative_pairs[:low_bound]

            # Create the triplet list and append
            triplet_list = [[anchor_positive_pairs[i], anchor_negative_pairs[i]] for i in range(low_bound)]
            self.triplet_pairs.extend(triplet_list)
        

    def get_triplets(self):
        return self.triplet_pairs


class OxfordDataset(Dataset):

    def __init__(self, labels_dir, image_dir, triplet_pairs, transforms=None):
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.triplet_pairs = triplet_pairs
        self.transforms = transforms
        

    def __getitem__(self, index):
        # Get query image name
        triplet = self.triplet_pairs[index]

        # Assert here
        assert(triplet[0][0] == triplet[1][0])
        anchor_img_name, positive_img_name, negative_img_name = triplet[0][0], triplet[0][1], triplet[1][1]

        # Get image paths
        anchor_path = os.path.join(self.image_dir, anchor_img_name)
        positive_path = os.path.join(self.image_dir, positive_img_name)
        negative_path = os.path.join(self.image_dir, negative_img_name)

        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        # Get mean and std for every image
        anchor_mean, anchor_std = np.mean(np.asarray(anchor_img)/255.0, axis=(0, 1)), np.std(np.asarray(anchor_img)/255.0, axis=(0, 1))
        positive_mean, positive_std = np.mean(np.asarray(positive_img)/255.0, axis=(0, 1)), np.std(np.asarray(positive_img)/255.0, axis=(0, 1))
        negative_mean, negative_std = np.mean(np.asarray(negative_img)/255.0, axis=(0, 1)), np.std(np.asarray(negative_img)/255.0, axis=(0, 1))
        
        # Transform the images
        if self.transforms is not None:
            anchor_img = transforms.functional.normalize(self.transforms(anchor_img), mean=anchor_mean, std=anchor_std, inplace=False)
            positive_img = transforms.functional.normalize(self.transforms(positive_img), mean=positive_mean, std=positive_std, inplace=False)
            neg_img = transforms.functional.normalize(self.transforms(negative_img), mean=negative_mean, std=negative_std, inplace=False)
            return anchor_img, positive_img, neg_img
        
        return self._normalize_np_array(anchor_img, anchor_mean, anchor_std),\
            self._normalize_np_array(positive_img, positive_mean, positive_std),\
            self._normalize_np_array(negative_img, negative_mean, negative_std)
    

    def __len__(self):
        return len(self.triplet_pairs)

    
    def _normalize_np_array(self, img_pil, mean, std):
        img = np.asarray(img_pil)/255.0
        return (img-mean)/std



class EmbeddingDataset(Dataset):
    def __init__(self, image_dir, query_img_list, transforms):
        if transforms == None:
            raise
        
        self.image_dir = image_dir
        self.transforms = transforms
        self.filenames = query_img_list
    

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert('RGB')
        mean, std = np.mean(np.asarray(image)/255.0, axis=(0, 1)), np.std(np.asarray(image)/255, axis=(0, 1))
        image = transforms.functional.normalize(self.transforms(image), mean=mean, std=std, inplace=False)
        return image
        

    def __len__(self):
        return len(self.filenames)


    def get_filenames(self):
        return self.filenames


# # Define directories
# labels_dir, image_dir = "./data/oxbuild/gt_files/", "./data/oxbuild/images/"

# # Create Query extractor object
# q = QueryExtractor(labels_dir, image_dir, "valid")

# # Get query list and query map
# triplets = q.get_triplets()

# from torchvision import transforms
# import torch
# transforms_test = transforms.Compose([transforms.Resize(460),
#                                     transforms.RandomResizedCrop(448),
#                                     transforms.ToTensor(),
#                                     #transforms.Normalize(mean=mean, std=std),                                 
#                                     ])
# # Create dataset
# ox = OxfordDataset(labels_dir, image_dir, triplets, transforms_test)
# a, p, n = ox.__getitem__(100)
# plt.imshow(a.numpy().transpose(1, 2, 0))
# plt.show()

# plt.imshow(p.numpy().transpose(1, 2, 0))
# plt.show()

# plt.imshow(n.numpy().transpose(1, 2, 0))
# plt.show()










