import os, glob
import numpy as np
from random import shuffle


class QueryExtractor():

    def __init__(self, labels_dir, image_dir, identifiers=['good', 'ok', 'junk']):
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.identifiers = identifiers
        self.query_list = self.create_query_list()
        self.query_map = dict()
        self.create_query_maps()


    def create_query_list(self):
        all_file_list = os.listdir(self.labels_dir)
        query_list = [file for file in all_file_list if file.endswith('query.txt')]
        return query_list


    def create_query_maps(self):
        for query in self.query_list:
            tmp = dict()
            good_file, ok_file, junk_file = self._get_query_image_files(query)
            tmp['good'], tmp['ok'], tmp['junk'] = self._read_txt_file(good_file), self._read_txt_file(ok_file), self._read_txt_file(junk_file)
            tmp_set = set(tmp['good'] + tmp['ok'] + tmp['junk'])
            tmp['bad'] = self._get_bad_image_files(tmp_set)
            self.query_map[query] = tmp
        

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



q = QueryExtractor("./data/oxbuild/gt_files/", "./data/oxbuild/images/")
#print(q.get_query_map())