import bz2
import csv
import json
import operator
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.base_dataset import BaseDataset



'''
define dataset for RB_SR
'''


## MusicalInstruments (version2)
class AmazonMusicalInstrumentsDataset_review(BaseDataset):
    
    def __init__(self, input_path, output_path):
        super(AmazonMusicalInstrumentsDataset_review, self).__init__(input_path, output_path)
        self.dataset_name = 'MI'

        # input file
        self.inter_file = os.path.join('./Musical_Instruments_5.json')
        self.item_file = os.path.join('./meta_Musical_Instruments.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # (modify) selected feature fields : add review
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             4: 'review:token_seq',
                             5: 'rating:float',
                             7: 'timestamp:float'}

        # (modify) : only features to use
        self.item_fields = {0: 'item_id:token',
                            6: 'categories:token_seq',
                            8: 'brand:token',
                            -1 : 'attributes:token_seq'}

    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        # modify
        inter_data = pd.read_json(self.inter_file, lines=True) 

        # review filtering
        ind = inter_data[inter_data['reviewText'] == ""].index
        print(f'review filtering -  drop len:{len(ind)}, drop index: {ind.values}')
        inter_data.drop(ind, inplace=True)
        
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        # modify
        all_categories_set = set()
        
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 4]
            categories = origin_data.iloc[i, 5]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):

                    # modify : prevent duplication with sep in recbole
                    category = categories[j][k].replace(',', '.')  
                    categories_set.add(category)
                    all_categories_set.add(category)
                    
            new_categories.append(str(categories_set)[1:-1])
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
                    
        finished_data.insert(4, 'sales_type', pd.Series(sales_type))
        finished_data.insert(5, 'sales_rank', pd.Series(sales_rank))
        finished_data.insert(6, 'categories', pd.Series(new_categories))

        # modify
        finished_data['brand'] = finished_data['brand'].fillna("").apply(lambda x : str(x).replace("nan", ""))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace("&#39", "'"))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace(",", ""))

        # get attributes (for S3Rec)
        finished_data['attributes'] =\
          finished_data['categories'] + finished_data['brand'].apply(lambda x : (f", '{x}'") if x !='' else '')

        print(f'brand nunique : {finished_data["brand"].nunique()-1}  ')  # searching without NaN & '' 
        print(f'categories nunique : {len(all_categories_set)}')


        # check num attributes
        attributes_set = set()
        for i in finished_data['attributes'] :
            li = i.split(', ')
            for x in li :
                attributes_set.add(x)
        
        print(f'attribute nunique : {len(attributes_set)}')  # brand and categories can be duplicated : (ex) "Disney" 
        
        return finished_data
        

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
                        # column(index) > column name
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    # method
    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            
            
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    # method
    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')


# OfficeProducts(version 2)
class AmazonOfficeProductsDataset_review(BaseDataset):
    def __init__(self, input_path, output_path):
        super(AmazonOfficeProductsDataset_review, self).__init__(input_path, output_path)
        self.dataset_name = 'OP'

        # input file
        self.inter_file = os.path.join('./Office_Products_5.json')
        self.item_file = os.path.join( './meta_Office_Products.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()


        # (modify) selected feature fields : add review
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             4: 'review:token_seq',
                             5: 'rating:float',
                             7: 'timestamp:float'}


        # (modify) : only features to use, <note> column order varies by dataset
        self.item_fields = {0: 'item_id:token',
                            7: 'categories:token_seq',
                            9: 'brand:token',
                            -1: 'attributes:token_seq'}
        
    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        inter_data = pd.read_json(self.inter_file, lines=True)     
        # review filtering
        ind = inter_data[inter_data['reviewText'] == ""].index
        print(f'review filtering -  drop len:{len(ind)}, drop index: {ind.values}')
        inter_data.drop(ind, inplace=True)        
        
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        
        all_categories_set = set()
        
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 5]
            categories = origin_data.iloc[i, 6]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):

                    # modify : prevent duplication with sep in recbole
                    category = categories[j][k].replace(',', '.')  
                    categories_set.add(category)
                    all_categories_set.add(category)

            new_categories.append(str(categories_set)[1:-1])
            
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
        finished_data.insert(5, 'sales_type', pd.Series(sales_type))
        finished_data.insert(6, 'sales_rank', pd.Series(sales_rank))
        finished_data.insert(7, 'categories', pd.Series(new_categories))

        # modify
        finished_data['brand'] = finished_data['brand'].fillna("").apply(lambda x : str(x).replace("nan", ""))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace("&#39", "'"))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace(",", ""))

        # get attributes (for S3Rec)
        finished_data['attributes'] =\
          finished_data['categories'] + finished_data['brand'].apply(lambda x : (f", '{x}'") if x !='' else '')

        print(f'brand nunique : {finished_data["brand"].nunique()-1}  ')  # searching without NaN & '' 
        print(f'categories nunique : {len(all_categories_set)}')


        # check num attributes
        attributes_set = set()
        for i in finished_data['attributes'] :
            li = i.split(', ')
            for x in li :
                attributes_set.add(x)
        
        print(f'attribute nunique : {len(attributes_set)}')  # brand and categories can be duplicated : (ex) "Disney" 

        return finished_data

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')


# toy(version 2)
class AmazonToysAndGamesDataset_review(BaseDataset):
    def __init__(self, input_path, output_path):
        super(AmazonToysAndGamesDataset_review, self).__init__(input_path, output_path)
        self.dataset_name = 'TOY'

        # input file
        self.inter_file = os.path.join('./Toys_and_Games_5.json')
        self.item_file = os.path.join( './meta_Toys_and_Games.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # (modify) selected feature fields : add review
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             4: 'review:token_seq',
                             5: 'rating:float',
                             7: 'timestamp:float'}


        # (modify) : only features to use, <note> column order varies by dataset
        self.item_fields = {0: 'item_id:token',
                            8: 'categories:token_seq',
                            7: 'brand:token',
                            -1: 'attributes:token_seq'}
        
    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        inter_data = pd.read_json(self.inter_file, lines=True)     
        # review filtering
        ind = inter_data[inter_data['reviewText'] == ""].index
        print(f'review filtering -  drop len:{len(ind)}, drop index: {ind.values}')
        inter_data.drop(ind, inplace=True)        
        
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        
        all_categories_set = set()
        
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 4]
            categories = origin_data.iloc[i, 7]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):

                    # modify : prevent duplication with sep in recbole
                    category = categories[j][k].replace(',', '.')  
                    categories_set.add(category)
                    all_categories_set.add(category)

            new_categories.append(str(categories_set)[1:-1])
            
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
        finished_data.insert(4, 'sales_type', pd.Series(sales_type))
        finished_data.insert(5, 'sales_rank', pd.Series(sales_rank))
        finished_data.insert(8, 'categories', pd.Series(new_categories))

        # modify
        finished_data['brand'] = finished_data['brand'].fillna("").apply(lambda x : str(x).replace("nan", ""))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace("&#39", "'"))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace(",", ""))

        # get attributes (for S3Rec)
        finished_data['attributes'] =\
          finished_data['categories'] + finished_data['brand'].apply(lambda x : (f", '{x}'") if x !='' else '')

        print(f'brand nunique : {finished_data["brand"].nunique()-1}  ')  # searching without NaN & '' 
        print(f'categories nunique : {len(all_categories_set)}')


        # check num attributes
        attributes_set = set()
        for i in finished_data['attributes'] :
            li = i.split(', ')
            for x in li :
                attributes_set.add(x)
        
        print(f'attribute nunique : {len(attributes_set)}')  # brand and categories can be duplicated : (ex) "Disney" 

        return finished_data

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')


class AmazonBeautyDataset_review(BaseDataset):
    def __init__(self, input_path, output_path):
        super(AmazonBeautyDataset_review, self).__init__(input_path, output_path)
        self.dataset_name = 'BU'

        # input file
        self.inter_file = os.path.join('./Beauty_5.json')
        self.item_file = os.path.join( './meta_Beauty.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # (modify) selected feature fields : add review
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             4: 'review:token_seq',
                             5: 'rating:float',
                             7: 'timestamp:float'}

        """
        self.item_fields = {0: 'item_id:token',
                            2: 'title:token',
                            4: 'sales_type:token',
                            5: 'sales_rank:float',
                            6: 'categories:token_seq',
                            7: 'price:float',
                            9: 'brand:token'}
        """

        
        # (modify) : only features to use, <note> column order varies by dataset
        self.item_fields = {0: 'item_id:token',
                            6: 'categories:token_seq',
                            9: 'brand:token',
                            -1: 'attributes:token_seq'}
        
    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        inter_data = pd.read_json(self.inter_file, lines=True)     
        # review filtering
        ind = inter_data[inter_data['reviewText'] == ""].index
        print(f'review filtering -  drop len:{len(ind)}, drop index: {ind.values}')
        inter_data.drop(ind, inplace=True)        
        
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        
        all_categories_set = set()
        
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 4]
            categories = origin_data.iloc[i, 5]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):

                    # modify : prevent duplication with sep in recbole
                    category = categories[j][k].replace(',', '.')  
                    categories_set.add(category)
                    all_categories_set.add(category)

            new_categories.append(str(categories_set)[1:-1])
            
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
        finished_data.insert(4, 'sales_type', pd.Series(sales_type))
        finished_data.insert(5, 'sales_rank', pd.Series(sales_rank))
        finished_data.insert(6, 'categories', pd.Series(new_categories))

        # modify
        finished_data['brand'] = finished_data['brand'].fillna("").apply(lambda x : str(x).replace("nan", ""))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace("&#39", "'"))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace(",", ""))

        # get attributes (for S3Rec)
        finished_data['attributes'] =\
          finished_data['categories'] + finished_data['brand'].apply(lambda x : (f", '{x}'") if x !='' else '')

        print(f'brand nunique : {finished_data["brand"].nunique()-1}  ')  # searching without NaN & '' 
        print(f'categories nunique : {len(all_categories_set)}')


        # check num attributes
        attributes_set = set()
        for i in finished_data['attributes'] :
            li = i.split(', ')
            for x in li :
                attributes_set.add(x)
        
        print(f'attribute nunique : {len(attributes_set)}')  # brand and categories can be duplicated : (ex) "Disney" 

        return finished_data

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')


class AmazonDigitalMusicDataset_review(BaseDataset):
    def __init__(self, input_path, output_path):
        super(AmazonDigitalMusicDataset_review, self).__init__(input_path, output_path)
        self.dataset_name = 'DM'

        # input file
        self.inter_file = os.path.join('./Digital_Music_5.json')
        self.item_file = os.path.join( './meta_Digital_Music.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # (modify) selected feature fields : add review
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             4: 'review:token_seq',
                             5: 'rating:float',
                             7: 'timestamp:float'}

        """
        self.item_fields = {0: 'item_id:token',
                            1: 'title:token',
                            2: 'price:float',
                            5: 'sales_type:token',
                            6: 'sales_rank:float',
                            7: 'categories:token_seq',
                            9: 'brand:token'}
        """

        
        # (modify) : only features to use, <note> column order varies by dataset
        self.item_fields = {0: 'item_id:token',
                            7: 'categories:token_seq',
                            9: 'brand:token',
                            -1: 'attributes:token_seq'}
        
    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        inter_data = pd.read_json(self.inter_file, lines=True)     
        # review filtering
        ind = inter_data[inter_data['reviewText'] == ""].index
        print(f'review filtering -  drop len:{len(ind)}, drop index: {ind.values}')
        inter_data.drop(ind, inplace=True)        
        
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        
        all_categories_set = set()
        
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 5]
            categories = origin_data.iloc[i, 6]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):

                    # modify : prevent duplication with sep in recbole
                    category = categories[j][k].replace(',', '.')  
                    categories_set.add(category)
                    all_categories_set.add(category)

            new_categories.append(str(categories_set)[1:-1])
            
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
        finished_data.insert(5, 'sales_type', pd.Series(sales_type))
        finished_data.insert(6, 'sales_rank', pd.Series(sales_rank))
        finished_data.insert(7, 'categories', pd.Series(new_categories))

        # modify
        finished_data['brand'] = finished_data['brand'].fillna("").apply(lambda x : str(x).replace("nan", ""))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace("&#39", "'"))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace(",", ""))

        # get attributes (for S3Rec)
        finished_data['attributes'] =\
          finished_data['categories'] + finished_data['brand'].apply(lambda x : (f", '{x}'") if x !='' else '')

        print(f'brand nunique : {finished_data["brand"].nunique()-1}  ')  # searching without NaN & '' 
        print(f'categories nunique : {len(all_categories_set)}')


        # check num attributes
        attributes_set = set()
        for i in finished_data['attributes'] :
            li = i.split(', ')
            for x in li :
                attributes_set.add(x)
        
        print(f'attribute nunique : {len(attributes_set)}')  # brand and categories can be duplicated : (ex) "Disney" 

        return finished_data

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')

# AmazonAutomotiveDataset
class AmazonAutomotiveDataset_review(BaseDataset):
    def __init__(self, input_path, output_path):
        super(AmazonAutomotiveDataset_review, self).__init__(input_path, output_path)
        self.dataset_name = 'AM'

        # input file
        self.inter_file = os.path.join('./Automotive_5.json')
        self.item_file = os.path.join( './meta_Automotive.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # (modify) selected feature fields : add review
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             4: 'review:token_seq',
                             5: 'rating:float',
                             7: 'timestamp:float'}

        """
        self.item_fields = {0: 'item_id:token',
                            1: 'categories:token_seq',
                            3: 'title:token',
                            4: 'price:float',
                            6: 'brand:token',
                            8: 'sales_type:token',
                            9: 'sales_rank:float'}
        """

        
        # (modify) : only features to use, <note> column order varies by dataset
        self.item_fields = {0: 'item_id:token',
                            1: 'categories:token_seq',
                            6: 'brand:token',
                            -1: 'attributes:token_seq'}
        
    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        inter_data = pd.read_json(self.inter_file, lines=True)     
        # review filtering
        ind = inter_data[inter_data['reviewText'] == ""].index
        print(f'review filtering -  drop len:{len(ind)}, drop index: {ind.values}')
        inter_data.drop(ind, inplace=True)        
        
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        
        all_categories_set = set()
        
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 8]
            categories = origin_data.iloc[i, 1]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):

                    # modify : prevent duplication with sep in recbole
                    category = categories[j][k].replace(',', '.')  
                    categories_set.add(category)
                    all_categories_set.add(category)

            new_categories.append(str(categories_set)[1:-1])
            
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
        finished_data.insert(1, 'categories', pd.Series(new_categories))            
        finished_data.insert(8, 'sales_type', pd.Series(sales_type))
        finished_data.insert(9, 'sales_rank', pd.Series(sales_rank))


        # modify
        finished_data['brand'] = finished_data['brand'].fillna("").apply(lambda x : str(x).replace("nan", ""))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace("&#39", "'"))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace(",", ""))

        # get attributes (for S3Rec)
        finished_data['attributes'] =\
          finished_data['categories'] + finished_data['brand'].apply(lambda x : (f", '{x}'") if x !='' else '')

        print(f'brand nunique : {finished_data["brand"].nunique()-1}  ')  # searching without NaN & '' 
        print(f'categories nunique : {len(all_categories_set)}')


        # check num attributes
        attributes_set = set()
        for i in finished_data['attributes'] :
            li = i.split(', ')
            for x in li :
                attributes_set.add(x)
        
        print(f'attribute nunique : {len(attributes_set)}')  # brand and categories can be duplicated : (ex) "Disney" 

        return finished_data

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')


# AmazonAutomotiveDataset
class AmazonGroceryAndGourmetFoodDataset_review(BaseDataset):
    def __init__(self, input_path, output_path):
        super(AmazonGroceryAndGourmetFoodDataset_review, self).__init__(input_path, output_path)
        self.dataset_name = 'GG'

        # input file
        self.inter_file = os.path.join('./Grocery_and_Gourmet_Food_5.json')
        self.item_file = os.path.join( './meta_Grocery_and_Gourmet_Food.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # (modify) selected feature fields : add review
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             4: 'review:token_seq',
                             5: 'rating:float',
                             7: 'timestamp:float'}

        """
        self.item_fields = {0: 'item_id:token',
                            2: 'title:token',
                            5: 'sales_type:token',
                            6: 'sales_rank:float',
                            7: 'categories:token_seq',
                            8: 'price:float',
                            9: 'brand:token'}
        """

        
        # (modify) : only features to use, <note> column order varies by dataset
        self.item_fields = {0: 'item_id:token',
                            7: 'categories:token_seq',
                            9: 'brand:token',
                            -1: 'attributes:token_seq'}
        
    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        inter_data = pd.read_json(self.inter_file, lines=True)     
        # review filtering
        ind = inter_data[inter_data['reviewText'] == ""].index
        print(f'review filtering -  drop len:{len(ind)}, drop index: {ind.values}')
        inter_data.drop(ind, inplace=True)        
        
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        
        all_categories_set = set()
        
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 5]
            categories = origin_data.iloc[i, 6]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):

                    # modify : prevent duplication with sep in recbole
                    category = categories[j][k].replace(',', '.')  
                    categories_set.add(category)
                    all_categories_set.add(category)

            new_categories.append(str(categories_set)[1:-1])
            
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
        finished_data.insert(5, 'sales_type', pd.Series(sales_type))
        finished_data.insert(6, 'sales_rank', pd.Series(sales_rank))
        finished_data.insert(7, 'categories', pd.Series(new_categories))

        # modify
        finished_data['brand'] = finished_data['brand'].fillna("").apply(lambda x : str(x).replace("nan", ""))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace("&#39", "'"))
        finished_data['brand'] = finished_data['brand'].apply(lambda x : str(x).replace(",", ""))

        # get attributes (for S3Rec)
        finished_data['attributes'] =\
          finished_data['categories'] + finished_data['brand'].apply(lambda x : (f", '{x}'") if x !='' else '')

        print(f'brand nunique : {finished_data["brand"].nunique()-1}  ')  # searching without NaN & '' 
        print(f'categories nunique : {len(all_categories_set)}')


        # check num attributes
        attributes_set = set()
        for i in finished_data['attributes'] :
            li = i.split(', ')
            for x in li :
                attributes_set.add(x)
        
        print(f'attribute nunique : {len(attributes_set)}')  # brand and categories can be duplicated : (ex) "Disney" 

        return finished_data

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')

