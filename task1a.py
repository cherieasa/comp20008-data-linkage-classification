# Code written by Terada Asavapakuna 1012869

# import libraries
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# read in csv file into pandas df
amazon_df = pd.read_csv("amazon_small.csv")
google_df = pd.read_csv("google_small.csv")
a_list_title = (amazon_df.title).tolist()
g_list_title = (google_df.name).tolist()
list_comp = []


google_desc = google_df.description.tolist()
amazon_desc = amazon_df.description.tolist()

# list of booleans, if True in that position there is a NaN
no_amazon_desc_index = pd.Index(amazon_df.description).isna()
no_google_desc_index = pd.Index(google_df.description).isna()

# function to change NaN description to ""
def nan_description(list_description, checking_list):
    cleansed_list = []
    count = 0
    for each_desc in list_description:
        # if the string is empty
        if (checking_list[count] == True):
            cleansed_list.append("")
        else:
            cleansed_list.append(each_desc)
            
        count += 1

    return cleansed_list

# removed nan description list
cleansed_amazon_desc = nan_description(amazon_desc, no_amazon_desc_index)
cleansed_google_desc = nan_description(google_desc, no_google_desc_index)

# remove punctuation
def remove_punct(title_or_desc_list):
    # remove punct and append to stripped_title
    table = str.maketrans('', '', string.punctuation)
    stripped_title = [w.translate(table) for w in title_or_desc_list]
    
    return(stripped_title)

# remove stop words
def stop_word_removal(stripped_title_or_desc):
    no_stop_word = []
    for sentence_list in stripped_title_or_desc:
            text_tokens = word_tokenize(sentence_list)
            tokens_no_stopword = [word for word in text_tokens if not word in stopwords.words()]
            joined = ' '.join(tokens_no_stopword)
            no_stop_word.append(joined)

    return no_stop_word

# removed stop words
a_stop_title = stop_word_removal(a_list_title)
a_stop_desc =  stop_word_removal(cleansed_amazon_desc)

g_stop_title = stop_word_removal(g_list_title)
g_stop_desc = stop_word_removal(cleansed_google_desc)

# removed punctuation
a_stripped_title = remove_punct(a_stop_title)
a_stripped_desc =  remove_punct(a_stop_desc)

g_stripped_title = remove_punct(g_stop_title)
g_stripped_desc = remove_punct(g_stop_desc)

# function to make a dictionary of ids and their new titles and description
def make_dict(list_item, ids):
    combined = dict()
    for i in range(len(ids)):
        combined[list_item[i]] = ids[i]
    
    return combined
    
a_title_dict = make_dict(a_stripped_title, (amazon_df.idAmazon).tolist())
a_desc_dict = make_dict(a_stripped_desc, (amazon_df.idAmazon).tolist())
g_title_dict = make_dict(g_stripped_title, (google_df.idGoogleBase).tolist())
g_desc_dict = make_dict(g_stripped_desc, (google_df.idGoogleBase).tolist())

# to find scores of each amazon item compared with google item
count = 0
compare_title_list = []
compare_desc_list = []

for each_itemid in amazon_df.idAmazon:
    
    # -------- compare title --------
    
    # get title of each_itemid
    each_title = a_stripped_title[count]
    
    # compare title with all names in google_small
    compare_title = process.extract(each_title,g_stripped_title,scorer = fuzz.token_set_ratio, limit = 1)
    compare_title_list.append(compare_title)
    
    
    # -------- compare desc --------
    
    if (no_amazon_desc_index[count] == False):
        # get desc of each_itemid and avoid empty string
        each_desc = a_stripped_desc[count]
        compare_desc = process.extract(each_desc,g_stripped_desc,scorer = fuzz.token_sort_ratio, limit = 1)
    
    # no description exists -> score = 0
    else:
        compare_desc = [("EMPTY", 0)]
        
    compare_desc_list.append(compare_desc)
    count += 1

# compare scores for title and desc
matched_item_list = []
for i in range(len(amazon_df.idAmazon)):
    
    # extract title similarity score
    for j in compare_title_list[i]:
        title_original = j[0]
        title_comp = j[1]
    
    # extract description similarity score
    for k in compare_desc_list[i]:
        desc_original = k[0]
        desc_comp = k[1]
        
    # use title match    
    if (title_comp >= desc_comp):
        matched_item = g_title_dict[title_original]
        
    # use desc match
    elif (title_comp < desc_comp):
        matched_item = g_desc_dict[desc_original]
   
            
    matched_item_list.append(matched_item)


# combine idAmazon with corresponding idGoogleBase

amazon_ids = amazon_df.idAmazon.tolist()
google_ids = matched_item_list

# create dict and output to csv file

task1a_dict = {'idAmazon': amazon_ids, 'idGoogleBase': google_ids}

task1a_df = pd.DataFrame(task1a_dict) 

task1a_df.to_csv('task1a.csv', index=False)

