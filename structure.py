# For dealing with files
import os
import shutil

# For using regex expressions
import re

# For splitting the data
from sklearn.model_selection import train_test_split

# cd to train directory
os.chdir('train')

# Get a list of all filenames inside (these will be used for training and validation)
files = os.listdir()

# Filter list using regex expressions
r_cat = re.compile('cat.*.jpg')
r_dog = re.compile('dog.*.jpg')
all_cat_filenames = list(filter(r_cat.match, files))
all_dog_filenames = list(filter(r_dog.match, files))

print('Found {} images of cats.\nFound {} images of dogs.'.format(len(all_cat_filenames), len(all_dog_filenames)))

# Get a subset of the entire training dataset (20%)
_, few_cat_filenames, _, few_dog_filenames = train_test_split(all_cat_filenames, 
                                                              all_dog_filenames, 
                                                              test_size=0.2, random_state=1)

# Split it into training and validation sets
few_cat_filenames_train, few_cat_filenames_val, few_dog_filenames_train, few_dog_filenames_val = \
train_test_split(few_cat_filenames, 
              few_dog_filenames, 
              test_size = 0.3,
              random_state=2)

print('The smaller dataset will be comprised of:')
print('Train:\t', len(few_cat_filenames_train), 'cats and', len(few_dog_filenames_train), 'dogs.')
print('Val:\t', len(few_cat_filenames_val), 'cats and', len(few_dog_filenames_val), 'dogs.')

# Create the train and val directories and subdirectories
if not os.path.isdir('../small_train'):
    os.mkdir('../small_train')
             
if not os.path.isdir('../small_train/cats'):
    os.mkdir('../small_train/cats')

if not os.path.isdir('../small_train/dogs'):
    os.mkdir('../small_train/dogs')
    
if not os.path.isdir('../small_val'):
    os.mkdir('../small_val')
    
if not os.path.isdir('../small_val/cats'):
    os.mkdir('../small_val/cats')

if not os.path.isdir('../small_val/dogs'):
    os.mkdir('../small_val/dogs')   
    
# Put the training and validation data in the respective folders
for f in few_cat_filenames_train:
    shutil.copyfile(f,'../small_train/cats/'+f)
    
for f in few_dog_filenames_train:
    shutil.copyfile(f,'../small_train/dogs/'+f)   
    
for f in few_cat_filenames_val:
    shutil.copyfile(f,'../small_val/cats/'+f)
    
for f in few_dog_filenames_val:
    shutil.copyfile(f,'../small_val/dogs/'+f)        

# Choose
my_split_ratio = 0.2

# Split it
all_cat_filenames_train, all_cat_filenames_val, all_dog_filenames_train, all_dog_filenames_val = \
train_test_split(all_cat_filenames,
                 all_dog_filenames,
                 test_size=my_split_ratio,
                 random_state=3)

print('The full dataset will be comprised of:')
print('Train:\t', len(all_cat_filenames_train), 'cats and', len(all_dog_filenames_train), 'dogs.')
print('Val:\t', len(all_cat_filenames_val), 'cats and', len(all_dog_filenames_val), 'dogs.')

# Create the train and val directories and subdirectories
if not os.path.isdir('cats'):
    os.mkdir('cats')

if not os.path.isdir('dogs'):
    os.mkdir('dogs')
    
if not os.path.isdir('../val'):
    os.mkdir('../val')
    
if not os.path.isdir('../val/cats'):
    os.mkdir('../val/cats')

if not os.path.isdir('../val/dogs'):
    os.mkdir('../val/dogs')   
    
# Put the training and validation data in the respective folders
for f in all_cat_filenames_train:
    shutil.move(f,'cats')
    
for f in all_dog_filenames_train:
    shutil.move(f,'dogs')   
    
for f in all_cat_filenames_val:
    shutil.move(f,'../val/cats/')
    
for f in all_dog_filenames_val:
    shutil.move(f,'../val/dogs/')       