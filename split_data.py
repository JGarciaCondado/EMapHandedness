import os, random, pickle

# Directory of dataset
dataset_dir = 'nrPDB/dataset/'
metadata_dir = 'nrPDB/metadata/'

# Fractions to split
train = 0.6
val = 0.2
test = 0.2

# Get files
files = [dataset_dir + f for f in os.listdir(dataset_dir) if f[-4:] == '.pdb' ]

# Shuffle
files_shuffled = random.shuffle(files)

# Splitting index
total_f = len(files)
train_split = int(total_f*train)
val_split = int(total_f*val)
test_split = int(total_f*test)

# Obtain file names
train_f = files[:train_split]
val_f = files[train_split:train_split+val_split]
test_f = files[-test_split:]

# Save as pickle object
with open(metadata_dir+'data_split.pkl', 'wb') as f:
     pickle.dump([train_f, val_f, test_f], f)
