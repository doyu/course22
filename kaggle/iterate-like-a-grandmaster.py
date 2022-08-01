#!/usr/bin/env python
# coding: utf-8

# ## Iterate like a grandmaster

# **Note**: If you're fairly new to Kaggle, NLP, or Transformers, I strongly recommend you read my [Getting Started](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) notebook first, and then come back to this one.
# 
# ---
# 
# There's a lot of impressive notebooks around on Kaggle, but they often  fall into one of two categories:
# 
# - Exploratory Data Analysis (EDA) notebooks with lots of pretty charts, but not much focus on understanding the key issues that will make a difference in the competition
# - Training/inference notebooks with little detail about *why* each step was chosen.
# 
# In this notebook I'll try to give a taste of how a competitions grandmaster might tackle the [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/) competition. The focus generally should be two things:
# 
# 1. Creating an effective validation set
# 1. Iterating rapidly to find changes which improve results on the validation set.
# 
# If you can do these two things, then you can try out lots of experiments and find what works, and what doesn't. Without these two things, it will be nearly impossible to do well in a Kaggle competition (and, indeed, to create highly accurate models in real life!)
# 
# I will show a couple of different ways to create an appropriate validation set, and will explain how to expand them into an appropriate cross-validation system. I'll use just plain HuggingFace Transformers for everything, and will keep the code concise and simple. The more code you have, the more you have to maintain, and the more chances there are to make mistakes. So keep it simple!
# 
# OK, let's get started...

# It's nice to be able to run things locally too, to save your Kaggle GPU hours, so set a variable to make it easy to see where we are, and download what we need:

# In[1]:


from pathlib import Path
import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    get_ipython().system('pip install -Uqq fastai')
else:
    import zipfile,kaggle
    path = Path('us-patent-phrase-to-phrase-matching')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)


# A lot of the basic imports you'll want (`np`, `pd`, `plt`, etc) are provided by fastai, so let's grab them in one line:

# In[2]:


from fastai.imports import *


# ## Import and EDA

# Set a path to our data. Use `pathlib.Path` because it makes everything so much easier, and make it work automatically regardless if you're working on your own PC or on Kaggle!

# In[3]:


if iskaggle: path = Path('../input/us-patent-phrase-to-phrase-matching')
path.ls()


# Let's look at the training set:

# In[4]:


df = pd.read_csv(path/'train.csv')
df


# ...and the test set:

# In[5]:


eval_df = pd.read_csv(path/'test.csv')
len(eval_df)


# In[6]:


eval_df.head()


# Let's look at the distribution of values of `target`:

# In[7]:


df.target.value_counts()


# We see that there's nearly as many unique targets as items in the training set, so they're nearly but not quite unique. Most importantly, we can see that these generally contain very few words (1-4 words in the above sample).
# 
# Let's check `anchor`:

# In[8]:


df.anchor.value_counts()


# We can see here that there's far fewer unique values (just 733) and that again they're very short (2-4 words in this sample).
# 
# Now we'll do `context`

# In[9]:


df.context.value_counts()


# These are just short codes. Some of them have very few examples (18 in the smallest case) The first character is the section the patent was filed under -- let's create a column for that and look at the distribution:

# In[10]:


df['section'] = df.context.str[0]
df.section.value_counts()


# It seems likely that these sections might be useful, since they've got quite a bit more data in each.
# 
# Finally, we'll take a look at a histogram of the scores:

# In[11]:


df.score.hist();


# There's a small number that are scored `1.0` - here's a sample:

# In[12]:


df[df.score==1]


# We can see from this that these are just minor rewordings of the same concept, and isn't likely to be specific to `context`. Any pretrained model should be pretty good at finding these already.

# ## Training

# Time to import the stuff we'll need for training:

# In[13]:


from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer


# In[14]:


if iskaggle:
    get_ipython().system('pip install -q datasets')
import datasets
from datasets import load_dataset, Dataset, DatasetDict


# HuggingFace Transformers tends to be rather enthusiastic about spitting out lots of warnings, so let's quieten it down for our sanity:

# In[15]:


warnings.simplefilter('ignore')
logging.disable(logging.WARNING)


# I tried to find a model that I could train reasonably at home in under two minutes, but got reasonable accuracy from. I found that deberta-v3-small fits the bill, so let's use it:

# In[16]:


model_nm = 'microsoft/deberta-v3-small'


# We can now create a tokenizer for this model. Note that pretrained models assume that text is tokenized in a particular way. In order to ensure that your tokenizer matches your model, use the `AutoTokenizer`, passing in your model name.

# In[17]:


tokz = AutoTokenizer.from_pretrained(model_nm)


# We'll need to combine the context, anchor, and target together somehow. There's not much research as to the best way to do this, so we may need to iterate a bit. To start with, we'll just combine them all into a single string. The model will need to know where each section starts, so we can use the special separator token to tell it:

# In[18]:


sep = tokz.sep_token
sep


# Let's now created our combined column:

# In[19]:


df['inputs'] = df.context + sep + df.anchor + sep + df.target


# Generally we'll get best performance if we convert pandas DataFrames into HuggingFace Datasets, so we'll convert them over, and also rename the score column to what Transformers expects for the dependent variable, which is `label`:

# In[20]:


ds = Dataset.from_pandas(df).rename_column('score', 'label')
eval_ds = Dataset.from_pandas(eval_df)


# To tokenize the data, we'll create a function (since that's what `Dataset.map` will need):

# In[21]:


def tok_func(x): return tokz(x["inputs"])


# Let's try tokenizing one input and see how it looks

# In[22]:


tok_func(ds[0])


# The only bit we care about at the moment is `input_ids`. We can see in the tokens that it starts with a special token `1` (which represents the start of text), and then has our three fields separated by the separator token `2`. We can check the indices of the special token IDs like so:

# In[23]:


tokz.all_special_tokens


# We can now tokenize the input. We'll use batching to speed it up, and remove the columns we no longer need:

# In[24]:


inps = "anchor","target","context"
tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))


# Looking at the first item of the dataset we should see the same information as when we checked `tok_func` above:

# In[25]:


tok_ds[0]


# ## Creating a validation set

# According to [this post](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/315220), the private test anchors do not overlap with the training set. So let's do the same thing for our validation set.
# 
# First, create a randomly shuffled list of anchors:

# In[26]:


anchors = df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)
anchors[:5]


# Now we can pick some proportion (e.g 25%) of these anchors to go in the validation set:

# In[27]:


val_prop = 0.25
val_sz = int(len(anchors)*val_prop)
val_anchors = anchors[:val_sz]


# Now we can get a list of which rows match `val_anchors`, and get their indices:

# In[28]:


is_val = np.isin(df.anchor, val_anchors)
idxs = np.arange(len(df))
val_idxs = idxs[ is_val]
trn_idxs = idxs[~is_val]
len(val_idxs),len(trn_idxs)


# Our training and validation `Dataset`s can now be selected, and put into a `DatasetDict` ready for training:

# In[29]:


dds = DatasetDict({"train":tok_ds.select(trn_idxs),
             "test": tok_ds.select(val_idxs)})


# BTW, a lot of people do more complex stuff for creating their validation set, but with a dataset this large there's not much point. As you can see, the mean scores in the two groups are very similar despite just doing a random shuffle:

# In[30]:


df.iloc[trn_idxs].score.mean(),df.iloc[val_idxs].score.mean()


# ## Initial model

# Let's now train our model! We'll need to specify a metric, which is the correlation coefficient provided by numpy (we need to return a dictionary since that's how Transformers knows what label to use):

# In[31]:


def corr(eval_pred): return {'pearson': np.corrcoef(*eval_pred)[0][1]}


# We pick a learning rate and batch size that fits our GPU, and pick a reasonable weight decay and small number of epochs:

# In[32]:


lr,bs = 8e-5,128
wd,epochs = 0.01,4


# Three epochs might not sound like much, but you'll see once we train that most of the progress can be made in that time, so this is good for experimentation.
# 
# Transformers uses the `TrainingArguments` class to set up arguments. We'll use a cosine scheduler with warmup, since at fast.ai we've found that's pretty reliable. We'll use fp16 since it's much faster on modern GPUs, and saves some memory. We evaluate using double-sized batches, since no gradients are stored so we can do twice as many rows at a time.

# In[33]:


def get_trainer(dds):
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=corr)


# In[34]:


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=wd, report_to='none')


# We can now create our model, and `Trainer`, which is a class which combines the data and model together (just like `Learner` in fastai):

# In[35]:


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
               tokenizer=tokz, compute_metrics=corr)


# Let's train our model!

# In[36]:


trainer.train();


# ## Improving the model

# We now want to start iterating to improve this. To do that, we need to know whether the model gives stable results. I tried training it 3 times from scratch, and got a range of outcomes from 0.808-0.810. This is stable enough to make a start - if we're not finding improvements that are visible within this range, then they're not very significant! Later on, if and when we feel confident that we've got the basics right, we can use cross validation and more epochs of training.
# 
# Iteration speed is critical, so we need to quickly be able to try different data processing and trainer parameters. So let's create a function to quickly apply tokenization and create our `DatasetDict`:

# In[37]:


def get_dds(df):
    ds = Dataset.from_pandas(df).rename_column('score', 'label')
    tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))
    return DatasetDict({"train":tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)})


# ...and also a function to create a `Trainer`:

# In[38]:


def get_model(): return AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

def get_trainer(dds, model=None):
    if model is None: model = get_model()
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=corr)


# Let's now try out some ideas...
# 
# Perhaps using the special separator character isn't a good idea, and we should use something we create instead. Let's see if that makes things better. First we'll change the separator and create the `DatasetDict`:

# In[39]:


sep = " [s] "
df['inputs'] = df.context + sep + df.anchor + sep + df.target
dds = get_dds(df)


# ...and create and train a model. 

# In[40]:


get_trainer(dds).train()


# That's looking quite a bit better, so we'll keep that change.
# 
# Often changing to lowercase is helpful. Let's see if that helps too:

# In[41]:


df['inputs'] = df.inputs.str.lower()
dds = get_dds(df)
get_trainer(dds).train()


# That one is less clear. We'll keep that change too since most times I run it, it's a little better.

# ## Special tokens

# What if we made the patent section a special token? Then potentially the model might learn to recognize that different sections need to be handled in different ways. To do that, we'll use, e.g. `[A]` for section A. We'll then add those as special tokens:

# In[42]:


df['sectok'] = '[' + df.section + ']'
sectoks = list(df.sectok.unique())
tokz.add_special_tokens({'additional_special_tokens': sectoks})


# We concatenate the section token to the start of our inputs:

# In[43]:


df['inputs'] = df.sectok + sep + df.context + sep + df.anchor.str.lower() + sep + df.target
dds = get_dds(df)


# Since we've added more tokens, we need to resize the embedding matrix in the model:

# In[44]:


model = get_model()
model.resize_token_embeddings(len(tokz))


# Now we're ready to train:

# In[45]:


trainer = get_trainer(dds, model=model)
trainer.train()


# It looks like we've made another bit of an improvement!
# 
# There's plenty more things you could try. Here's some thoughts:
# 
# - Try a model pretrained on legal vocabulary. E.g. how about [BERT for patents](https://huggingface.co/anferico/bert-for-patents)?
# - You'd likely get better results by using a sentence similarity model. Did you know that there's a [patent similarity model](https://huggingface.co/AI-Growth-Lab/PatentSBERTa) you could try?
# - You could also fine-tune any HuggingFace model using the full patent database (which is provided in BigQuery), before applying it to this dataset
# - Replace the patent context field with the description of that context provided by the patent office
# - ...and try out your own ideas too!
# 
# Before submitting a model, retrain it on the full dataset, rather than just the 75% training subset we've used here. Create a function like the ones above to make that easy for you!"

# ## Cross-validation

# In[46]:


n_folds = 4


# Once you've gotten the low hanging fruit, you might want to use cross-validation to see the impact of minor changes. This time we'll use [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html#sklearn.model_selection.StratifiedGroupKFold), partly just to show a different approach to before, and partly because it will give us slightly better balanced datasets.

# In[47]:


from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=n_folds)


# Here's how to split the data frame into `n_folds` groups, with non-overlapping anchors and matched scores, after randomly shuffling the rows:

# In[48]:


df = df.sample(frac=1, random_state=42)
scores = (df.score*100).astype(int)
folds = list(cv.split(idxs, scores, df.anchor))
folds


# We can now create a little function to split into training and validation sets based on a fold:

# In[49]:


def get_fold(folds, fold_num):
    trn,val = folds[fold_num]
    return DatasetDict({"train":tok_ds.select(trn), "test": tok_ds.select(val)})


# Let's try it out:

# In[50]:


dds = get_fold(folds, 0)
dds


# We can now pass this into `get_trainer` as we did before. If we have, say, 4 folds, then doing that for each fold will give us 4 models, and 4 sets of predictions and metrics. You could ensemble the 4 models to get a stronger model, and can also average the 4 metrics to get a more accurate assessment of your model. Here's how to get the final epoch metrics from a trainer:

# In[51]:


metrics = [o['eval_pearson'] for o in trainer.state.log_history if 'eval_pearson' in o]
metrics[-1]


# I hope you've found this a helpful guide to improving your results in this competition - and on Kaggle more generally! If you like it, please remember to give it an upvote, and don't hesitate to add a comment if you have any questions or thoughts to add. And if the ideas here are helpful to you in creating your models, I'd really appreciate a link back to this notebook or a comment below to let me know what helped.

# In[ ]:




