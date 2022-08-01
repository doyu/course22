#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# In a recent notebook I tried to answer the question "[Which image models are best?](https://www.kaggle.com/code/jhoward/which-image-models-are-best)" This showed which models in Ross Wightman's [PyTorch Image Models](https://timm.fast.ai/) (*timm*) were the fastest and most accurate for training from scratch with Imagenet.
# 
# However, this is not what most of us use models for. Most of us fine-tune pretrained models. Therefore, what most of us really want to know is which models are the fastest and most accurate for fine-tuning. However, this analysis has not, to my knowledge, previously existed.
# 
# Therefore I teamed up with [Thomas Capelle](https://tcapelle.github.io/about/) of [Weights and Biases](https://wandb.ai/) to answer this question. In this notebook, I present our results.

# ## The analysis

# There are two key dimensions on which datasets can vary when it comes to how well they fine-tune a model:
# 
# 1. How similar they are to the pre-trained model's dataset
# 2. How large they are.
# 
# Therefore, we decided to test on two datasets that were very different on both of these axes. We tested pre-trained models that were trained on Imagenet, and tested fine-tuning on two different datasets:
# 
# 1. The [Oxford IIT-Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which is very similar to Imagenet. Imagenet contains many pictures of animals, and each picture is a photo in which the animal is the main subject. IIT-Pet contains nearly 15,000 images, that are also of this type.
# 2. The [Kaggle Planet](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data) sample contains 1,000 satellite images of Earth. There are no images of this kind in Imagenet.
# 
# So these two datasets are of very different sizes, and very different in terms of their similarity to Imagenet. Furthermore, they have different types of labels - Planet is a multi-label problem, whereas IIT-Pet is a single label problem.
# 
# To test the fine-tuning accuracy of different models, Thomas put together [this script](https://github.com/tcapelle/fastai_timm/blob/main/fine_tune.py). The basic script contains the standard 4 lines of code needed for fastai image recognition models, plus some code to handle various configuration options, such as learning rate and batch size. It was particularly easy to handle in fastai since fastai supports all timm models directly.
# 
# Then, to allow us to easily try different configuration options, Thomas created Weights and Biases (*wandb*) YAML files such as [this one](https://github.com/tcapelle/fastai_timm/blob/main/sweep_planets_lr.yaml). This takes advantage of the convenient [wandb "sweeps"](https://wandb.ai/site/sweeps) feature which tries a range of different levels of a model input and tracks the results.
# 
# wandb makes it really easy for a group of people to run these kinds of analyses on whatever GPUs they have access to. When you create a sweep using the command-line wandb client, it gives you a command to run to have a computer run experiments for the project. You run that same command on each computer where you want to run experiments. The wandb client automatically ensures that each computer runs different parts of the sweep, and has each on report back its results to the wandb server. You can look at the progress in the wandb web GUI at any time during or after the run. I've got three GPUs in my PC at home, so I ran three copies of the client, with each using a different GPU. Thomas also ran the client on a [Paperspace Gradient](https://gradient.run/notebooks) server.
# 
# I liked this approach because I could start and stop the clients any time I wanted, and wandb would automatically handle keeping all the results in sync. When I restarted a client, it would automatically grab from the server whatever the next set of sweep settings were needed. Furthermore, the integration in fastai is really exceptional, thanks particularly to [Boris Dayma](https://github.com/borisdayma), who worked tirelessly to ensure that wandb automatically tracks every aspect of all fastai data processing, model architectures, and optimisation.

# ## Hyperparameters

# We decided to try out all the timm models which had reasonable performance on timm, and which are capable of working with 224x224 px images. We ended up with a list of 86 models and variants to try.
# 
# Our first step was to find a good set of hyper-parameters for each model variant and for each dataset. Our experience at fast.ai has been that there's generally not much difference between models and datasets in terms of what hyperparameter settings work well -- and that experience was repeated in this project. Based on some initial sweeps across a smaller number of representative models, on which we found little variation in optimal hyperparameters, in our final sweep we included all combinations of the following options:
# 
# - Learning rate (AdamW): 0.008 and 0.02
# - Resize method: [Squish](https://docs.fast.ai/vision.augment.html#Resize)
# - Pooling type: [Concat](https://docs.fast.ai/layers.html#AdaptiveConcatPool2d) and Average Pooling
# 
# For other parameters, we used defaults that we've previously found at fast.ai to be reliable across a range of models and datasets (see the fastai docs for details).

# ## Analysis

# Let's take a look at the data. I've put a CSV of the results into a gist:

# In[1]:


from fastai.vision.all import *
import plotly.express as px

url = 'https://gist.githubusercontent.com/jph00/959aaf8695e723246b5e21f3cd5deb02/raw/sweep.csv'


# For each model variant and dataset, for each hyperparameter setting, we did three runs. For the final sweep, we just used the hyperparameter settings listed above.
# 
# For each model variant and dataset, I create a group with the minimum error and fit time, and GPU memory use if used. I use the minimum because there might be some reason that a particular run didn't do so well (e.g. maybe there was some resource contention), and I'm mainly interested in knowing what the best case results for a model can be.
# 
# I create a "score" which, somewhat arbitrarily combines the accuracy and speed into a single number. I tried a few options until I came up with something that closely matched my own opinions about the tradeoffs between the two. (Feel free of course to fork this notebook and adjust how that's calculated.)

# In[2]:


df = pd.read_csv(url)
df['family'] = df.model_name.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')
df.loc[df.family=='swinv2', 'family'] = 'swin'
pt_all = df.pivot_table(values=['error_rate','fit_time','GPU_mem'], index=['dataset', 'family', 'model_name'],
                        aggfunc=np.min).reset_index()
pt_all['score'] = pt_all.error_rate*(pt_all.fit_time+80)


# ### IIT Pet

# Here's the top 15 models on the IIT Pet dataset, ordered by score:

# In[3]:


pt = pt_all[pt_all.dataset=='pets'].sort_values('score').reset_index(drop=True)
pt.head(15)


# As you can see, the [convnext](https://arxiv.org/abs/2201.03545), [swin](https://arxiv.org/abs/2103.14030), and [vit](https://arxiv.org/abs/2010.11929) families are fairly dominent. The excellent showing of `convnext_tiny` matches my view that we should think of this as our default baseline for image recognition today. It's fast, accurate, and not too much of a memory hog. (And according to Ross Wightman, it could be even faster if NVIDIA and PyTorch make some changes to better optimise the operations it relies on!)
# 
# `vit_small_patch16` is also a good option -- it's faster and leaner on memory than `convnext_tiny`, although there is some performance cost too.
# 
# Interestingly, resnets are still a great option -- especially the [`resnet26d`](https://arxiv.org/abs/1812.01187) variant, which is the fastest in our top 15.
# 
# Here's a quick visual representation of the seven model families which look best in the above analysis (the "fit lines" are just there to help visually show where the different families are -- they don't necessarily actually follow a linear fit):

# In[4]:


w,h = 900,700
faves = ['vit','convnext','resnet','levit', 'regnetx', 'swin']
pt2 = pt[pt.family.isin(faves)]
px.scatter(pt2, width=w, height=h, x='fit_time', y='error_rate', color='family', hover_name='model_name', trendline="ols",)


# This chart shows that there's a big drop-off in performance towards the far left. It seems like there's a big compromise if we want the fastest possible model. It also seems that the best models in terms of accuracy, convnext and swin, aren't able to make great use of the larger capacity of larger models. So an ensemble of smaller models may be effective in some situations.
# 
# Note that `vit` doesn't include any larger/slower models, since they only work with larger images. We would recommend trying larger models on your dataset if you have larger images and the resources to handle them.
# 
# I particularly like using fast and small models, since I wanted to be able to iterate rapidly to try lots of ideas (see [this notebook](https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster) for more on this). Here's the top models (based on accuracy) that are smaller and faster than the median model:

# In[5]:


pt.query("(GPU_mem<2.7) & (fit_time<110)").sort_values("error_rate").head(15).reset_index(drop=True)


# ...and here's the top 15 models that are the very fastest and most memory efficient:

# In[6]:


pt.query("(GPU_mem<1.6) & (fit_time<90)").sort_values("error_rate").head(15).reset_index(drop=True)


# [ResNet-RS](https://arxiv.org/abs/2103.07579) performs well here, with lower memory use than convnext but nonetheless high accuracy. A version trained on the larger Imagenet-22k dataset (like `convnext_tiny_in22k` would presumably do even better, and may top the charts!)
# 
# [RegNet-y](https://arxiv.org/abs/2003.13678) is impressively miserly in terms of memory use, whilst still achieving high accuracy.

# ### Planet

# Here's the top-15 for Planet:

# In[7]:


pt = pt_all[pt_all.dataset=='planet'].sort_values('score').reset_index(drop=True)
pt.head(15)


# Interestingly, the results look quite different: *vit* and *swin* take most of the top positions in terms of the combination of accuracy and speed. `vit_small_patch32` is a particular standout with its extremely low memory use and also the fastest in the top 15.
# 
# Because this dataset is so different to Imagenet, what we're testing here is more about how quickly and data-efficiently a model can learn new features that it hasn't seen before. We can see that the transformers-based architectures able to do that better than any other model. `convnext_tiny` still puts in a good performance, but it's a bit let down by it's relatively poor speed -- hopefully we'll see NVIDIA speed it up in the future, because in theory it's a light-weight architecture which should be able to do better.
# 
# The downside of vit and swin models, like most transformers-based models, is that they can only handle one input image size. Of course, we can always squish or crop or pad our input images to the required size, but this can have a significant impact on performance. For instance, recently in looking at the [Kaggle Paddy Disease](https://www.kaggle.com/competitions/paddy-disease-classification) competition I've found that the ability of convnext models to handle dynamically sized inputs to be very convenient.

# Here's a chart of the seven top families, this time for the Planet dataset:

# In[8]:


pt2 = pt[pt.family.isin(faves)]
px.scatter(pt2, width=w, height=h, x='fit_time', y='error_rate', color='family', hover_name='model_name', trendline="ols")


# One striking feature is that for this dataset, there's little correlation between model size and performance. Regnetx and vit are the only families that show much of a relationship here. This suggests that if you have data that's very different to your pretrained model's data, that you might want to focus on smaller models. This makes intuitive sense, since these models have more new features to learn, and if they're too big they're either going to overfit, or fail to utilise their capacity effectively.
# 
# Here's the most accurate small and fast models on the Planet dataset:

# In[9]:


pt.query("(GPU_mem<2.7) & (fit_time<25)").sort_values("error_rate").head(15).reset_index(drop=True)


# `convnext_tiny` is still the most accurate option amongst architectures that don't have a fixed resolution. Resnet 18 has very low memory use, is fast, and is still quite accurate.
# 
# Here's the subset of ultra lean/fast models on the Planet dataset:

# In[10]:


pt.query("(GPU_mem<1.6) & (fit_time<21)").sort_values("error_rate").head(15).reset_index(drop=True)


# ## Conclusions

# It really seems like it's time for a changing of the guard when it comes to computer vision models. There are, as at the time of writing (June 2022) three very clear winners when it comes to fine-tuning pretrained models:
# 
# - [convnext](https://arxiv.org/abs/2201.03545)
# - [vit](https://arxiv.org/abs/2010.11929)
# - [swin](https://arxiv.org/abs/2103.14030) (and [v2](https://arxiv.org/abs/2111.09883)).
# 
# [Tanishq Abraham](https://www.kaggle.com/tanlikesmath) studied the top results of a [recent Kaggle computer vision competition](https://www.kaggle.com/c/petfinder-pawpularity-score) and found that the above three approaches did indeed appear to the best approaches. However, there were two other architectures which were also very strong in that competition, but which aren't in our top models above:
# 
# - [EfficientNet](https://arxiv.org/abs/1905.11946) and [v2](https://arxiv.org/abs/2104.00298)
# - [BEiT](https://arxiv.org/abs/2106.08254).
# 
# BEiT isn't there because it's too big to fit on my GPU (even the smallest BEiT model is too big!) This is fixable with gradient accumulation, so perhaps in a future iteration we'll add it in. EfficientNet didn't have any variants that were fast and accurate enough to appear in the top 15 on either dataset. However, it's notoriously fiddly to train, so there might well be some set of hyperparameters that would work for these datasets. Having said that, I'm mainly interested in knowing which architectures can be trained quickly and easily without to much mucking around, so perhaps EfficientNet doesn't really fit here anyway!
# 
# Thankfully, it's easy to try lots of different models, especially if you use fastai and timm, because it's literally as easy as changing the model name in one place in your code. Your existing hyperparameters are most likely going to continue to work fine regardless of what model you try. And it's particularly easy if you use [wandb](https://wandb.ai/), since you can start and stop experiments at any time and they'll all be automatically tracked and managed for you.
# 
# If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. And if you have any questions or comments, please pop them below -- I read every comment I receive!

# In[ ]:




