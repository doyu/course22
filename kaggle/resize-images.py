#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -Uqq fastai')


# In[2]:


from fastai.vision.all import *

path = Path('../input/small-jpegs-fgvc')
trn_path = path/'train'
tst_path = path/'test'

trns_path = Path('train')
trns_path.mkdir(exist_ok=True)
tsts_path = Path('test')
tsts_path.mkdir(exist_ok=True)


# In[3]:


resize_images(trn_path, max_workers=8, dest=trns_path, max_size=512)


# In[4]:


resize_images(tst_path, max_workers=8, dest=tsts_path, max_size=512)


# In[ ]:




