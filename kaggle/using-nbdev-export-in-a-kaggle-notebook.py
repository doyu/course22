#!/usr/bin/env python
# coding: utf-8

# It can be handy to create script files from notebooks, using nbdev's `notebook2script`. But since Kaggle doesn't actually save the notebook to the file-system, we have to do some workarounds to make this happen. Here's all the steps needed to export a notebook to a script:

# In[1]:


# nbdev requires jupyter, but we're already in a notebook environment, so we can install without dependencies
get_ipython().system('pip install -U nbdev')


# In[2]:


#|default_exp app


# In[3]:


#|export
a=1


# In[4]:


# NB: This only works if you run all the cells in order - click "Save Version" to do this automatically
get_ipython().run_line_magic('notebook', '-e testnbdev.ipynb')


# In[5]:


from nbdev.export import nb_export
nb_export('testnbdev.ipynb', '.')


# In[6]:


get_ipython().system('cat app.py')

