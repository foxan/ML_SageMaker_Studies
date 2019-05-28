# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plagiarism Text Data
#
# In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either plagiarized or not, depending on how similar the text file is when compared to a provided source text. 
#
# The first step in working with any dataset is loading the data in and noting what information is included in the dataset. This is an important step in eventually working with this data, and knowing what kinds of features you have to work with as you transform and group the data!
#
# So, this notebook is all about exploring the data and noting patterns about the features you are given and the distribution of data. 
#
# > There are not any exercises or questions in this notebook, it is only meant for exploration. This notebook will note be required in your final project submission.
#
# ---

# %% [markdown]
# ## Read in the Data
#
# The cell below will download the necessary data and extract the files into the folder `data/`.
#
# This data is a slightly modified version of a dataset created by Paul Clough (Information Studies) and Mark Stevenson (Computer Science), at the University of Sheffield. You can read all about the data collection and corpus, at [their university webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html). 
#
# > **Citation for data**: Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [Download]

# %%
# !wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip
# !unzip data

# %%
# import libraries
import pandas as pd
import numpy as np
import os

# %% [markdown]
# This plagiarism dataset is made of multiple text files; each of these files has characteristics that are is summarized in a `.csv` file named `file_information.csv`, which we can read in using `pandas`.

# %%
csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
plagiarism_df.head(10)

# %% [markdown]
# ## Types of Plagiarism
#
# Each text file is associated with one **Task** (task A-E) and one **Category** of plagiarism, which you can see in the above DataFrame.
#
# ###  Five task types, A-E
#
# Each text file contains an answer to one short question; these questions are labeled as tasks A-E.
# * Each task, A-E, is about a topic that might be included in the Computer Science curriculum that was created by the authors of this dataset. 
#     * For example, Task A asks the question: "What is inheritance in object oriented programming?"
#
# ### Four categories of plagiarism 
#
# Each text file has an associated plagiarism label/category:
#
# 1. `cut`: An answer is plagiarized; it is copy-pasted directly from the relevant Wikipedia source text.
# 2. `light`: An answer is plagiarized; it is based on the Wikipedia source text and includes some copying and paraphrasing.
# 3. `heavy`: An answer is plagiarized; it is based on the Wikipedia source text but expressed using different words and structure. Since this doesn't copy directly from a source text, this will likely be the most challenging kind of plagiarism to detect.
# 4. `non`: An answer is not plagiarized; the Wikipedia source text is not used to create this answer.
# 5. `orig`: This is a specific category for the original, Wikipedia source text. We will use these files only for comparison purposes.
#
# > So, out of the submitted files, the only category that does not contain any plagiarism is `non`.
#
# In the next cell, print out some statistics about the data.

# %%
# print out some stats about the data
print('Number of files: ', plagiarism_df.shape[0])  # .shape[0] gives the rows 
# .unique() gives unique items in a specified column
print('Number of unique tasks/question types (A-E): ', (len(plagiarism_df['Task'].unique())))
print('Unique plagiarism categories: ', (plagiarism_df['Category'].unique()))

# %% [markdown]
# You should see the number of text files in the dataset as well as some characteristics about the `Task` and `Category` columns. **Note that the file count of 100 *includes* the 5 _original_ wikipedia files for tasks A-E.** If you take a look at the files in the `data` directory, you'll notice that the original, source texts start with the filename `orig_` as opposed to `g` for "group." 
#
# > So, in total there are 100 files, 95 of which are answers (submitted by people) and 5 of which are the original, Wikipedia source texts.
#
# Your end goal will be to use this information to classify any given answer text into one of two categories, plagiarized or not-plagiarized.

# %% [markdown]
# ### Distribution of Data
#
# Next, let's look at the distribution of data. In this course, we've talked about traits like class imbalance that can inform how you develop an algorithm. So, here, we'll ask: **How evenly is our data distributed among different tasks and plagiarism levels?**
#
# Below, you should notice two things:
# * Our dataset is quite small, especially with respect to examples of varying plagiarism levels.
# * The data is distributed fairly evenly across task and plagiarism types.

# %%
# Show counts by different tasks and amounts of plagiarism

# group and count by task
counts_per_task=plagiarism_df.groupby(['Task']).size().reset_index(name="Counts")
print("\nTask:")
display(counts_per_task)

# group by plagiarism level
counts_per_category=plagiarism_df.groupby(['Category']).size().reset_index(name="Counts")
print("\nPlagiarism Levels:")
display(counts_per_category)

# group by task AND plagiarism level
counts_task_and_plagiarism=plagiarism_df.groupby(['Task', 'Category']).size().reset_index(name="Counts")
print("\nTask & Plagiarism Level Combos :")
display(counts_task_and_plagiarism)

# %% [markdown]
# It may also be helpful to look at this last DataFrame, graphically.
#
# Below, you can see that the counts follow a pattern broken down by task. Each task has one source text (original) and the highest number on `non` plagiarized cases.

# %%
import matplotlib.pyplot as plt
% matplotlib inline

# counts
group = ['Task', 'Category']
counts = plagiarism_df.groupby(group).size().reset_index(name="Counts")

plt.figure(figsize=(8,5))
plt.bar(range(len(counts)), counts['Counts'], color = 'blue')

# %% [markdown]
# ## Up Next
#
# This notebook is just about data loading and exploration, and you do not need to include it in your final project submission. 
#
# In the next few notebooks, you'll use this data to train a complete plagiarism classifier. You'll be tasked with extracting meaningful features from the text data, reading in answers to different tasks and comparing them to the original Wikipedia source text. You'll engineer similarity features that will help identify cases of plagiarism. Then, you'll use these features to train and deploy a classification model in a SageMaker notebook instance. 
