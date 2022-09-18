# UNSPSC Classification

This project prepares a classifier for the United Nations Standard Product and Services Classification system using the SageMaker Ecosystem. It is completed in partial satisfaction of the Udacity AWS Machine Learning Engineer NanoDegree. 

## TL:DR

The United Nations Standard Product & Service Classification categorises goods and services into a hierarchy of classification areas which are non-overlapping and provide good coverage of typical product and services. Procurement organisations use these classifications to identify what money they are spending. This git repo contains conde which will (in a properly configured AWS environment) automate the process of downloading training data, training models and deploying these models to an endpoint to use in production for classification. It also provides information on how to configure AWS infrastructure to enable this deployment.

Training data is all in the public domain and provided by various anglophone governments.

## Problem Definition

When describing what goods and services are procured, it is necessary to have a common langauge to categorise these goods and services. Descriptors of goods and services are most often provided in short text descriptions and require manual encoding to categorise them. This categorisation is frequently subject and often relies upon the skill of the coder to ensure that items are correclty classified. Additionally, in order to be useful, classification systems must provide sufficient (but not too many) classifications with interpretable content to allow reprorting from these classification systmes to be understood by non-technical audiences.

### Trade Mark Use Case

#### Concerning Nice Classification

The Nice Classfiication system of goods and services provides 45 high level classification symbols to assign goods and service usage information to trade marks. Trade marks may exist in all 45 of these symbols concurrently, or they may be limited to one or two specific symbols. The Nice Classification system is insufficiently specific for the purposes of identifying goods and services to which current and prospective trade marks may be applied. 

#### Goods & Services Item Descriptions

When an application for a trade mark is submitted, along with classification information in the relevant standard (most nations use Nice -- the US has its own system), applicants are required to submit descriptions in plain language of the goods and services to which their prospective mark may attach. Consider, for example, Australian Trade Mark number 957064:

> Edible sandwiches for consumption on or off the premises, coffee, coffee substitutes, tea, cocoa, sugar, honey, rice, tapioca, flour, breakfast cereals, processed cereals, cereal-based snack foods and ready to eat cereal-derived food bars; bread, biscuits, cakes, pastries, dairy-based shakes, soft-serve ice cream, ice milk and frozen yoghurt; yeast, baking powder, salt, mustard, pepper, sauces, spices, seasonings and ice.<br>
> See https://search.ipaustralia.gov.au/trademarks/search/view/957064 Class 30.

This is a partial goods and services item description for the "I'm lovin' it" mark used by McDonald's Restaurants in Australia. Let us consider the definition provided by WIPO for Nice Class 30:

>Coffee, tea, cocoa and substitutes therefor; rice, pasta and noodles; tapioca and sago; flour and preparations made from cereals; bread, pastries and confectionery; chocolate; ice cream, sorbets and other edible ices; sugar, honey, treacle; yeast, baking-powder; salt, seasonings, spices, preserved herbs; vinegar, sauces and other condiments; ice (frozen water).<br>
> See https://www.wipo.int/classifications/nice/nclpub/en/fr/?basic_numbers=show&class_number=30

We should pay attention though that this trade mark also has items in Nice Class 29. WIPO says that Nice class 29 is:

>Meat, fish, poultry and game; meat extracts; preserved, frozen, dried and cooked fruits and vegetables; jellies, jams, compotes; eggs; milk, cheese, butter, yogurt and other milk products; oils and fats for food.<br>
>See https://www.wipo.int/classifications/nice/nclpub/en/fr/?basic_numbers=show&class_number=29

While the items that are covered between the two classes are clearly different, the short descriptions make it very difficult to identify what a relative rise in the number of class 30 trade marks would mean for the food branding sector over an increase in the number of class 29 trade marks. When providing high level reporting on brand landscapes, the use of Nice classification as a category variable is very difficult and in practice it is rarely used in such analysis in my experience.

### What is UNSPSC

The United Nations Standard Product & Service Classification categorises goods and services into a hierarchy of classification areas which are non-overlapping and provide good coverage of typical product and services. The standard has four levels of classification:

1. Segment
2. Family
3. Class
4. Commodity

At the highest level, there are around 57 market segments and at the most granular, there are in the tens of thousands of commodities. This allows for scalable classification specificity. Market segement classification is enough to provide equivalent levels of granularity to Nice. In addition, the meaning of the classification symbols is much more clear:

Consier the UNSPSC classification description for market Segment code 23000000:

>23000000, Industrial Manufacturing and Processing Machinery and Accessories

We are able to get a good understanding of the likely things that may be included within the classification symbol from the description alone. Application of the UNSPSC Classification, therefore, to the goods and services item descriptions of trade marks would allow more useful analysis of the brand space than the Nice classification system alone. Additionally, if we are able to further extend this work to more granular levels of the classification standard, we can significantly enhance the utility of Trade Mark data to both IP offices and the general public.

### Available Data 

In order to train a classifier, it is necessary to identify labelled datasets of goods and service descriptions which can be used to train a natural language classification model. The main domain of use currently of the UNSPSC classification system appears to be in the area of procurement descriptions for expense reports. Importantly, three governments were identified with open data holdings which provide both the tagged UNSPSC code (usually at the Commodity level), along with a written description of varying lengths of the goods and services procured. Data is available from:

1. Californian Government
2. Australian Government
3. Canadian Government

However, the Canadian data is not mature yet as Canada is in the process of abandoning a legacy coding system in favour of the adoption of the UNSPSC system for providing procurement data to the general public. As a result it was abandoned:

#### Exploitation of Code Hierarchies

Code data itself is provided by the government of Oklahoma. As our intent is to classify only to the highest level of the classification system, we can use all of the descriptions of more granular classification symbols nested within the target as additional training data. This training data is also highly valuable as because it is from the classification system itself, it is extremely clean in comparison to the procurement data.

The other consequence of classification information being provided by procurement agencies is that these descriptions can be inferred to belong to the classes at all higher levels of classification than the one that they are provided.


### Performance Metrics

When building the classifier, we need to be aware of how we would measure the success of the project. Initially, the original scoping of the project deemed that accuracy would the best metric by which to measure the accuracy of the classifier. However, when considering the significant class imbalance problems present in the dataset (please see the sections below), it was decided to employ a [balanced accuracy score from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score). This metric is the average of the recall on each of the classes. This corrects in the case of imbalanced class datasets by preventing a classifier from being rewarded from merely biasing all classification to the over-represented class. Even where the support for a class is small, failure to correctly classify members of this class is equally weighted with classes which have higher support. This will lead to a classifier that has better performance across unseen data than one that exploits the imbalance in the training set.

## Analysis

### Dataset Preparation

The file `wrangling_segment.py` allows the user to specify either to use segment, family or specify their own summary groups of the above. This function is intended to be used in cases where a series of classifiers are used to first conduct preclassification into groups prior to classification into a target.

If the flag `--download` is passed to this script, the script will download and unpack the dataset files from the open data repositories that source them.

For the purposes of the demonstration of this project, it is sufficient to demonstrate the capabilities of this classification system to simply train at the family level.

As part of this preprocessing script, the following steps are conducted to preprocess the data:

1. If required strips html tags from the descriptions.
2. Strips accents from descriptions.
3. Strips excessive white space.
4. Strips extraneous carriage returns.

### Sequence Length

After this processing, exploratory analysis was conducted in accordance with the content of [`exploratory_data_analysis.ipynb`](./exploratory_data_analysis.ipynb) in this repository. The median length of the descriptions sequences is 6 words after preprocessing. The modal sequence length was 4 words. The distribution is heavily right skewed. 

In order to ensure maximum sequence preservation, I note that a large sequence length was chosen for trimming activities. Less than 1% of the data was found to have over 128 words. This means that choosing to trim sequences of this length is highly unlikely to affect the performacne of the classifier.

### Class Imbalance

In the exploratory data analysis, it was noted that there are significant class imbalance issues in the dataset at the family level. The largest class was noted to have only 85 members, while the largest was found to have over 55,000 members. These datasets are obtained from a combination of processed classification standard datasets as well as government purchasing record datasets. There are products and services which governments are unlikely to procure compared to the general population. As a result, class imbalance given the bias of the training data was not unexpected. Class imbalance was treated using a data augmentation method described below.

### Solution Options: RNN

Some of the earlier models used to solve text classification problems were Recurrent Neural Networks. These networks allow sequence information to be processed in such a way that previous parts of a sequence are fed into the network as inputs allowing the model to learn from the content of a sequence rather than the individual parts of the sequence alone.

These models were largely surplanted first by Long Short Term Memory (LSTM) models which are a modified RNN architecture. 

#### Benchmark Model

The benchmark model for the classification is a from scratch trained recurrent neural network model. This model was trained over 200 epochs and includes 3 hidden layers to allow for nonlinearity given the large number of target classes compared to the input size. In development of this model, acceptable performance was first obtained using a locally trained model prior to that model being ported to sagemaker.

The model achieved an accuracy of 70.3% and a balanced accuracy of 51.6%

### Solution Options: Transformers

Transformer models represent the state of the art in sequence embedding language processing. These models are trained on large datasets and utilise a mechanism of self-attention to transform sequence embedding information into some level of semantic content of language sequences. These models can be used to perform semantic text similarity analysis along with other machine language tasks.

#### HuggingFace

HuggingFace is an open source project which contains implementations of common language models in a range of programming languages. HuggingFace allows users to download the weights for pretrained language models that would otherwise be beyond the reach of analysts without the resources of large technology enterprises like google. HuggingFace makes available GPT-2, BERT and other popular language models, along with tokenizers.

#### DistilBERT
In order to minimise the complexity of the resulting model, a smaller language model was chosen. DistilBERT is a model which seeks to replicate the performance of the larger language model BERT without the need for so many trainable parameters. This will hopefully improve training times over a more full featured model while retaining the majority of the performance.

## Methodology 

### Download & Preprocessing
As noted above, the `wrangling_segment.py` script, if passed the `--download` flag will download the files from the relevant open data sources from the internet. This data then has some basic string cleaning applied as described above.

#### Data Augmentation Using Randon Synonyms

The class imbalance issue in this dataset was noted to be a significant concern as a result of the largest class having over 600 times the number of members as the smallest class. In [this](https://neptune.ai/blog/data-augmentation-nlp) blog by Neptune.ai, the author discusses the issue of class imbalance in training data for Natural Language Processing (nlp) problems. The author advocates the use of data augmentation using a number of methods. From these methods, random synonym replacement was used through the python [`nlpaug`](https://pypi.org/project/nlpaug/) library. This method uses WordNet to identify synonyms for randomly selected words in the provided strings.

The augmentation script `data_augmentation.py` allows the user to pass a desired number of members. Any class in the dataset which has fewer than this desired number of members will be randomly sampled and have synonym replacement applied. The user can also specify if the classes with more than this number of members should be undersampled to produce a perfectly balanced set of classes in the training data. If this option is specified, the excess members of the now undersampled training classes are transferred to the test set to allow them to be used in evaluation.

The default behaviour of the script is to only augment the under-represented classes and leave the over-represented classes unmodified. 

This is the method used for this demonstration. AS a result the training set contains over 400,000 records.

#### String Length Truncation

In line with the findings of the exploratory data analysis, for the baseline model, strings of longer than 64 words were truncated and for the DistilBERT model, sequences of longer than 128 were used. These are likely to truncate fewer than 1% of the samples.

### Examples Used to Iterate

Udacity provided an example of the use of a sentence transformer model for text classification. This model was used as a starting point for iteration and modification for the present use case.

[This](https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-rnn-for-text-classification-tasks) blog post was used to inform the initial approaches to the development of the RNN baseline model.

### Initial Local Implementation

My local machine has an NVidia GTX1060 GPU. I have installed CUDA on this machine to allow local development in the initial stages prior to redeploying the produced model to Sagemaker. If you would like to use this model, I strongly recommend the use of a local GPU if available as a way of reducing costs.

#### The Need for Data Augmentation

The locally trained model was initially implemented without class rebalancing and utilised accuracy rather than balanced accuracy as the performance measure. However, without data augmentation, the model was noted to suffer from performance issues with classes that it had limited exposure to. This was obscured in the performance metrics by the use of accuracy rather than balanced accuracy.

#### Modification for AWS Environment

The scripts were modified for use in the AWS environment by adding arguments which allowed the scripts to run in containerised training environments. It was necessary in the case of the RNN to provide a `requirements.txt` file as the default AWS Sagemaker environment does not come preconfigured with the required `torchtext` library used for preprocessing the files into batch loaders for use with the training loop.

##### Hyperparameter tuning

The `running_notebook.ipynb` file was modified to work with the Sagemaker enfironment by considering a small range of four hyperparameters

##### Profiling Results


## Evaluation


### Accuracy vs Balanced Accuracy


### Metric Comparison


## Deployment


### IAM


### Lambda

### Concurrency & Autoscaling


## Discussion




## Conclusion