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

> Edible sandwiches for consumption on or off the premises, coffee, coffee substitutes, tea, cocoa, sugar, honey, rice, tapioca, flour, breakfast cereals, processed cereals, cereal-based snack foods and ready to eat cereal-derived food bars; bread, biscuits, cakes, pastries, dairy-based shakes, soft-serve ice cream, ice milk and frozen yoghurt; yeast, baking powder, salt, mustard, pepper, sauces, spices, seasonings and ice.
> See https://search.ipaustralia.gov.au/trademarks/search/view/957064 Class 30.

This is a partial goods and services item description for the "I'm lovin' it" mark used by McDonald's Restaurants in Australia. Let us consider the definition provided by WIPO for Nice Class 30:

>Coffee, tea, cocoa and substitutes therefor; rice, pasta and noodles; tapioca and sago; flour and preparations made from cereals; bread, pastries and confectionery; chocolate; ice cream, sorbets and other edible ices; sugar, honey, treacle; yeast, baking-powder; salt, seasonings, spices, preserved herbs; vinegar, sauces and other condiments; ice (frozen water).
> See https://www.wipo.int/classifications/nice/nclpub/en/fr/?basic_numbers=show&class_number=30

We should pay attention though that this trade mark also has items in Nice Class 29. WIPO says that Nice class 29 is:

>Meat, fish, poultry and game; meat extracts; preserved, frozen, dried and cooked fruits and vegetables; jellies, jams, compotes; eggs; milk, cheese, butter, yogurt and other milk products; oils and fats for food.
>See https://www.wipo.int/classifications/nice/nclpub/en/fr/?basic_numbers=show&class_number=29

While the items that are covered between the two classes are clearly different, the short descriptions make it very difficult to identify what a relative rise in the number of class 30 trade marks would mean for the food branding sector over an increase in the number of class 29 trade marks. When providing high level reporting on brand landscapes, the use of Nice classification is in practice highly difficult.



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

When building the classifier, we need to be aware of how we would measure the success of the project. 

## Analysis

### Data Quality


### Class Imbalance


### Solution Options: RNN

#### Benchmark Model


### Solution Options: Transformers

#### HuggingFace

#### DistilBERT


## Methodology 


### Download & Preprocessing

#### Data Augmentation Using Randon Synonyms


#### String Length Truncation


### Examples Used to Iterate


### Initial Local Implementation

#### Local Model Development 

##### The Need for Data Augmentation

#### Modification for AWS Environment


##### Hyperparameter tuning


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