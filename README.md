# UNSPSC Classification

This project prepares a classifier for the United Nations Standard Product and Services Classification system using the SageMaker Ecosystem. It is completed in partial satisfaction of the Udacity AWS Machine Learning Engineer NanoDegree. 

## TL:DR


## Problem Definition


### What is UNSPSC


### Purpose of Classification


### Available Data 


### Performance Metrics


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


### Initial Local Implementation

#### Local Model Development 

##### The Need for Data Augmentation

#####


## Classification Methodology


### HuggingFace Transformers




### Class Imbalance & NLP Data Augmentation


#### Class Imbalance in the Training Set





## Comparison Baseline


## Hyperparameter Tuning


## Optimal Model Performance


## AWS Environment


### Running Notebook Configuration


### Training Instance Requirements


### IAM Roles 


### Model Deployment


#### Instance Size for Deployment


#### Endpoint Concurrency


#### AWS Lambda 


### 