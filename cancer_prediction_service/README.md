

## Installation

```
pip install django
```

## Usage

```
bat post http://127.0.0.1:8000/cancer_predict/predict/ cancer_features="10,10,10,8,6,1,8,9,1;6,2,1,1,1,1,7,1,1"
```

## Implementation

```
django-admin startproject cancer_prediction_service

python manage.py startapp cancer_predict
```