
# NewsBERT: A DistilBert-based News Classifier with Flask and Docker
```
├── static
│   ├──css 
│         ├── style.css         >CSS file
├── templates   
│            ├──index.html      >HTML file
├── .gitattributes              > to track the  large saved  model files using git-lfs
├── app.py                     >flask  application main file
├── dockerfile                 >dockerFile used to automate the process of building a Docker image
├── news_classification.ipynb  >news classification model trained using  DistilBERT
├── README.md    
├── saved_model                >Stores the best_model
└── requirements.txt           > Stores the information about all the libraries, modules, and packages in itself that are used while developing the project
```
## Project Overview
NewsBERT is a comprehensive application that classifies news articles into different categories based on their headlines and short descriptions. The project leverages a fine-tuned DistilBert model for the classification task, a Flask application for the frontend (rendering HTML) and backend, and Docker for deployment.

## Dataset and Preprocessing
The dataset used for this project is sourced from Kaggle. It contains approximately 210k+ news headlines from HuffPost, spanning from 2012 to 2022. The dataset includes the following columns: link, headline, category, short_description, authors, and date.

### Exploratory Data Analysis
The initial exploratory data analysis involved removing duplicate entries and reducing the number of categories from 42 to 26 by merging similar categories. Data visualization was performed to display the count of news articles belonging to each category, the average length of headlines, and the average length of descriptions.

### Data Preprocessing
The headline and short_description columns were preprocessed by removing punctuations, single letters, special symbols, and lemmatizing words. This step was crucial to prepare the text data for the DistilBert model. Further analysis included tokenizing text, counting top words by category, and generating word clouds to better understand the data and the distribution of words across different categories.
The data preprocessing involved encoding category labels, splitting the data into train, test, and validation datasets, and tokenizing the text. The tokenization was performed using the DistilBert tokenizer, which converts text into numerical representations that can be fed into a machine learning model.


## Model Building and Training
The model is built using a transformer layer, specifically a pre-trained DistilBERT model. The model is compiled with an optimizer that includes weight decay for regularization. The F1 score is used as a metric for multi-class classification.

The model is then trained using the training and validation datasets. The learning rate is reduced when the validation loss stops improving, using the ReduceLROnPlateau callback. The model is trained for a specified number of epochs, and the training history is saved for further analysis.

## Model Evaluation
After training, the best model is loaded and used to make predictions on the test set. The class with the highest predicted probability is selected as the predicted class for each example.
The model's performance on the test set is as follows:
- Accuracy: 71.22%
- Precision (macro-averaged): 64.15%
- Recall (macro-averaged): 61.16%
- F1-score (macro-averaged): 61.85%



## Flask Application

The Flask application serves as the backbone of this project, providing both frontend and backend functionalities:

- Frontend: The frontend is designed using HTML and CSS, which are rendered through Flask. It provides a user-friendly interface where users can enter the title and headline of a news article. The application then uses the DistilBert model to predict the news category and displays the result.

- Backend: The backend of the application handles URL routing. When a user enters a title and headline, the Flask application routes the request to the appropriate function. The backend also includes a function for making predictions using the saved DistilBert model.

This Flask app allows users to submit input requests either through an API or a web application form.
## Docker Deployment
The application and associated files are containerized using Docker, which ensures that it runs consistently across different environments. Docker packages up the application with all of the parts it needs, including the libraries and other dependencies, and ships it all out as one package.
### Build Docker Image
You can build the docker image manually by cloning the Git repo.
```
$ git clone https://github.com/Parthkh28/DataScience.git
$ docker build -t news_classification .
```
### Download Precreated Image
Alternatively, you can download the existing image from DockerHub.
```
$ docker pull parthkh28/news_classification
```
### Run the Container
Create a container from the image.
```
$ docker run --name my-container  -d -p 8000:8000 news_classification
```
In the above example, we are running a docker container with the name my-container in detached mode and mapping port 8000 of the host to port 8000 of the container. The image we are using is news_classification.

Now visit http://localhost:8000 to interact with the application.
