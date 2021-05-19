# Capstone 3: Plant Village Image Detection
## Data and EDA:
In this project, a model was built to determine if a plant has a certain disease as well as if it is healthy. The images were used from the Kaggle repository: https://www.kaggle.com/emmarex/plantdisease. The images are broken into several types. The prodominant plant in the images is the tomato plant and it has several different diseases that it can have. Tomato Curl Virus images make up a large portion of the dataset. The Peppers and Potatoes do have the least number of images. having such a small number of images may make it more difficult to correctly model these images.

![alt text](https://github.com/michaellaephillips/Capstone3/blob/master/amounts_of_photos.png?raw=true)

Looking at the images for the Totmato curl virus (below), there is a large variation in the color and shape of the leaves. That being said every image is very similar in the style is has been taken in: light background with a single leaf. This was the case for every image in this dataset. This could benefit or hinder the model as there is less variation in the background. If you were to test a model solely built on the images one could expect that if an image of a leaf on a different surface or taken right off the plant may perform poorly. 

![alt text](https://github.com/michaellaephillips/Capstone3/blob/master/example_photos.png?raw=true)

## Supervised Learning:
I chose to build my model using a LeNet achitecture with the Keras library. I chose to use two drop out statements in the model to help prevent over fitting and only 15 epochs. Even so, the model's training appears to have done well. The training accuracy and validation accuracy both approach 100%, which is great. The weighted average accuracy was 0.95 and the weighted average f-1 score was 0.95. The loss for both the validation and the training appraoches 0. 

![alt text](https://github.com/michaellaephillips/Capstone3/blob/master/Training%20and%20Validation%20accuracy.png?raw=true)

![alt text](https://github.com/michaellaephillips/Capstone3/blob/master/Training%20and%20Validation%20loss.png?raw=true)

On testing five images not used for training or validation, I got a mix of correctly identificed and misidentifed plants. Looking at the distribution of images and my method for splitting into validation and training sets, I believe I need to use something different to help to evenly distribute more of the images such as using imbalanced-learn. Alternatively the model could also be imporoved by providing images from different sources. Having the same format for the images is not ideal. By adding in more images collected from different sources, background, lighting, etc would help to improve the classification model. Lastly, I chose not to distort the images in my model. This could also be added to add more variability to the dataset. 

## The Application. 
I followed along with denistanjingyu's repository (https://github.com/denistanjingyu/Image-Classification-Web-App-using-PyTorch-and-Streamlit/blob/main/streamlit_ui.py). In following this repository, I was able to build my own web app with streamlit. 
