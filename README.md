# Farmalytics
---
This is my final year college academic project.

Farming is a tiresome process and it has a huge labor cost and to optimize it, technical tools and methods should be implemented so make it productive and efficient.

Our Project Farmalytics will be an API or a model that will allow the users to analyze the unwanted plants and weeds growing near the actual crop they have harvested when farming will be done using Automation in farming work or using robots. There are specific characteristics that determine if the crop is a weed or not. Based on those characteristics we will develop a machine learning model that will use certain algorithms to help drones and machine vehicles to recognize weed and plants. If this model is further developed with the help of robotics specific fertilizer can also be spread through the drone according to the crop and to cut the weed. If this module is implemented into the Robots used for farming, weeds can be simultaneously cut while detection and thus reduce the labor cost.

Easily this module can be implemented into the personal devices also so that the local users can get the idea of the plant and if they can afford buying the robotic device that can implement by connecting it to their personal device this model can easily help the farmers with the removal of the weeds. Lowering the cost and the labour work has been the prime factors supporting the benefits of this module.

---
### Requirements 
* Python
* TensorFlow
* Keras
* open CV
* Scikit-learn  
* pickle
* pillow

---
### Data Pre-processing
* Data Augmentation
For making dataset large augmentation is done.
  * foctors: rescale, Crop, resize, Flip image horizontally and vertically.
* Image resize
### Implementation
**Training Model** : We are using a classification approach for differentiating the images. We are using a Sequential Model to train our augmented dataset which is converted into a numpy array and dumping it using the pickle library with batch size of 32 and with 25 epochs which resulted the accuracy of approx 0.9.

The image following here is outout of the script

 ![picture alt](https://serving.photos.photobox.com/602206617747f08a810e94deefedb3aee15877f772818680d4cd43668f7a6fd8427e09fb.jpg "Image")

---
You can use this script for any image classification.
