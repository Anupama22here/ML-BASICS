#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
from keras.models import Sequential
from sklearn.metrics import confusion_matrix 
from keras.layers import Dense,Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[48]:


np.random.seed(0)


# In[49]:


from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[56]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# visualizing

# In[51]:


classes=10
f,ax=plt.subplots(1,classes,figsize=(20,20))
for i in range(0,classes):
    s=x_train[y_train==i][0]
    ax[i].imshow(s,cmap='gray') #rgb data
    ax[i].set_title("Label: {}".format(i),fontsize=16)
    ax[i].axis('off')


# Data Preparation

# In[52]:


for i in range(10):
    print(y_train[i])


# In[53]:


original_y_train = y_train.copy()


# In[54]:


#vector for the numbers-hot encoding
y_train=keras.utils.to_categorical(y_train,classes)
y_test=keras.utils.to_categorical(y_test,classes)
for i in range(10):
    print(y_train[i])


# Preparation

# In[58]:


#normalize data
x_train=x_train/255.0
x_test=x_test/255.0


# In[62]:


#reshape data
x_train=x_train.reshape(x_train.shape[0],-1)
x_testn=x_test.reshape(x_test.shape[0],-1)
print(x_test.shape)
print(x_train.shape)


# Building model

# In[63]:


model=Sequential()


# In[67]:


model.add(Dense(units=128,input_shape=(784,),activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# Training

# In[68]:


batch_size=512
epochs=10
model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs)


# Evaluate

# In[71]:


x_test_flat = x_test.reshape(-1, 28 * 28)  # Reshape to (num_samples, 784)
test_loss,test_acc=model.evaluate(x_test_flat,y_test)
print("Test Loss: {},Test Accuracy: {}".format(test_loss,test_acc))


# In[73]:


x_test_flat = x_test.reshape(-1, 28 * 28)  # Reshape to (num_samples, 784) 
y_pred = model.predict(x_test_flat)
y_pred_c = np.argmax(y_pred, axis=1)
print(y_pred_c)


# In[74]:


random_idx=np.random.choice(len(x_test))
x_sample=x_test[random_idx]
y_true=np.argmax(y_test,axis=1)
y_sample_true=y_true[random_idx]
y_sample_pred_c=y_pred_c[random_idx]

plt.title("Predicted: {}, True: {}".format(y_sample_pred_c,y_sample_true),fontsize=16)
plt.imshow(x_sample.reshape(28,28),cmap='gray')


# Confudion matrix

# In[82]:


y_test_labels = np.argmax(y_test, axis=1)
y_pred_c = np.argmax(y_pred, axis=1)  # Use this if y_pred is in one-hot format
# If you converted y_test to class labels
confusion_mtx = confusion_matrix(y_test_labels, y_pred_c)
fiq,ax=plt.subplots(figsize=(15,15))
ax=sns.heatmap(confusion_mtx,annot=True,fmt='d',ax=ax,cmap="Blues")
ax.set_xlabel('Predcted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix')


# In[ ]:




