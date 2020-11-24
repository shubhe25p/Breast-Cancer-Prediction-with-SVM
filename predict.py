#!/usr/bin/env python
# coding: utf-8


# # SVMs and Kernels


# We will import SVM classifier class (SVC) as well some other packages you will use.

# In[1]:


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.graph_objs as go


# Let's turn off the scientific notation for floating point numbers.

# In[2]:


np.set_printoptions(suppress=True)


# ## Loading and examining the data

# We will load our data from a CSV file and put it in a pandas an object of the `DataFrame` class.
# 
# This dataset is the breast cancer Wisconsin (diagnostic) dataset which contains 30 different features computed from a images of a fine needle aspirate (FNA) of breast masses for 569 patients with each example labeled as being a _benign_ or _malignant_ mass.
# 
# * This was taken and modified from the Machine Learning dataset repository of School of Information and Computer Science of University of California Irvine (UCI):
#  
# > _Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science._

# In[13]:


df = pd.read_csv('data_svms_and_kernels.csv')


# Let's take a look at the data:

# In[4]:


df


# To do that we first need to extract our data from the dataframe in NumPy arrays. We use `LabelEncoder` from scikit-learn to transform labels into $\{-1,+1\}$:

# In[5]:


X = df.drop('Label', axis=1).to_numpy()
y_text = df['Label'].to_numpy()
y = (2 * LabelEncoder().fit_transform(y_text)) - 1


# Let's check `X`, `y_text` and `y`:

# In[6]:


X


# In[7]:


X.shape


# Let's do the same thing for `y_text`:

# In[8]:


y_text


# ...and for shape of `y_text`:

# In[9]:


y_text.shape


# Finally, let's check `y`:

# In[10]:


y


# In[11]:


y.shape


# Let's also do a scatter plot of our data:

# In[12]:


points_colorscale = [
                     [0.0, 'rgb(239, 85, 59)'],
                     [1.0, 'rgb(99, 110, 250)'],
                    ]

points = go.Scatter(
                    x=df['Feature 1'],
                    y=df['Feature 2'],
                    mode='markers',
                    marker=dict(color=y,
                                colorscale=points_colorscale)
                   )
layout = go.Layout(
                   xaxis=dict(range=[-1.05, 1.05]),
                   yaxis=dict(range=[-1.05, 1.05])
                  )

fig = go.Figure(data=[points], layout=layout)
fig.show()


# ## Splitting data

# Let's split our data into training, validation and test sets. Let's use 60% for training, 20% for validation and  20% for test data.

# In[ ]:


(X_train, X_vt, y_train, y_vt) = train_test_split(X, y, test_size=0.4, random_state=0)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_vt, y_vt, test_size=0.5, random_state=0)


# ## Building and visualizing a SVM

# Finally, let's build our SVM. We will use the `SVC` class from scikit-learn. For now, we are not using kernels, so we should set the `kernel` argument of `SVC` to `'linear'`. We don't need to specify any other parameters for now. You can find the documentation for `SVC` here:
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# 
# So, make an object of class `SVC` and assign it to the name `svm`:

# In[ ]:


svm=SVC(kernel='rbf')


# Now, fit `svm` to `X_train` and `y_train`:

# In[ ]:


### begin your code here (1 line).
svm.fit(X_train,y_train)
### end your code here.


# You will get a summary for the model:
# 
# > SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# >   decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
# >   kernel='rbf', max_iter=-1, probability=False, random_state=None,
# >   shrinking=True, tol=0.001, verbose=False)
# 
# * You may also get a warning because you have not explicitly set gamma and the default setting for that is going to change in newer versions of scikit-learn. Don't worry about that warniong.
# 
# Let's visualize the decision surface our `svm` with its supprt vectors:

# In[ ]:


decision_colorscale = [
                       [0.0, 'rgb(239,  85,  59)'],
                       [0.5, 'rgb(  0,   0,   0)'],
                       [1.0, 'rgb( 99, 110, 250)']
                      ]

detail_steps = 100

(x_vis_0_min, x_vis_1_min) = (-1.05, -1.05) #X_train.min(axis=0)
(x_vis_0_max, x_vis_1_max) = ( 1.05,  1.05) #X_train.max(axis=0)

x_vis_0_range = np.linspace(x_vis_0_min, x_vis_0_max, detail_steps)
x_vis_1_range = np.linspace(x_vis_1_min, x_vis_1_max, detail_steps)

(XX_vis_0, XX_vis_1) = np.meshgrid(x_vis_0_range, x_vis_0_range)

X_vis = np.c_[XX_vis_0.reshape(-1), XX_vis_1.reshape(-1)]

YY_vis = svm.decision_function(X_vis).reshape(XX_vis_0.shape)

points = go.Scatter(
                    x=df['Feature 1'],
                    y=df['Feature 2'],
                    mode='markers',
                    marker=dict(
                                color=y,
                                colorscale=points_colorscale),
                    showlegend=False
                   )
SVs = svm.support_vectors_
support_vectors = go.Scatter(
                             x=SVs[:, 0],
                             y=SVs[:, 1],
                             mode='markers',
                             marker=dict(
                                         size=15,
                                         color='black',
                                         opacity = 0.1,
                                         colorscale=points_colorscale),
                             line=dict(dash='solid'),
                             showlegend=False
                            )

decision_surface = go.Contour(x=x_vis_0_range,
                              y=x_vis_1_range,
                              z=YY_vis,
                              contours_coloring='lines',
                              line_width=2,
                              contours=dict(
                                            start=0,
                                            end=0,
                                            size=1),
                              colorscale=decision_colorscale,
                              showscale=False
                             )

margins = go.Contour(x=x_vis_0_range,
                     y=x_vis_1_range,
                     z=YY_vis,
                     contours_coloring='lines',
                     line_width=2,
                     contours=dict(
                                   start=-1,
                                   end=1,
                                   size=2),
                     line=dict(dash='dash'),
                     colorscale=decision_colorscale,
                     showscale=False
                    )

fig2 = go.Figure(data=[margins, decision_surface, support_vectors, points], layout=layout)
fig2.show()


# The datapoints, the decision surface (which is a line here), the margins and the support vectors are shown in the plot.

# ## Kernels

# As you can see, the decision surface is underfiiting the data. Let's use a polynomial kernel. Define `svm_p2` to be an instance of class `SVC` but this time with arguments `kernel='poly'` and `degree=2` to define a degree-2 polynomial kernel:

# In[ ]:


### begin your code here (1 line).

### end your code here.


# ..and fit it to your training data:

# In[ ]:


### begin your code here (1 line).

### end your code here.


# You will get a summary of your model:
#     
# > SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# >   decision_function_shape='ovr', degree=2, gamma='auto_deprecated',
# >   kernel='poly', max_iter=-1, probability=False, random_state=None,
# >   shrinking=True, tol=0.001, verbose=False)
# 
# Now, let's visualize this model:

# In[ ]:


YY_vis_p2 = svm_p2.decision_function(X_vis).reshape(XX_vis_0.shape)

SVs_p2 = svm_p2.support_vectors_
support_vectors_p2 = go.Scatter(
                                x=SVs_p2[:, 0],
                                y=SVs_p2[:, 1],
                                mode='markers',
                                marker=dict(
                                            size=15,
                                            color='black',
                                            opacity = 0.1,
                                            colorscale=points_colorscale),
                                line=dict(dash='solid'),
                                showlegend=False
                               )

decision_surface_p2 = go.Contour(x=x_vis_0_range,
                                 y=x_vis_1_range,
                                 z=YY_vis_p2,
                                 contours_coloring='lines',
                                 line_width=2,
                                 contours=dict(
                                               start=0,
                                               end=0,
                                               size=1),
                                 colorscale=decision_colorscale,
                                 showscale=False
                                )

margins_p2 = go.Contour(x=x_vis_0_range,
                        y=x_vis_1_range,
                        z=YY_vis_p2,
                        contours_coloring='lines',
                        line_width=2,
                        contours=dict(
                                      start=-1,
                                      end=1,
                                      size=2),
                        line=dict(dash='dash'),
                        colorscale=decision_colorscale,
                        showscale=False
                       )

fig3 = go.Figure(data=[margins_p2, decision_surface_p2, support_vectors_p2, points], layout=layout)
fig3.show()


# Looks much better. But let's try a degree 3 model. Define `svm_p3` like `svm_p2` but with `degree=3` this time:

# In[ ]:


### begin your code here (1 line).

### end your code here.


# Next, fit your `svm_p3` model to the training data:

# In[ ]:


### begin your code here (1 line).

### end your code here.


# Your model summary will be:
# 
# > SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# >   decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
# >   kernel='poly', max_iter=-1, probability=False, random_state=None,
# >   shrinking=True, tol=0.001, verbose=False)
# 
# Let's visualize `svm_p3`:

# In[ ]:


YY_vis_p3 = svm_p3.decision_function(X_vis).reshape(XX_vis_0.shape)

SVs_p3 = svm_p3.support_vectors_
support_vectors_p3 = go.Scatter(
                                x=SVs_p3[:, 0],
                                y=SVs_p3[:, 1],
                                mode='markers',
                                marker=dict(
                                            size=15,
                                            color='black',
                                            opacity = 0.1,
                                            colorscale=points_colorscale),
                                line=dict(dash='solid'),
                                showlegend=False
                               )

decision_surface_p3 = go.Contour(x=x_vis_0_range,
                                 y=x_vis_1_range,
                                 z=YY_vis_p3,
                                 contours_coloring='lines',
                                 line_width=2,
                                 contours=dict(
                                               start=0,
                                               end=0,
                                               size=1),
                                 colorscale=decision_colorscale,
                                 showscale=False
                                )

margins_p3 = go.Contour(x=x_vis_0_range,
                        y=x_vis_1_range,
                        z=YY_vis_p3,
                        contours_coloring='lines',
                        line_width=2,
                        contours=dict(
                                      start=-1,
                                      end=1,
                                      size=2),
                        line=dict(dash='dash'),
                        colorscale=decision_colorscale,
                        showscale=False
                       )

fig4 = go.Figure(data=[margins_p3, decision_surface_p3, support_vectors_p3, points], layout=layout)
fig4.show()


# Let's try a RBF (Radial Basis Function) kernel as well. RBFs are the default kernel for scikit-learn's SVC. So build a model `svm_r` with either `kernel=rbf` argument setting or just skip the `kernel` (also the `degree` argument is uselss here, since we are not using a polynomial kernel, so just skip that): 

# In[ ]:


### begin your code here (1 line).

### end your code here.


# Fit your `svm_r` model to the training data as well:

# In[ ]:


### begin your code here (1 line).

### end your code here.


# This will be the parameter summary:
# 
# > SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# >   decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
# >   kernel='rbf', max_iter=-1, probability=False, random_state=None,
# >   shrinking=True, tol=0.001, verbose=False)
# 
# We will visualize this model as well:

# In[ ]:


YY_vis_r = svm_r.decision_function(X_vis).reshape(XX_vis_0.shape)

SVs_r = svm_r.support_vectors_
support_vectors_r = go.Scatter(
                                x=SVs_r[:, 0],
                                y=SVs_r[:, 1],
                                mode='markers',
                                marker=dict(
                                            size=15,
                                            color='black',
                                            opacity = 0.1,
                                            colorscale=points_colorscale),
                                line=dict(dash='solid'),
                                showlegend=False
                               )

decision_surface_r = go.Contour(x=x_vis_0_range,
                                 y=x_vis_1_range,
                                 z=YY_vis_r,
                                 contours_coloring='lines',
                                 line_width=2,
                                 contours=dict(
                                               start=0,
                                               end=0,
                                               size=1),
                                 colorscale=decision_colorscale,
                                 showscale=False
                                )

margins_r = go.Contour(x=x_vis_0_range,
                        y=x_vis_1_range,
                        z=YY_vis_r,
                        contours_coloring='lines',
                        line_width=2,
                        contours=dict(
                                      start=-1,
                                      end=1,
                                      size=2),
                        line=dict(dash='dash'),
                        colorscale=decision_colorscale,
                        showscale=False
                       )

fig5 = go.Figure(data=[margins_r, decision_surface_r, support_vectors_r, points], layout=layout)
fig5.show()


# ## Model selection

# Let's pick the best model then. We will use the validation data for that. Let's predict using `svm` and `X_train` and assign it the name `yhat_train`. Also, predict `X_validation` and assign it the name `yhat_validation` (. The closeness of the accuracy of predictions on these two datasets will be helpful to us):

# In[ ]:


### begin your code here (2 lines).


### end your code here.


# Let's measure the accuracy:

# In[ ]:


print(accuracy_score(yhat_train, y_train), accuracy_score(yhat_validation, y_validation))


# We got ??.??% and ??.?%.
# 
# Let's repeat thge predictions for `svm_p2` and put the results in `yhat_train_p2` and `yhat_validation_p2`:

# In[ ]:


### begin your code here (2 lines).


### end your code here.


# We can calculate the accuracies:

# In[ ]:


print(accuracy_score(yhat_train_p2, y_train), accuracy_score(yhat_validation_p2, y_validation))


# ??.??% and ??.??%.
# 
# Let's try predicting with `svm_p3` and put it in `yhat_train_p3` and `yhat_validation_p3`:

# In[ ]:


### begin your code here (2 lines).


### end your code here.


# Now, if we predict the accuracy on these:

# In[ ]:


print(accuracy_score(yhat_train_p3, y_train), accuracy_score(yhat_validation_p3, y_validation))


# The accuracy is ??.??% and ??.??%.
# 
# Finally, let's predict `yhat_train_r` and `yhat_validatin_r` using `svm_r`:

# In[ ]:


### begin your code here (2 lines).


### end your code here.


# We can measure the accuracy of the SVM with RBF kernel:

# In[ ]:


print(accuracy_score(yhat_train_r, y_train), accuracy_score(yhat_validation_r, y_validation))


# We got ??.??% and ??.??% when using the RBF kernel.
# 
# From all these number we can see that the RBF model works best as the accuracy on validation data is high and also the gap between the accuracy on training and validation data is not big. We can further tune the generalization power of our model by tuning the argument `C` of `SVC` which is the inverse of a regularization coefficient. We won't do that here now though. 

# ## Final assessment

# Finally, let's check accuracy on the test data to get a final performance number. Predict `yhat_test_r` from `X_test` on `svm_r`:

# In[ ]:


### begin your code here (1 line).

### end your code here.
accuracy_score(yhat_test_r, y_test)


# ??.??%. We have good performance of test data.
