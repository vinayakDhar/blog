---
layout: post
title:  "Lets talk about Linear Regression?"
date:   2018-04-07 01:00:56 -0400
categories: machine learning
---

### Getting Started

Probably a rhetorical question, but why in the world would anyone want to talk about Linear Regression? It is like the most simplest of topics
in machine learning, most courses/tutorials use it as a stepping stone to bigger things. While I would agree it is a stepping stone, but it is
a *wonderful* stepping stone to explain the basic machine learning pipeline to a beginner.

So as an example exercise I will pick up a real data set from the wild and try some linear regression on it and see how does it perform. The python notebook is available at the [Github Repo][repo] & the data is sourced from the [Indian Census][censusindia.gov.in]

### So What is Linear Regression?
> In statistics, linear regression is a **linear approach** for modelling the **relationship** between a scalar **dependent variable** y and one or more explanatory variables (or **independent variables**) denoted X.  
> -- Wikipedia

For someone well versed in statistics that does make sense, however for the non statistically inclined there are quite a few terms here which might need some simplification.

**linear approach** in simplest mathematical term means f(x) = y, where a function takes an input variable x and outputs a value y.

**relationship** is the function f(x) being used for determining the value of y

**dependent variable** is the value which is being predicted by using the function

**independent variables** the value(s) which will determine the outcome

A simple example of a linear model is area of a square, where y (Area) & x (Length of Side) are the variables considered. The function f(x) to calculate the area of the square is represented by **_f(x) = x<sup>2</sup>_** and **_y = f(x)_**. Here y is a dependent variable whose value is dependent on the independent variable x which can be changed to reflect a proportional change in y (_2*x using differentiation_).

It is possible that multiple independent variables are also involved in the relationship to determine the value of y (area of rectangle?), hence the input x is represented as a vector (x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, .., x<sub>n</sub>) where each x<sub>i</sub> represents the i<sup>th</sup> independent variable in the relationship.

If I know the formulae the what is the use of machine learning? the idea here is to capture the inherent relationship/coefficient/factor/weightage by which each independent variable (x) affects the value of the dependent variable (y). To represent such a relationship the following equation can be used.

> **y = x<sub>1</sub> * w<sub>1</sub> + x<sub>2</sub> * w<sub>2</sub> + ... + x<sub>n</sub> * w<sub>n</sub>**

Machine learning helps us in determining the most probable W<sub>i</sub> for the x<sub>i</sub>, while overcoming bias & variance of the data provided to the machine learning model. I will cover bias, variance & model evaluation in another post in detail as that is huge topic in itself.

So how does linear regression determine the correct value for the coefficient/weights? for linear regression to work it needs the following things
- Dataset of label values (x, y) to train on
- A loss function to determine difference between its predicted value & actual value in the datasets

If I am providing the dataset then how can the model compare its performance on the dataset it is trained on? for that reason the dataset is shuffled and divided into usually 3 subsets
- Training Set (60-80%) (when less than million records)
- Test Set (20-40%) (used for comparing the trained coefficient performance)
- Validation Set (20%) (this is used as control data set for tuning the model performance, lets leave this out for now)

Linear regression model is trained on the Training Set then the performance is recorded using the Test set against the predicted vs known values, the residual value (r = y<sub>predict</sub> - y<sub>test</sub>) is very important in helping us to determine the performance of the model.

**y<sub>predict</sub>** is the predicted value given a X<sub>Test</sub> value to the trained model

**y<sub>test</sub>** is the value already available to us as part of the data setting

### Behind the Scene

The linear regression model first has to come up with relationship which can be used to determine the coefficients, so how does it do that? here we have to understand the mathematics behind the scenes.

#### Representing the relationship
Using vector notation,
> f(x) = W<sup>T</sup> X

where W is the Weights/Coefficients matrix representing the weights for each independent variable in matrix X. The next step is to use a loss function to determine the quality of the model output vs the actual value so that an effort can be made towards bridging the gap.

#### Loss Function
Loss function is represented as,
> **L(W) = 1/l * \|\|XW - y\|\|<sup>2</sup>**

Break down the formulae step by step
- **l** is the size of training set
- **XW** the product of the coefficient & independent variable
- **y** is the actual value in the dataset available fot the record of the independent variable(s)
- **\|\|XW - y\|\|<sup>2</sup>** is the mean squared error summed for each input in the dataset

#### Optimizing the loss function (Gradient Decent)

Now that we have a way to represent the relationship & the loss function, we have to somehow find out the best solution for the value of W in the relationship which explains our data best (less bias & variance for the prediction vs the known value). So going back to the loss function we see that it is defined in terms of W as in L(W) , now the problem become a minimization problem where min L(w) will determine the best fit model coefficients.

> The goal is to **min L(W)** to get the best coefficient or weights for X

For the mathematical genius out there it the problem seems simple to solve using matrix inversion like below,
> w = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y

However the doing a matrix inversion for high dimensional data is [pretty hard][matrix-complexity] and computationally expensive. So are we out of luck here or there is some kind of human intuition we can apply for the machine learning (after all humans are pretty great at abstraction).

> Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point. If instead one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent.
-- Wikipedia

Looks like a pretty heavy definition, lets break it down some important words.

**first order** simply means we have predicates, quantifiers and variables defining a simple relation like y = x<sup>2</sup>

**iterative** since we don't have a exact formulae, the movement towards a good minimized value will have to be done many times

**decent/ascent** if we plot a function onto a graph, its has a curve which can we used to determine its minima & maxima, depending on where we are on the curve we want to either decent or ascent it to reach the destination

The following image source from wikipedia illustrates, how over each iteration the value inches closer to the minima in the center.

![Graident Decent]({{ "/images/linearregression/512px-Gradient_descent.svg.png" | absolute_url }})

The next question obviously is how do we even know if we need to ascent or descent and by what value in the iteration, when do we stop?

It can be summed up by the following pseudo code
```
W[0] = initialize to small random value
while True:
  W[t] = W[t-1] - alpha * dL(W[t-1])
  if ||W[t] - W[t-1]|| < epsilon
    break

```

Lets define the variables

- **W[t]** is the value of W in the current iteration
- **W[t-1]** is the value of W in the last iteration
- **dL** partial derevative of the Loss function wrt to W which is 2/l X<sup>T</sup>(XW - y)
- **\|\|W[t] - W[t-1]\|\|** absolute difference between current & pervious W
- **epsilon** the minimum difference between the W in iterations for it to be worthwhile to continue (stop iterating when gain is very small)
- **alpha** is the learning rate or the step size taken every iteration to update the value (this is one hyper parameter to look into)


So now we have a way to minimize the loss function L(W) which in turn helps us determine the Weights W which are then combined with the independent variable X to help us finally create a model which determines the best relationship between X & Y.

There is a whole lot of things which can be talked about the hyperparameters such as alpha & epsilon, but that is a topic for a while other post.

### Putting it into Practice

So now that we understand what is linear regression and hopefully have got a basic understanding of machine learning, lets put it into practise using python & sklearn along with a reasonable dataset.

The complete python notebook is available at [Github Repo][repo].

{% highlight python %}
# load the dataset into a pandas dataframe
data = pd.read_csv("../data/India-Women-Stats-Population-2011.csv")
data.head()
{% endhighlight %}

![Data Head 1]({{ "/images/linearregression/data-head.png" | absolute_url }})

After looking at the available data, I think why not try to create a linear model for determining the TFR or the total children per women given the age range of the woman in India.

For this we will subset out the data relevant to us and also create a new column in the dataset with TFR by using the formulae given in the code below. Now since the age range is a category we will also have to assign it a more machine digestible format (like 1,2,3, etc).

{% highlight python %}
# subset out columns which are of interest to us
df = data[['Area Name', 'Present Age', 'Total Women', 'Total Children Ever Born - Persons']].copy()

# remove records which not required (like national ave & all ages)
df = df[df['Area Name'] != 'INDIA'][df['Present Age'] != 'All Ages']

# create new columns for TFR
df['TFR'] = df['Total Children Ever Born - Persons'] / df['Total Women']

# one hot encode the age range from category to a numeric class label
age_cat_dict = {
 'Less than 15' : 1,
    '15-19' : 2,
    '20-24' : 3,
    '25-29' : 4,
    '30-34' : 5,
    '35-39' : 6,
    '40-44' : 7,
    '45-49' : 8,
    '50-54' : 9,
    '55-59' : 10,
    '60-64' : 11,
    '65-69' : 12,
    '70-74' : 13,
    '75-79' : 14,
    '80+' : 15
}

df['Present Age Cat'] = df['Present Age']
df['Present Age Cat'].replace(age_cat_dict, inplace = True)
{% endhighlight %}

![Data Head 1]({{ "/images/linearregression/data-head-2.png" | absolute_url }})

After the data has been manipulated to what we need it to look like, lets us create the training & test set for the linear regression model.

{% highlight python %}
# Generate the training & test dataset

ages_X = df['Present Age Cat'].values.reshape(-1, 1);

len_X = len(ages_X)
# Split into training & test data (when data is small 80:20 works out)
train_len_X = int(len_X * 0.8);
test_len_X = len_X - train_len_X
ages_X_train = ages_X[:train_len_X]
ages_X_test = ages_X[train_len_X:]

tfr_Y = df['TFR'].values
len_Y = len(tfr_Y)
# Split into training & test data (when data is small 80:20 works out)
train_len_Y = int(len_Y * 0.8);
test_len_Y = len_X - train_len_Y
tfr_Y_train = tfr_Y[:train_len_Y]
tfr_Y_test = tfr_Y[train_len_Y:]
{% endhighlight %}

Finally it is time to create a linear regression object and fit it against the training data, then evaluate the performace of the model using the reserved test dataset which was split as a 20% of the original dataset.

{% highlight python %}
# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(ages_X_train, tfr_Y_train)

# Time to make predictions on the test data (20% of the data set aside)
tfr_Y_pred = reg.predict(ages_X_test)
# The coefficients
print('Feature Coefficients (x1, x2, ... ): \n', reg.coef_)

# The mean squared error & R^2 Score
print("Mean squared error: %.2f"
      % mean_squared_error(tfr_Y_test, tfr_Y_pred))
print('(R^2) Variance score: %.2f' % r2_score(tfr_Y_test, tfr_Y_pred))
{% endhighlight %}

The output observed from the dataset is as follows
- Coefficients/Weightage value for the Age Range is determined as 0.30992102
- Mean Squared Error is 0.50
- Variance / R2 Score is 0.78

Well it means that we are able to explain the variability of the data in the test dataset 78% of the times successfully, there are other measures such as precision, recall & f1 scores which provide better measure for the quality of the model. For today we are happy that a simple model is 78% accurate in predicting values for us.

Let us draw a plot to see how the prediction fares against the actual inputs

{% highlight python %}
# Plot outputs as a scatter chart
plt.figure(figsize = (15, 5))
plt.scatter(ages_X_test, tfr_Y_test,  color='black')
plt.plot(ages_X_test, tfr_Y_pred, color='blue', linewidth=3)

plt.xlabel('Age Range')
plt.ylabel('TFR')

plt.xticks(range(1, len(age_cat_dict) + 1), list(age_cat_dict.keys()))
plt.yticks((range(7)))

plt.show()
{% endhighlight %}

![Regression Plot]({{ "/images/linearregression/regression.png" | absolute_url }})

Here the blue line is the regression model values predicted & the datapoints in black represent the actual values, it is clearly visible that a linear model is quite simplistic but is able to capture a general trend (78%) of the TFR value based on the Age Range.

Let try and generate some predictions.

{% highlight python %}
# perform predictions for all age ranges based on the model
for age in age_cat_dict.keys():
  age_input = [ [ age_cat_dict[age] ] ];
  pred_children = reg.predict(age_input)[0];
  print("Women in Age (%s) will likely have %.3f children" % (age, pred_children))

{% endhighlight %}


Women in Age (Less than 15) will likely have 0.582 children

Women in Age (15-19) will likely have 0.892 children

Women in Age (20-24) will likely have 1.202 children

Women in Age (25-29) will likely have 1.512 children

Women in Age (30-34) will likely have 1.822 children

Women in Age (35-39) will likely have 2.132 children

Women in Age (40-44) will likely have 2.442 children

Women in Age (45-49) will likely have 2.752 children

Women in Age (50-54) will likely have 3.062 children

Women in Age (55-59) will likely have 3.372 children

Women in Age (60-64) will likely have 3.681 children

Women in Age (65-69) will likely have 3.991 children

Women in Age (70-74) will likely have 4.301 children

Women in Age (75-79) will likely have 4.611 children

Women in Age (80+) will likely have 4.921 children

### Conclusion

Covered quite a lot of content in this post, however the objective was to explain machine learning & use linear regression as a building block for explaining the inner magic of machine learning, there are lots of in details which are skipped over in the post to keep things as simple as possible, however in my future posts I would be covering how tuning & evaluation works in machine learning.

[censusindia.gov.in]: http://www.censusindia.gov.in/datagov/F-01/2011-F01-0000-Rev4-MDDS.xls
[repo]: https://github.com/vinayakDhar/Machine-Learning/tree/master/LinearRegression
[matrix-complexity]: https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra
