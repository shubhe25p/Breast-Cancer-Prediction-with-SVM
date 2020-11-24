# Breast-Cancer-Prediction-with-SVM

We have used an SVM with multiple kernel method to check which works best for our data.

# Output

| Method | Accuracy |
| ------ | ------ |
| Linear | 94.82% |
| Poly-2 | 46.55%|
| Poly-3 | 51.72%|
| RBF | 98.28% |

# Dataset

We will load our data from a CSV file and put it in a pandas an object of the `DataFrame` class.

This dataset is the breast cancer Wisconsin (diagnostic) dataset which contains 30 different features computed from a images of a fine needle aspirate (FNA) of breast masses for 569 patients with each example labeled as being a _benign_ or _malignant_ mass.

* This was taken and modified from the Machine Learning dataset repository of School of Information and Computer Science of University of California Irvine (UCI):
 
> _Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science._

# Steps

* Finally, let's build our SVM. We will use the `SVC` class from scikit-learn. For now, we are not using kernels, so we should set the `kernel` argument of `SVC` to `'linear'`. We don't need to specify any other parameters for now.

* We saw, the decision surface is underfiiting the data. Let's use a polynomial kernel with polynomial degree 2 and 3.

* Let's try a RBF (Radial Basis Function) kernel as well. RBFs are the default kernel for scikit-learn's SVC. 

* Let's pick the best model then. We will use the validation data for that.

# Contact

Feel Free to [contact](mailto:shubhpachchigar@gmail.com) me in case of any doubt!



