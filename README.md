# OrdinalClassifier
Ordinal multiclass strategy.

This classifier is based on a "Simple Approach to Ordinal Classification"
by Frank and Hall as oultined in this paper.

https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf

## Adapted Abstract:

Machine learning methods for classification problems commonly assume
that the class values are unordered. However, in many practical applications
the class values do exhibit a natural order—for example, when learning how to grade
or when classifying sentiment (disagree < neutral < agree), temperatures (cold <
warm < hot).  The standard approach to ordinal classification converts the class
value into a numeric quantity and applies a regression learner to the transformed
data, translating the output back into a discrete class value in a post-processing
step. A disadvantage of this method is that it can only be applied in conjunction with a
regression scheme.

The method enables standard classification algorithms to make use of ordering information
in class attributes.   The authors have shown in their work this classifier
outperforms the naive state.

The method utilizes a 'simple trick' to allow the underlying classifiers to take
advantage of the ordinal class information.   First, the data is tranformed from a k-class
ordinal problem to a k-1 (sklearn nomenclature: n_classes-1) binary class problem. 
Training starts by deriving new datasets from the original dataset, one for each of the k-1 
binary class attributes.

Ordinal attribute A* with ordered values V1, V2, ..., Vk into k-1 binary attrbutes,
one for each of the original attribute's first K-1 values.  The ith binary attribute
represents the test A* > Vi.

Initial and prelminary testing does seem to show improved classification results as measured by precision
and f1 scores.  Also, as raised by Muhammad and Chistopher Coffee on Muhammad's medium post, there 
was inital concern about binary probabilities not summing to 1.   For this, I used the same method 
Sklearn uses in the OnevsRest classifier which normalizes the binary predict_prob to 1 

https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c

Also, as raised by Coffee, this method does seem to improve the lowest class in the ordered class (eg. "cold"
in cold<warm<hot) more so than the other classes.  So, if the positive class is "hot," consideration should be 
to reverse the classses in order to improve scores on "hot."

Classsifier has support for custom ordering and reversed ordering of classes.  Reversal does not seem to matter 
much but order does.

Adapted from https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
Adapted by Lee Prevost

# Testing (in progress)

[Cross validated test results on sklean diabetes dataset.  See evaluate.py for code](/ordinal_cv_test.md)
