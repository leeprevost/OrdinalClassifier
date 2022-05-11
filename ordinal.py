import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.utils.deprecation import deprecated
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import (
    _check_partial_fit_first_call,
    type_of_target
)
from sklearn.utils.metaestimators import _safe_split, available_if
from sklearn.utils.fixes import delayed
from sklearn.multiclass import (
    _fit_binary,

    _estimators_has
)
from joblib import Parallel




class OrdinalClassifier(
    MultiOutputMixin, ClassifierMixin, MetaEstimatorMixin, BaseEstimator
):
    """Ordinal multiclass strategy.

    This classifier is based on a "Simple Approach to Oridinal Classification"
    by Frank and Hall as oultined in this paper.

    https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
    Adapted Abstract:
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
    ordinal problem to a k-1 (n-1?) binary class problem. Training starts by deriving new datasets from
    the original dataset, one for each of the k-1 new binary class attributes.

    --------
    Ordinal attribute A* with ordered values V1, V2, ..., Vk into k-1 binary attrbutes,
    one for each of the original attribute's first K-1 values.  The ith binary attribute
    represents the test A* > Vi.
    --------

    @todo: should this stay in?  My starting point was to use OvR as basis.
    OrdinalClassifier can also be used for multilabel classification. To use
    this feature, provide an indicator matrix for the target `y` when calling
    `.fit`. In other words, the target labels should be formatted as a 2D
    binary (0/1) matrix, where [i, j] == 1 indicates the presence of label j
    in sample i. This estimator uses the binary relevance method to perform
    multilabel classification, which involves training one binary classifier
    independently for each label.
    Read more in the :ref:`User Guide <ovr_classification>`.
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and one of
        :term:`decision_function` or :term:`predict_proba`.
    n_jobs : int, default=None
        The number of jobs to use for the computation: the `n_classes`
        k-1 (n-1) ordinal problems problems are computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None
    Attributes  (based on OvR classifier -- @todo: edit)
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function. This attribute
        exists only if the ``estimators_`` defines ``coef_``.
        .. deprecated:: 0.24
            This attribute is deprecated in 0.24 and will
            be removed in 1.1 (renaming of 0.26). If you use this attribute
            in :class:`~sklearn.feature_selection.RFE` or
            :class:`~sklearn.feature_selection.SelectFromModel`,
            you may pass a callable to the `importance_getter`
            parameter that extracts feature the importances
            from `estimators_`.
    intercept_ : ndarray of shape (1, 1) or (n_classes, 1)
        If ``y`` is binary, the shape is ``(1, 1)`` else ``(n_classes, 1)``
        This attribute exists only if the ``estimators_`` defines
        ``intercept_``.
        .. deprecated:: 0.24
            This attribute is deprecated in 0.24 and will
            be removed in 1.1 (renaming of 0.26). If you use this attribute
            in :class:`~sklearn.feature_selection.RFE` or
            :class:`~sklearn.feature_selection.SelectFromModel`,
            you may pass a callable to the `importance_getter`
            parameter that extracts feature the importances
            from `estimators_`.
    classes_ : array, shape = [`n_classes`]
        Class labels.
    n_classes_ : int
        Number of classes.
    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.
        .. versionadded:: 1.0
    See Also
    --------
    MultiOutputClassifier : Alternate way of extending an estimator for
        multilabel classification.
    sklearn.preprocessing.MultiLabelBinarizer : Transform iterable of iterables
        to binary indicator matrix.
    Examples  (@todo: redo based on OvR)
    --------
    >>> import numpy as np
    >>> from sklearn.multiclass import OneVsRestClassifier
    >>> from sklearn.svm import SVC
    >>> X = np.array([
    ...     [10, 10],
    ...     [8, 10],
    ...     [-5, 5.5],
    ...     [-5.4, 5.5],
    ...     [-20, -20],
    ...     [-15, -20]
    ... ])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> clf = OneVsRestClassifier(SVC()).fit(X, y)
    >>> clf.predict([[-19, -20], [9, 9], [-5, 5]])
    array([2, 0, 1])

    Adapted by: Lee Prevost, https://github.com/leeprevost
    """

    def __init__(self, estimator, *, n_jobs=None):
        # @todo: I think I need a way to pass non default class ordering.   Default is np.sort(np.unique(y)) but what if another order needed such that classes_[0] > classes_[n] (eg. hot, cold, warm)
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.class_order = None

    def fit(self, X, y, reverse_classes=False):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.
        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        reverse_classes: reorders classes to make last class more important (eg. hot>mild>cold)

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        # @todo: keep? same as ovr?
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.

        # following improvised from
        # https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
        # by Muhammad
        classes = np.sort(np.unique(y))   # don't I need a way to have input on order?
        if reverse_classes:
            self.classes_ = classes[::-1]
        else:
            self.classes_ = classes

        #self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        #self.label_binarizer_.fit(y)
        self.y_type_ = type_of_target(y)

        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.

        if self.classes_.shape[0] > 2:
            # for each k - 1 ordinal value we fit a binary classification problem

            # @todo: question - should I allow for this to be reversed with kwargs in order to
            # emphasize the positive class (eg. "hot" in cold < warm < hot three class problem)

            # @todo: derived estimators: classes become imbalanced? how to balance classes?
            # probable answer: make use of "weight" kwarg when passing estimator to "init".   Warning?

            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_binary)(
                    self.estimator,
                    X,
                    np.where(y > self.classes_[i], 1, 0),
                    classes=[
                        '= %s' % i,
                        "> %s" % i,
                    ],
                )
                for i in range(self.classes_.shape[0] - 1)
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @available_if(_estimators_has("partial_fit"))
    def partial_fit(self, X, y, classes=None):
        """Partially fit underlying estimators.
        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iteration.
        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.
        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.
        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.
        Returns
        -------
        self : object
            Instance of partially fitted estimator.
        """

        pass  # for now bypass this and edit it later.  @todo: implement partial_fit

        '''
        if _check_partial_fit_first_call(self, classes):
            if not hasattr(self.estimator, "partial_fit"):
                raise ValueError(
                    ("Base estimator {0}, doesn't have partial_fit method").format(
                        self.estimator
                    )
                )
            self.estimators_ = [clone(self.estimator) for _ in range(self.n_classes_)]

            # A sparse LabelBinarizer, with sparse_output=True, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory consumption
            # in the fit_ovr function overall.
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(
                (
                        "Mini-batch contains {0} while classes " + "must be subset of {1}"
                ).format(np.unique(y), self.classes_)
            )

        # this is where we need n-1 targets from binarizer.
        # y > Vi
        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(estimator, X, column)
            for estimator, column in zip(self.estimators_, columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_

        return self '''

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        **estimator must have predict_proba method.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.
        Returns
        -------
        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Predicted multi-class targets.
        """
        check_is_fitted(self)

        n_samples = _num_samples(X)
        if self.y_type_ == "multiclass":  #this work for pred_proba but not decision function and that is primary method
            return np.argmax(self.predict_proba(X), axis=1)

        # need to rewrite the following if not "multiclass" or no predict_proba or want to use threshold
        else:
            #replaced elaborate else logic from OvR with NotImplementedError
            raise NotImplementedError("This type of y target not implemented:  type: ".format(self.y_type_))


    @available_if(_estimators_has("predict_proba"))
    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by label of classes.
        Note that in the multilabel case, each sample can have any number of
        labels. This returns the marginal probability that the given sample has
        the label in question. For example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.
        In the single label multiclass case, the rows of the returned matrix
        sum to 1.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        Returns
        -------
        T : (sparse) array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.

        @todo: multilabel is untested
        """
        check_is_fitted(self)
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

        if len(self.estimators_) == 1:  # binary problem
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)
            predicted = Y

        else:
            predicted = self._ordinal_binary_to_class_array(Y)

        if not self.multilabel_:
            # Then, probabilities should be normalized to 1.
            predicted /= np.sum(predicted, axis=1)[:, np.newaxis]

        return predicted

    @available_if(_estimators_has('decision_function'))
    def decision_function(self, X):
        """Decision function for the OneVsRestClassifier.
        Return the distance of each sample from the decision boundary for each
        class. This can only be used with estimators which implement the
        `decision_function` method.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes) or (n_samples,) for \
            binary classification.
            Result of calling `decision_function` on the final estimator.
            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """

        check_is_fitted(self)
        if len(self.estimators_) == 1:
            return self.estimators_[0].decision_function(X)
        else:
            Y = np.array([e.predict_proba(X) for e in self.estimators_]).T
            decision = self._ordinal_binary_to_class_array(Y)
            return decision



    def _ordinal_binary_to_class_array(self, Y):
        predicted = {}
        pr_name = "Pr(y={}"
        for i, cls in enumerate(self.classes_):
            pr_names = "Pr"
            if i == 0:  # first pass
                #Pr(V1) = 1 − Pr(Target > V1)
                predicted.update({pr_name.format(cls): 1 - Y[:, 0]})  # first class
            elif cls == self.classes_[-1]:  # last pass
                #Pr(Vk) = Pr(Target > Vk−1)
                predicted.update({pr_name.format(cls): Y[:, -1]})  # last class
            elif i>0:  # middle passes, need qualifier so it doesn't overwrite last class
                #Pr(Vi) = Pr(Target > Vi−1) − Pr(Target > Vi) , 1 < i < k
                predicted.update({pr_name.format(cls): Y[:, cls - 1] - Y[:, cls]})  # middle classes

        self.ordinal_prob_names_ = predicted.keys()

        predicted = np.vstack(list(predicted.values())).T
        return predicted

    @property
    def multilabel_(self):
        """Whether this is a multilabel classifier."""
        return self.y_type_.startswith("multilabel")

    @property
    def n_classes_(self):
        """Number of classes."""
        return len(self.classes_)

    # TODO: Remove coef_ attribute in 1.1
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `coef_` was deprecated in "
        "version 0.24 and will be removed in 1.1 (renaming of 0.26). "
        "If you observe this warning while using RFE "
        "or SelectFromModel, use the importance_getter "
        "parameter instead."
    )
    @property
    def coef_(self):
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "coef_"):
            raise AttributeError("Base estimator doesn't have a coef_ attribute.")
        coefs = [e.coef_ for e in self.estimators_]
        if sp.issparse(coefs[0]):
            return sp.vstack(coefs)
        return np.vstack(coefs)

    # TODO: Remove intercept_ attribute in 1.1
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `intercept_` was deprecated in "
        "version 0.24 and will be removed in 1.1 (renaming of 0.26). "
        "If you observe this warning while using RFE "
        "or SelectFromModel, use the importance_getter "
        "parameter instead."
    )
    @property
    def intercept_(self):
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "intercept_"):
            raise AttributeError("Base estimator doesn't have an intercept_ attribute.")
        return np.array([e.intercept_.ravel() for e in self.estimators_])

    # TODO: Remove in 1.1
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `_pairwise` was deprecated in "
        "version 0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def _pairwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return getattr(self.estimator, "_pairwise", False)

    def _more_tags(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return {"pairwise": _safe_tags(self.estimator, key="pairwise")}

    @property
    def _has_decision_function(self):
        return hasattr(self.estimator, "decision_function")

    @property
    def _has_predict_proba(self):
        return hasattr(self.estimator, "predict_proba")