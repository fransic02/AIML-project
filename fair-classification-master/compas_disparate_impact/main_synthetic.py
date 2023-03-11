import os,sys
import numpy as np
from prepare_compas_data import *
sys.path.insert(0, '../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints


def test_compas_data():
    x_train, y_train, x_control_train = load_data(load_data_size=None,
                                               data_path='../data/compas_synthetic.csv')  # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup

    print("== Computing p-rule on test set (real data) ==")
    x_test, y_test, x_control_test = load_data(load_data_size=None,  data_path='../data/compas.csv')  # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
    ut.compute_p_rule(x_control_test['race'], y_test)  # compute the p-rule in the original data

    x_train = ut.add_intercept(x_train)  # add intercept to X before applying the linear classifier
    x_test = ut.add_intercept(x_test)

    """ Split the data into train and test """
    #train_fold_size = 0.7
    #x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

    apply_fairness_constraints = None
    apply_accuracy_constraint = None
    sep_constraint = None

    loss_function = lf._logistic_loss
    sensitive_attrs = ['race']
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    def train_test_classifier():
        w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints,
                           apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh,
                           gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train,
                                                                                                 x_test, y_test, None,
                                                                                                 None)
        distances_boundary_test = (np.dot(x_test, w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test,
                                                    sensitive_attrs)
        cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test,
                                                            sensitive_attrs)
        p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test],
                                                    sensitive_attrs[0])
        return w, p_rule, test_score

    """ Classify the data while optimizing for accuracy """
    print
    print('== Unconstrained (original) classifier ==')
    # all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
    apply_fairness_constraints = 0
    apply_accuracy_constraint = 0
    sep_constraint = 0
    w_uncons, p_uncons, acc_uncons = train_test_classifier()

    """ Now classify such that we optimize for accuracy while achieving perfect fairness """
    apply_fairness_constraints = 1  # set this flag to one since we want to optimize accuracy subject to fairness constraints
    apply_accuracy_constraint = 0
    sep_constraint = 0
    sensitive_attrs_to_cov_thresh = {'race': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03]} # item #1 corresponds to "Afrocan-American"
    print
    print('== Classifier with fairness constraint ==')
    w_f_cons, p_f_cons, acc_f_cons = train_test_classifier()

    """ Classify such that we optimize for fairness subject to a certain loss in accuracy """
    apply_fairness_constraints = 0  # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
    apply_accuracy_constraint = 1  # now, we want to optimize fairness subject to accuracy constraints
    sep_constraint = 0
    gamma = 0.07  # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
    print('== Classifier with accuracy constraint ==')
    w_a_cons, p_a_cons, acc_a_cons = train_test_classifier()


    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_compas_data()
