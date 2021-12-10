import numpy
import pandas
import sklearn.metrics

def calc_sedi(conf_mat):
    hr1 = conf_mat[1,1] / (conf_mat[1,0] + conf_mat[1,1])
    fa1 = conf_mat[0,1] / (conf_mat[0,0] + conf_mat[0,1])
    sedi_score1 = (
        (numpy.log(fa1) - numpy.log(hr1) - numpy.log(1.0-fa1) + numpy.log(1.0-hr1) )
        / (numpy.log(fa1) + numpy.log(hr1) + numpy.log(1.0 - fa1) + numpy.log(1.0-hr1) )  )
    return sedi_score1


def calculate_metric_suite(classifiers_dict, train_test_dict, threshold=None):
    metrics1 = {}
    metrics_df_dict = {}
    for set_name, (X1, y1) in train_test_dict.items():
        md1 = {'classifier': [],
               'precision_noRotor': [], 'precision_rotor': [],
               'recall_noRotor': [], 'recall_rotor': [], 
               'f1_noRotor': [], 'f1_rotor': [], 
               'hit_rate': [], 'false_alarm_rate': []
              }
        for clf_name, clf1 in classifiers_dict.items():
            md1['classifier'] += [clf_name]
            if threshold is None:
                pred_result = clf1.predict(X1)
            else:
                pred_result = ((clf1.predict_proba(X1) > threshold) > threshold)[:,1]
            prec, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(pred_result, y1)
            md1['precision_noRotor'] += [prec[0]]
            md1['precision_rotor'] += [prec[1]]
            md1['recall_noRotor'] += [recall[0]]
            md1['recall_rotor'] += [recall[1]]
            md1['f1_noRotor'] += [f1[0]]
            md1['f1_rotor'] += [f1[1]]
            cm1 = sklearn.metrics.confusion_matrix(pred_result, y1)
            hit_rate = cm1[1,1] / (cm1[1,1]+cm1[0,1])
            md1['hit_rate'] += [hit_rate]
            false_alarm_rate = cm1[0,1] / (cm1[0,1]+cm1[0,0])
            md1['false_alarm_rate'] += [false_alarm_rate]
        metrics1[set_name] = md1
        metrics_df_dict[set_name] = pandas.DataFrame(md1)
    return (metrics1, metrics_df_dict)

def calculate_sedi_suite(thresholds_list, y_actual, y_predicted_raw):
    thresholds_list = list(numpy.arange(1e-3,0.995,5e-3))
    hit_rates = []
    false_alarm_rates = []
    sedi_list = []
    for threshold in thresholds_list:
        y_pred = list(map(float, y_predicted_raw > threshold)) 
        cm1 = sklearn.metrics.confusion_matrix(y_actual, y_pred)
        hit_rates += [cm1[1,1] / (cm1[1,0] + cm1[1,1])]
        false_alarm_rates += [cm1[0,1] / (cm1[0,0] + cm1[0,1])]
        sedi_list += [calc_sedi(cm1)]
        
    results = { 
        'hit_rates': hit_rates,
        'false_alarm_rates': false_alarm_rates,
        'sedi_list': sedi_list,
    }
    return results
        
        
    