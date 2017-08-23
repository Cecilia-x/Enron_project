from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import sys
sys.path.append("../tools/")
import loadEnron

def print_kbest_result(features, labels, ft_list):
    kb = SelectKBest(score_func=f_classif, k='all')
    kb.fit(features, labels)
    scores = kb.scores_
    for i,e in enumerate(scores):
        print "%.2f" % round(e,2), ft_list[i+1]
        
### Excluded 'restricted_stock_deferred', 'loan_advances', for too many nulls.
features_kbest = ['poi','salary','bonus', 'long_term_incentive','deferral_payments',
                'deferred_income', 'expenses','other','director_fees',
                'exercised_stock_options','total_payments',
                'total_stock_value', 'incentives', 'fix_income',
                'poi_in_recvmsg_ratio','share_recp_with_poi_ratio','poi_in_sentmsg_ratio']   


labels, features = loadEnron.load_n_transf(features_kbest)
print_kbest_result(features, labels, features_kbest)
