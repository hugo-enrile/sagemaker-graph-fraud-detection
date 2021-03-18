import os
import argparse
from itertools import combinations
import logging

import pandas as pd
import numpy as np 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--transactions', type=str, default='transaction.csv', help='name of file with transactions')
    parser.add_argument('--identity', type=str, default='identity.csv', help='name of file with identity info')
    #columns with identity information related to a user or a transaction: IP, Phone Number... These column types become edge types in the graph, and the entries
    #in these columns become the nodes
    parser.add_argument('--id-cols', type=str, default='', help='comma separated id cols in transactions table')
    #columns with categorical information, such as age range. These columns are used as node attributes in the heterogeneous graph
    parser.add_argument('--cat-cols', type=str, default='', help='comma separated categorical cols in transactions')
    parser.add_argument('--train-data-ratio', type=float, default=0.8, help='fraction of data to use in training set')
    parser.add_argument('--construct-homogeneous', action="store_true", default=False,
                        help='use bipartite graphs edgelists to construct homogenous graph edgelist')
    return parser.parse_args()

    #numerical columns: columns with numerical features like number of times a transaction has been tried. Entries are used as node attributes

def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format = log_format, level = logging.INFO)
    return logger

def load_data(data_dir, transaction_data, identity_data, train_data_ratio, output_dir):
    #load the transaction file for training
    transaction_df = pd.read_csv(os.path.join(data_dir, transaction_data), sep = ';')
    logging.info("Shape of transaction data is {}".format(transaction_df.shape))
    logging.info("# Tagged transactions: {}".format(len(transaction_df) - transaction_df.Risk.isnull().sum()))

    #load the identity attributes for training
    identity_df = pd.read_csv(os.path.join(data_dir, identity_data), sep = ';')
    logging.info("Shape of identity data is {}".format(identity_df.shape))
    print(identity_df)

    #split for test/validation
    n_train = int(transaction_df.shape[0] * train_data_ratio)
    test_ids = transaction_df.id.values[n_train:]
    
    #determinate the proportion of fraud transaction in the proposed data
    get_risk_frac = lambda x: 100 * sum(x)/len(x)
    logging.info("Percent risk for train transactions: {}".format(get_risk_frac(transaction_df.Risk[:n_train])))
    logging.info("Percent risk for test transactions: {}".format(get_risk_frac(transaction_df.Risk[n_train:])))
    logging.info("Percent risk for all transactions: {}".format(get_risk_frac(transaction_df.Risk)))

    #in the test.csv file hosted in the output folder it writes the TransactionIDs from the transactions that will be used for testing
    with open(os.path.join(output_dir, 'test.csv'), 'w') as f:
        f.writelines(map(lambda y: str(y) + '\n', test_ids))
    logging.info("Wrote test to file: {}".format(os.path.join(output_dir, 'test_csv')))

    return transaction_df, identity_df, test_ids

def get_features_and_labels(transactions_df, transactions_id_cols, transactions_cat_cols, output_dir):
    #get features
    #isFraud and TransactionDT are not features: one determines whether a transaction is labeled as fraud or not and the other one is an associated timedelta
    #columns in transactions_id_cols are related to identity attributes, not real features
    non_feature_cols = ['Risk'] + transactions_id_cols.split(",")
    #rest of the columns are feature ones
    feature_cols = [col for col in transactions_df.columns if col not in non_feature_cols]
    #make a distintion between categorical and numerical columns
    logging.info("Categorical columns: {}".format(transactions_cat_cols.split(",")))
    #convert each categorical variable into dummy variables. Fill NA with 0 (not every entry has values in its columns)
    features = pd.get_dummies(transactions_df[feature_cols], columns = transactions_cat_cols.split(",")).fillna(0)
    #apply logarithmic transformation to the amount value --> easier to interpret
    features['Credit amount'] = features['Credit amount'].apply(np.log10)
    logging.info("Transformed feature columns: {}".format(list(features.columns)))
    logging.info("Shape of features: {}".format(features.shape))
    #save the transformed features into a file in the output folder
    features.to_csv(os.path.join(output_dir, 'features.csv'), index = False, header = False)
    logging.info("Wrote features to file: {}".format(os.path.join(output_dir, 'features.csv')))

    #get labels
    transactions_df[['id','Risk']].to_csv(os.path.join(output_dir, 'tags.csv'), index = False)
    logging.info("Wrote labels to file: {}".format(os.path.join(output_dir, 'tags.csv')))

def get_relations_and_edgelist(transactions_df, identity_df, transactions_id_cols, output_dir):
    #get relations
    #transactions_id_cols are the identity columns specified in the command line (the take part of the transactions data)
    edge_types = list(identity_df.columns)
    logging.info("Found the following distinct relation types: {}".format(edge_types))
    #define id_cols as the columns with identity information from the transactions data
    #id_cols = ['id'] + transactions_id_cols.split(",")
    #merge the identity information from the transaction data with the whole identity data by the TransactionID column
    full_identity_df = identity_df
    logging.info("Shape of identity columns: {}".format(full_identity_df.shape))

    #get edges
    edges = {}
    for etype in edge_types:
        edgelist = full_identity_df[['id', etype]].dropna()
        edgelist.to_csv(os.path.join(output_dir, 'relation_{}_edgelist.csv').format(etype), index = False, header = True)
        logging.info("Wrote edgelist to: {}".format(os.path.join(output_dir, 'relation_{}_edgelist.csv').format(etype)))
        edges[etype] = edgelist

    return edges

def create_homogeneous_edgelist(edges, output_dir):
    homogeneous_edges = []
    for etype, relations in edges.items():
        for frame in relations.groupby(etype):
            new_edges = [(a, b) for (a, b) in combinations(frame.id.values,2)
                        if (a, b) not in homogeneous_edges and (b, a) not in homogeneous_edges]
            homogeneous_edges.extend(new_edges)
    
    with open(os.path.join(output_dir, 'homogeneous_edgelist.csv'), 'w') as f:
        f.writelines(map(lambda x: "{}, {}\n".format(x[0], x[1]), homogeneous_edges))
    logging.info("Wrote homogeneous edgelist to file: {}".format(os.path.join(output_dir, 'homogeneous_edgelist.csv')))

if __name__ == '__main__':
    logging = get_logger(__name__)

    args = parse_args()

    transactions, identity, test_transactions = load_data(args.data_dir,
                                                            args.transactions,
                                                            args.identity,
                                                            args.train_data_ratio,
                                                            args.output_dir)
    
    get_features_and_labels(transactions, args.id_cols, args.cat_cols, args.output_dir)
    relational_edges = get_relations_and_edgelist(transactions, identity, args.id_cols, args.output_dir)

    if args.construct_homogeneous:
        create_homogeneous_edgelist(relational_edges, args.output_dir)