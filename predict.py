# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 rdorff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import pandas as pd
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn import neighbors
from xgboost import XGBClassifier


##########################################################################
def find_hypers(sport='baseball',start_year=2002,end_year=2018):


    models = {'LogReg': LogisticRegression(max_iter=1000,solver='lbfgs'),
              'Ada': AdaBoostClassifier(n_estimators=100,learning_rate=.5),
              'Forest': RandomForestClassifier(max_depth=None,min_samples_split=2),
              'KNN': neighbors.KNeighborsClassifier(5, weights="distance"),
              'Trees': ExtraTreesClassifier(max_depth=None,min_samples_split=2),
              'XGB': XGBClassifier()  }
        
    data = pd.read_csv('Data/' + sport + '.csv',
                       dtype={'Position':'category','left':object,
                              'College':'category'})  
    data = data[data['Year'] >= start_year]
    data = data[data['Year'] <= end_year]
    labels = data['Target']
    data = data.drop(['Target','Year'],axis=1)
    data = pd.get_dummies(data, columns=["Position","Team"])
    data['College'] = data['College'].astype('category')
    data['College'] = data['College'].cat.codes
    
       # Create test/training data 
    x_train,x_test,y_train,y_test = train_test_split(data, labels,
                                                     test_size=.3)

    algos = {"Log":LogisticRegression(), "Ada":AdaBoostClassifier(),
            'Forest':RandomForestClassifier(), 'KNN':neighbors.KNeighborsClassifier(),
            'Trees':ExtraTreesClassifier(), 'XGB':XGBClassifier()}
    
    params = {'Log':{'Log__penalty':['l1', 'l2','none'],
                    'Log__solver':['newton-cg','lbfgs','liblinear','saga'],
                    'Log__fit_intercept':[False,True]},
               'Ada':{'Ada__n_estimators':[25,50,75],
                      'Ada__algorithm':['SAMME','SAMME.R'],
                      'Ada__learning_rate':[.5,1]},
               'Forest':{'Forest__n_estimators':[40,100],
                      'Forest__criterion':['gini'],#,'entropy'],
                      'Forest__max_depth':[20,30,None],
                      'Forest__min_samples_split':[2,4],
                      'Forest__min_samples_leaf':[1,5],
                      'Forest__max_features':['auto'],#,'sqrt','log2'],
                      'Forest__max_leaf_nodes':[10,20,None]},
               'KNN': {'KNN__n_neighbors':[3,5,8],
                     'KNN__weights':['uniform','distance'],
                     'KNN__algorithm':['auto','ball_tree','kd_tree','brute'],
                     'KNN__leaf_size':[15,30,45]},
               'Trees':{'Trees__n_estimators':[30,50,100],
                     'Trees__criterion':['gini'],#,'entropy'],
                     'Trees__max_depth':[20,50,None],
                     'Trees__min_samples_split':[2,4,6],
                     'Trees__min_samples_leaf':[1,5],
                     'Trees__max_features':['auto'],#'sqrt','log2'],
                     'Trees__max_leaf_nodes':[12,20,None]},
                'XGB':{'XGB__learning_rate':[.1,.3],
                       'XGB___max_depth':[3,6],
                       'XGB__reg_alpha':[0,.2],
                       'XGB__min_child_weight':[1,2]}}
    
    alg = 'Trees'
    pipe = Pipeline([("scaler", StandardScaler()),
                     (alg, algos[alg])])
    

    pipe_gs = GridSearchCV(pipe, params[alg],scoring="f1_micro").fit(x_train, y_train)
    
    params = pipe_gs.best_params_
    print("Best classifier:", params)

##########################################################################
def remove_duplicates(probs,curr_teams,classes):
    """ Iterates through algorithm predictions and if any prediction is the
        same as a player's current team, replaces it with the next highest 
        scoring team.
        
        Parameters:
            probs (np 
                   array): array of probability of each class for each player
            curr_teams (np array): current teams
            classes (np array): array of classes used by algorithm
            
        Returns:
            predictions (list): modified algorithm predictions
    """
    predictions = []
    
    for i in range(len(probs)):
            
        top_two = classes[np.argpartition(probs[i], -2)[-2:]] 
        
        # if top team is curr team, choose second choie
        if top_two[1] == curr_teams[i]:
            predictions.append(top_two[0])
        else:
            predictions.append(top_two[1]) # top team is not curr team

    return predictions

def run_algo(model_name,x_train,y_train,x_test,y_test,curr_teams, score,
             sport='baseball'):
    """ Predict labels using model and return measure.

    Parameters:
        model_name (str): name of model to use 
        x_train (dataframe)
        y_train (series) 
        x_test (dataframe)
        y_test (series)
        curr_teams (array): current teams
        score (str): goodness measure, either accuracy or f1              
    Return:
        accuracy (float): % of correctly predicted labels 
        f1 (float): f1 score
    """
    
    # baseball
    if sport == 'baseball':
       models = {'LogReg': LogisticRegression(max_iter=1000,
                                                penalty='l1',solver='liblinear'),
                  'Ada': AdaBoostClassifier(n_estimators=100,algorithm='SAMME'),
                  'Forest': RandomForestClassifier(min_samples_leaf=5,
                                                    min_samples_split=2,
                                                    n_estimators=40,
                                                    max_features='log2',
                                                    max_leaf_nodes=20,
                                                    max_depth=20),
                  'KNN': neighbors.KNeighborsClassifier(n_neighbors=5,
                                                        leaf_size=15,
                                                        weights='distance'),
                  'Trees': ExtraTreesClassifier(max_depth=50,max_leaf_nodes=20,
                                                max_features='sqrt',
                                                min_samples_leaf=5,
                                                min_samples_split=2,
                                                n_estimators=50),
                  'XGB': XGBClassifier(max_depth=3,learning_rate=.1,
                                        min_child_weight=1,reg_alpha=.5)}
        
    # basketball
    else:
        models = {'LogReg': LogisticRegression(max_iter=1000,fit_intercept=False,
                                               penalty='l1',solver='liblinear'),
                  'Ada': AdaBoostClassifier(n_estimators=75,algorithm='SAMME'),
                  'Forest': RandomForestClassifier(min_samples_leaf=1,
                                                   min_samples_split=2,
                                                   n_estimators=100,
                                                   max_features='auto',
                                                   max_leaf_nodes=None,
                                                   max_depth=30),
                  'KNN': neighbors.KNeighborsClassifier(n_neighbors=3,
                                                        leaf_size=15,
                                                        weights='distance'),
                  'Trees': ExtraTreesClassifier(max_depth=None,max_leaf_nodes=None,
                                                max_features='auto',
                                                min_samples_leaf=1,
                                                min_samples_split=4,
                                                n_estimators=100),
                  'XGB': XGBClassifier(max_depth=3,learning_rate=.3,
                                   min_child_weight=1,reg_alpha=0)}
    
    model = models[model_name]
    model.fit(x_train, y_train)
    
    probs = model.predict_proba(x_test)
    
    # Remove predictions that are the player's current team
    no_dup = remove_duplicates(probs,curr_teams,model.classes_)
       
    # Get percentage of correct predictions   
    accuracy = accuracy_score(no_dup, y_test)
    f1 = f1_score(no_dup, y_test,average='micro')
    
    
    return accuracy if score=='accuracy' else f1
    


#############################################################################
def predict_teams(sport='baseball',score='Accuracy',algorithms=['Forest','XGB'],
                  features={'Position':False,'Team':False,
                               'CareerLen':False,'Performance':False,
                               'Rank':False,'Value':False,
                               'College':False,'Social':True},
                  start_year=2001,end_year=2019):
    
     
    
    ''' Determine which players will leave the league based off of the previous 
        year's performance
        
        Parameters:
            sport (str): baseball or basketball
            score (string): Accuracy or f1
            algorithms (list): list of alogrithm names to run 
            features (dict): dictionary of features to use
            start_year (int): start year
            end_year (int): end year           
    
        Return:
            results (list): list of results from each algorithm
    
    '''

    results = []
    if sport == 'baseball': 
        perf = ['OWn%','BtRuns','BtWins','Fld%']
        teams = ['ARI','ATL','CHC','CIN','COL','LAD','MIA',
                          'MIL','NYM','PHI','PIT','SDP','SFG','STL','WSN',
                          'BAL','BOS','CHW','CLE','DET','HOU', 'KCR',
                          'LAA','MIN','NYY','OAK','SEA','TBR','TEX', 'TOR']  
    else:   
        perf = ['bpm', 'per', 'ws']
        teams =  ['ATL','BOS','BRK','CHI','CHO','CLE','DAL',
                         'DEN','DET','GSW', 'HOU','IND','LAC','LAL',
                         'MEM','MIA','MIL','MIN','NOP','NYK','OKC',
                         'ORL','PHI','PHO','POR','SAC','SAS','TOR',
                         'UTA','WAS']
    
    drop_features = ['Position','CareerLen','Rank','Value']


    data = pd.read_csv('Data/' + sport + '.csv',
                       dtype={'Position':'category','left':object,
                              'College':'category'})  

    if sport == 'full_basketball':
        data = data[data['College']!= '0']
        data = data[data['left']=='True']
        data = data[data['Target']!='Retire']
        data.drop(['player','age','twitter_id','g','mp','ts_pct',
                   'fg3a_per_fga_pct','fta_per_fga_pct','orb_pct','drb_pct',
                   'trb_pct','ast_pct','stl_pct','blk_pct','tov_pct','usg_pct',
                   'ows', 'dws','ws_per_48','obpm','dbpm','vorp','affinity'
                   ,'left'],
                  axis=1,inplace=True)

    # Constrain year
    data = data[data['Year'] >= start_year]
    data = data[data['Year'] <= end_year]

    labels = data['Target']

    # One hot encode if necessary      
    if features['Position'] == True:
        data = pd.get_dummies(data, columns=["Position"])
    if (sport == 'basketball') or (sport == 'full_basketball'):
        if features['College'] == False:
            data = data.drop('College', axis=1)
        else:
            data['College'] = data['College'].astype('category')
            data['College'] = data['College'].cat.codes

    # Drop social and performance if necessary
    if features['Social'] == False:
        data = data.drop(teams,axis=1)       
    if features['Performance'] == False:
        data = data.drop(perf,axis=1)

    # Remove columns based on feature parameters
    drop_columns = [key for key, value in features.items() if (value == False and 
                                 ( key in drop_features))]
    data = data.drop(drop_columns,axis=1)
    data = data.drop(['Year','Target'],axis=1)

    # Create test/training data 
    x_train,x_test,y_train,y_test = train_test_split(data, labels, 
                                                     test_size=.3)

    # Get the current teams for the test set
    test_teams = x_test['Team'].to_numpy()
    # One hot encode or drop team column
    if features['Team'] == True:
        x_train = pd.get_dummies(x_train, columns=['Team'])
        x_test = pd.get_dummies(x_test,columns=['Team'])
    else:
        x_test = x_test.drop('Team',axis=1)
        x_train = x_train.drop('Team',axis=1)

    # Run Algorithms   
    for algo in algorithms:
        results.append(run_algo(algo,x_train, y_train,x_test,y_test,
                                test_teams,score,sport=sport))   
    return results

def run_scenario(sport='baseball',score='accuracy', algs=['Forest','XGB'],
                     features={'Position':False,'Team':False,
                               'CareerLen':False,'Performance':False,
                               'Rank':False,'Value':False,
                               'College':False,'Social':True},
                start_year=2002,end_year=2019):
    
    ''' Runs a scenario 10 times and returns the mean accuracy or score
    
        Parameters:
            sport (str): baseball or basketball
            score (string): Accuracy or f1
            algorithms (list): list of alogrithm names to run 
            features (dict): dictionary of features to use
            start_year (int): start year
            end_year (int): end year           
    
        Return:
            average results over 10 runs
    '''
     
    test_result = []

    # run scenario 10 times and average
    for i in range(10):
        results = predict_teams(sport,score,algs,features,start_year,end_year)
        test_result.append(results)

    return [np.array([result[i] for result in test_result]).mean() 
                    for i in range(len(algs))]

def run_tests(sport='baseball',
              score='accuracy', algs=['Forest','XGB'],
              start_year=2002,end_year=2019):
    ''' Run predictions multiple times over all possible scenarios and 
        saves output 
        
        Parameters:

            sport (str): baseball, basketball, or full_basketball (for college)
            score (string): Accuracy or f1
            algorithms (list): list of alogrithm names to run 
            start_year (int): start year
            end_year (int): end year           


    '''
    all_results = []
    social_boo = [False,True]
    columns =['Position','Team','CareerLen','Performance','Rank','Value']
    n_col = len(columns)
    
    features={'Position':False,'Team':False,
                               'CareerLen':False,'Performance':False,
                               'Rank':False,'Value':False,
                               'College':False,'Social':True}

    index = []
    for i in range(n_col+1):
        print(i)
        # set features and index
        if i != n_col:
            feature = columns[i]
            features[feature] = True
        # all scenario cases
        else:
            features = {key:True for key in features.keys()}
            feature = 'All'
        if sport == 'basketball':
            index.extend([feature,feature+'/College',feature +'/Social',
                         feature +'/College/Social'])
        elif sport == 'baseball':
            index.extend([feature,feature + '/Social'])
        else: # college only
            index.extend([feature, feature+'/College'])
            
        
        # run with and without social\
        for j in range(2):
            if sport == 'full_basketball':
                features['Social'] = False
                features['College'] = social_boo[j]
                all_results.append(run_scenario(sport,score,algs,
                                            features,start_year,end_year))               
            
            else:
                features['Social'] = social_boo[j]
                all_results.append(run_scenario(sport,score,algs,
                                            features,start_year,end_year))
                # run college
                if sport == 'basketball':
                    features['College'] = True 
                    all_results.append(run_scenario(sport,score,algs,
                                                    features,start_year,end_year))
                    features['College'] = False
        features[feature] = False
    
    features = {key:False for key in features.keys()}

    # run college and social
    if sport != 'full_basketball':

        features['Social'] = True
        index.append('Social')
        all_results.append(run_scenario(sport,score,algs,
                                        features,start_year,end_year))
    
    if sport == 'basketball':
        features['College'] = True
        index.append('Social/College')
        all_results.append(run_scenario(sport,score,algs,
                                        features,start_year,end_year))       

    if sport == 'basketball' or sport == 'full_basketball':
        # college only
        features['Social'] = False
        features['College'] = True

        index.append('College')
        all_results.append(run_scenario(sport,score,algs,
                                        features,start_year,end_year))       

    # Create DataFrame
    result_df = pd.DataFrame(all_results,columns=algs,index=index)
    result_df = result_df.round(3)
    result_df.to_csv('Results/'+sport+'_'+score+'new.csv')
    
    print(result_df)
            
if __name__ == '__main__':

    run_tests(sport='basketball',score='accuracy',
                        algs=['Ada','LogReg','Forest','XGB','KNN','Trees'],
                        start_year=2001,end_year=2018)
    # run_tests(sport='baseball',score='f1',
    #             algs=['Ada','LogReg','Forest','XGB','KNN','Trees'],
    #             start_year=2002,end_year=2018)
    # run_tests(sport='full_basketball',score='accuracy',
    #             algs=['Ada','LogReg','Forest','XGB','KNN','Trees'],
    #             start_year=2001,end_year=2019)
    # run_scenario(sport='full_basketball',score='accuracy', algs=['Forest','XGB'],
    #                   features={'Position':False,'Team':False,
    #                             'CareerLen':False,'Performance':False,
    #                             'Rank':False,'Value':False,
    #                             'College':True,'Social':False},
    #             start_year=2001,end_year=2019)
    # find_hypers(sport='basketball',start_year=2001,end_year=2018)