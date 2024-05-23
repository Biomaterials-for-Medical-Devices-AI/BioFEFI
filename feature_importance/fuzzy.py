import argparse
import pandas as pd

from feature_importance.call_methods import save_importance_results
from feature_importance.feature_importance_methods import (
    calculate_shap_values, calculate_lime_values)
#from machine_learning import train

class Fuzzy:
    """
    Fuzzy class to interpret synergy of importance between features within context.

    """

    def __init__(self, opt: argparse.Namespace, logger: object = None) -> None:
        self._opt = opt
        self._logger = logger
        self._local_importance_methods = self._opt.local_importance_methods
        self.importance_type = 'local' # local feature importance


    
    def interpret(self, models, ensemble_results, data):
        '''
        Interpret the model results using the selected feature importance methods and ensemble methods.
        Parameters:
            models (dict): Dictionary of models.
            data (object): Data object.
        Returns:
            dict: Dictionary of feature importance results.
        '''
        # create a copy of the data
        X_train, X_test, _, _ = data.X_train, data.X_test, data.y_train, data.y_test
        self._logger.info(f"-------- Start of fuzzy interpretation logging--------")
        # Step 1: fuzzy feature selection to select top features for fuzzy interpretation
        if self._opt.fuzzy_feature_selection:
            # Select top features for fuzzy interpretation
            topfeatures = self._select_features(ensemble_results['Majority Vote'])
            X_train = X_train[topfeatures]
            X_test = X_test[topfeatures]
        # Step 2: Assign granularity to features e.g. low, medium, high categories
        if self._opt.is_granularity:
            X_train = self._fuzzy_granularity(X_train)
            X_test = self._fuzzy_granularity(X_test)

        # Update data object with new features
        data.X_train, data.X_test = X_train, X_test
        # Step 3: Train and evaluate models
        #trained_models = train.run(ml_opt, data, self._logger)

        # Step 4: Master feature importance dataframe for granular features from local feature importance methods and ML models
        master_importance_df = self._local_feature_importance(models, data.X_train, data.y_train)

        # Step 5: Extract fuzzy rules from master dataframe
        fuzzy_rules_df = self._fuzzy_rule_extraction(master_importance_df)

        # Step 6: Identify most occuring fuzzy rules by context (e.g. target category:low, medium, high)





        #local_importance_results = self._local_feature_importance(models, X, y)
        self._logger.info(f"-------- End of fuzzy interpretation logging--------") 

        return fuzzy_rules_df

    def _select_features(self, majority_vote_results):
        '''
        Select top features from majority vote ensemble feature importance.
        Parameters:
            majority_vote_results: Dictionary of feature importance results.
        Returns:
            list: List of top features.
        '''
        self._logger.info(f"Selecting top {self._opt.number_fuzzy_features} features...")
        fi = majority_vote_results.sort_values(by=0, ascending=False)
        # Select top n features for fuzzy interpretation
        topfeatures = fi.index[:self._opt.number_fuzzy_features].tolist()
        return topfeatures
    
    def _fuzzy_granularity(self, X):
        '''
        Assign granularity to features.
        Parameters:
            X (pd.DataFrame): Features.
        Returns:
            pd.DataFrame: Features with granularity.
        '''
        import numpy as np
        import skfuzzy as fuzz
        import warnings

        # Suppress all warnings
        warnings.filterwarnings('ignore'
                                )
        self._logger.info(f"Assigning granularity to features...")
        # find interquartile values for each feature
        df_top_qtl = X.quantile([0,0.25, 0.5, 0.75,1])
        # Create membership functions based on interquartile values for each feature
        membership_functions = {}
        universe = {}
        for feature in X.columns:
            
            # Define the universe for each feature
            universe[feature] = np.linspace(X[feature].min(), X[feature].max(), 100)

            # Define membership functions
            # Highly skewed features
            if df_top_qtl[feature][0.00] == df_top_qtl[feature][0.50]:
                low_mf = fuzz.trimf(universe[feature], [df_top_qtl[feature][0.00],df_top_qtl[feature][0.50],
                                                        df_top_qtl[feature][0.75]])
                medium_mf = fuzz.trimf(universe[feature], [df_top_qtl[feature][0.50],df_top_qtl[feature][0.75],
                                                        df_top_qtl[feature][1.00]])
                high_mf = fuzz.smf(universe[feature], df_top_qtl[feature][0.75], df_top_qtl[feature][1.00])
            
            else:
                low_mf = fuzz.zmf(universe[feature], df_top_qtl[feature][0.00],df_top_qtl[feature][0.50])
                medium_mf = fuzz.trimf(universe[feature], [df_top_qtl[feature][0.25],df_top_qtl[feature][0.50],
                                                        df_top_qtl[feature][0.75]])
                high_mf = fuzz.smf(universe[feature], df_top_qtl[feature][0.50], df_top_qtl[feature][1.00])
            
            membership_functions[feature] = {'low': low_mf, 'medium': medium_mf, 'high': high_mf}

        # Create granular features using membership values
        new_df_features = []
        for feature in X.columns:
            X.loc[:, f'{feature}_small'] = fuzz.interp_membership(universe[feature], membership_functions[feature]['low'], X[feature])
            new_df_features.append(f'{feature}_small')
            X.loc[:, f'{feature}_mod'] = fuzz.interp_membership(universe[feature], membership_functions[feature]['medium'], X[feature])
            new_df_features.append(f'{feature}_mod')
            X.loc[:, f'{feature}_large'] = fuzz.interp_membership(universe[feature], membership_functions[feature]['high'], X[feature])
            new_df_features.append(f'{feature}_large')
        X = X[new_df_features]
                
        return X
    
    def _fuzzyset_selection(self, uni, mf1, mf2, mf3, val):
        '''
        Select fuzzy set with highest membership value.
        Parameters:
            uni (np.array): Universe.
            mf1 (np.array): Low membership function
            mf2 (np.array): Moderate membership function
            mf3 (np.array): High membership function
            val (float): Value.
        Returns:
            str: Fuzzy set with highest membership value
        '''
        import skfuzzy as fuzz
        mf_values = []
        # Calculate membership values for each fuzzy set
        mf_values.append(fuzz.interp_membership(uni, mf1, val))

        mf_values.append(fuzz.interp_membership(uni, mf2, val))

        mf_values.append(fuzz.interp_membership(uni, mf3, val))

        # Select fuzzy set with highest membership value
        index_of_max = mf_values.index(max(mf_values))

        # Return fuzzy set
        if index_of_max == 0:
            return 'low'
        if index_of_max == 1:
            return 'medium'
        if index_of_max == 2:
            return 'high'
    
    def _fuzzy_rule_extraction(self, df):
        '''
        Extract fuzzy rules from granular features.
        Parameters:
            df (Dataframe): master dataframe of feature importances from local feature importance methods and ML models.
        Returns:
            pd.DataFrame: Features with fuzzy rules.
        '''
        import numpy as np
        import skfuzzy as fuzz

        self._logger.info(f"Extracting fuzzy rules...")
        #TODO: convert target to categorical

        # Create membership functions based on interquartile values for each feature
        membership_functions = {}
        universe = {}
        for feature in df.columns[:-1]:
            # Define the universe for each feature
            universe[feature] = np.linspace(df[feature].min(), df[feature].max(), 100)
            
            # Define membership functions            
            low_mf = fuzz.zmf(universe[feature], 0.00,0.5)
            medium_mf = fuzz.trimf(universe[feature], [0.25,0.5,0.75])
            high_mf = fuzz.smf(universe[feature], 0.5, 1.00)
            
            membership_functions[feature] = {'low': low_mf, 'medium': medium_mf, 'high': high_mf}

        # Create fuzzy rules
        fuzzy_rules = []

        # Loop through each row in the dataframe and extract fuzzy rules
        for i, _ in df.iterrows():
            df_instance = {} # Dictionary to store observation values
            fuzzy_sets = {} # Dictionary to store fuzzy sets
            for feature in df.columns[:-1]:
                df_instance[feature] = df.loc[i,feature]
            # Extract fuzzy set for each feature
            for feature in df.columns[:-1]:
                fuzzy_sets[feature] = self._fuzzyset_selection(universe[feature],
                                                                membership_functions[feature]['low'],
                                                                membership_functions[feature]['medium'],
                                                                membership_functions[feature]['high'],
                                                                df_instance[feature]
                                                                )
            
            fuzzy_sets[df.columns[-1]] = df.loc[i,df.columns[-1]]
            fuzzy_rules.append(fuzzy_sets)

        # Create dataframe of fuzzy rules
        fuzzy_rules_df = pd.DataFrame(fuzzy_rules, index=df.index)

        # log fuzzy rules
        self._logger.info(f"Fiver fuzzy rules extracted: \n{fuzzy_rules_df.head(5)}")

        return fuzzy_rules_df

    
    def _local_feature_importance(self, models,  X, y):
        '''
        Calculate feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        Returns:
            dict: Dictionary of feature importance results.
        '''
        self._logger.info(f"Creating master feature importance dataframe...")
        feature_importance_results = {}

        if not any(self._local_importance_methods.values()):
            self._logger.info("No local feature importance methods selected")
        else:
            for model_type, model in models.items():
                self._logger.info(f"Local feature importance methods for {model_type}...")

                feature_importance_results[model_type] = {}

                # Run methods with TRUE values in the dictionary of feature importance methods
                for feature_importance_type, value in self._local_importance_methods.items():
                    if value['value']:
                        if feature_importance_type == 'LIME':
                            # Run LIME importance                            
                            lime_importance_df = calculate_lime_values(model, X, self._opt,self._logger)
                            # Normalise LIME coefficients between 0 and 1 (0 being the lowest impact and 1 being the highest impact)
                            lime_importance_df = lime_importance_df.abs()
                            lime_importance_df_norm = (lime_importance_df - lime_importance_df.min()) / (lime_importance_df.max() - lime_importance_df.min())
                            #Add class to local feature importance
                            lime_importance_df_norm = pd.concat([lime_importance_df_norm, y], axis=1)
                            #lime_importance_df = pd.concat([lime_importance_df, y], axis=1)
                            feature_importance_results[model_type][feature_importance_type] = lime_importance_df_norm

                        if feature_importance_type == 'SHAP':
                            # Run SHAP
                            shap_df, shap_values = calculate_shap_values(model, X, value['type'], self._opt,self._logger)
                            # Normalise SHAP values between 0 and 1 (0 being the lowest impact and 1 being the highest impact)
                            shap_df = shap_df.abs()
                            shap_df_norm= (shap_df - shap_df.min()) / (shap_df.max() - shap_df.min())
                            #Add class to local feature importance
                            shap_df_norm = pd.concat([shap_df_norm, y], axis=1)
                            feature_importance_results[model_type][feature_importance_type] = shap_df_norm
        
        # Concatenate the results
        master_df = pd.DataFrame()
        for model_type, feature_importance in feature_importance_results.items():
            for feature_importance_type, result in feature_importance.items():
                master_df = pd.concat([master_df, result], axis=0)
        
        

        return master_df




    
         
