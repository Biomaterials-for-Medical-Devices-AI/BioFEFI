# Feature importance
Once you have trained some models in your experiment, you can then perform feature importance analyses to assess which features more most influential in the models' decision making. You can get to the Feature Importance page by clinking on **Feature Importance** on the left hand side of the page.

<!-- Insert image here -->

To begin explaining your models, you can click the **"Explain all models"** toggle and have all your models evaluated, or you can use the dropdown menu to select specific models to evaluate.

<!-- Insert image here -->

## Global feature importance methods
These methods evaluate the influence of individual features overall on a model's decisions. There are two methods available.

- Permutative importance
- SHAP (<u>SH</u>apely <u>A</u>dditive ex<u>P</u>lanations)

## Ensemble feature importance methods
Ensemble methods combine results from multiple feature importance techniques, enhancing robustness. To use ensemble methods, you must configure at least one global importance method. There are two methods available.

- Mean: use the mean of importance estimates from the selected global methods.
- Majority vote: take the majority vote of importance estimates from the selected global methods.

## Local feature importance methods
These methods are used to interpret feature importance on a *per prediction* basis. You can see which features had the most influence - and in which direction - on each prediction. There are two methods available.

- LIME (<u>L</u>ocal <u>I</u>nterpretable <u>M</u>odel-agnostic <u>E</u>xplanation)
- SHAP (<u>SH</u>apely <u>A</u>dditive ex<u>P</u>lanations)

## Additional configuration options
- Number of most important features to plot

  Change how many top features will be plotted.

- Scoring function for permutative importance

- Number of repetitions for permutation importance

  The number of times to permute the features using permutative importance.

- Percentage of data to consider for SHAP

  The proportion of the data that will be used to perform SHAP analyses.

## Fuzzy feature selection
Convert features to fuzzy features and then perform feature importance analyses on the fuzzy features. To use this feature, you must first configure ensemble and local feature importance methods.

- Number of features for fuzzy interpretation

  Select the top number of features to be used for fuzzy interpretation.

- Granular features

  Check this box to perform a granular analysis of fuzzy features.

- Number of clusters for target variable

  Convert the target variable into this many clusters.

- Names of clusters (comma-separated)

  The list of names for the clusters. This should be the same length as number of clusters for target variable. The names should be separated by a comma followed by a single space. *e.g.* very low, low, medium, high, very high.

- Number of top occurring rules for fuzzy synergy analysis

  Set the number of most frequent fuzzy rules for synergy analysis.

## Select outputs to save
- Save feature importance options
- Save feature importance results

## Run the analysis
Press the **"Run Feature Importance"** button to run your analysis. Be patient as this can take a little more time than the model training.

<!-- Insert image here -->