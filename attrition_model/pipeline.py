import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


from attrition_model.config.core import config
#from attrition_model.processing.features import embarkImputer
from attrition_model.processing.features import Mapper
from attrition_model.processing.features import OutlierHandler

attrition_pipe=Pipeline([
    
   
     ##==========Mapper======##
     ("map_businesstravel", Mapper(config.model_config_.businesstravel_var, config.model_config_.businesstravel_mappings)
      ),
     ("map_dept", Mapper(config.model_config_.dept_var, config.model_config_.dept_mappings )
     ),
     ("map_edu", Mapper(config.model_config_.edu_var, config.model_config_.edu_mappings)
     ),
      ("map_gender", Mapper(config.model_config_.gender_var, config.model_config_.gender_mappings)
      ),
     ("map_jobrole", Mapper(config.model_config_.jobrole_var, config.model_config_.jobrole_mappings )
     ),
     ("map_martialsts", Mapper(config.model_config_.maritalsts_var, config.model_config_.martialsts_mappings)
     ),
      ("map_overtime", Mapper(config.model_config_.overtime_var, config.model_config_.overtime_mappings)
      ),
     #("map_attrition", Mapper(config.model_config_.attrition_var, config.model_config_.attrition_mappings )
     #),
    
       ######## Handle outliers ########
    ('handle_outliers_income_var', OutlierHandler(variable = config.model_config_.income_var)),
    ('handle_outliers_compworked_var', OutlierHandler(variable = config.model_config_.compworked_var)),
    ('handle_outliers_stock_var', OutlierHandler(variable = config.model_config_.stock_var)),
    ('handle_outliers_rating_var', OutlierHandler(variable = config.model_config_.rating_var)),
    ('handle_outliers_workyr_var', OutlierHandler(variable = config.model_config_.workyr_var)),
    ('handle_outliers_train_var', OutlierHandler(variable = config.model_config_.train_var)),
    ('handle_outliers_yrcomp_var', OutlierHandler(variable = config.model_config_.yrcomp_var)),
    ('handle_outliers_yrcurrent_var', OutlierHandler(variable = config.model_config_.yrcurrent_var)),
    ('handle_outliers_yrpromotion_var', OutlierHandler(variable = config.model_config_.yrpromotion_var)),
    ('handle_outliers_yrmgr_var', OutlierHandler(variable = config.model_config_.yrmgr_var)),
    
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config_.n_estimators, 
                                         max_depth=config.model_config_.max_depth, 
                                         max_features=config.model_config_.max_features,
                                         random_state=config.model_config_.random_state))
          
     ])
