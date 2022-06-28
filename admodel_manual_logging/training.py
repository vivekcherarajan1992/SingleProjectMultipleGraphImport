import mlrun
import json
import joblib
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
# import dummy_ad

# dummy_ad=mlrun.function_to_module('./dummy_ad.py')
dummy_ad=mlrun.function_to_module('/User/vivek/SingleProjectMultipleGraphImport/admodel_manual_logging/dummy_ad.py')
                                  
def train_ad_model(context:mlrun.MLClientCtx,insightpak_name,instance_id:str):
    model_dir=Path('./model')
    df = pd.DataFrame(np.random.randint(0,100,size=(150, 3)),
                      columns=list('ABC'),
                      index = pd.date_range('2020-01-01', periods=150, freq='5T'))
    for col in df:
        df.loc[df.sample(frac=0.2).index, col] = np.nan
    df_json = dummy_ad.ModelHelper.convert_df_to_json(df)
    ad_model_obj = dummy_ad.DummyAD()
    ad_model_obj.train_model(df_json)
#     model1 = ad_model_obj.get_model()
    # Apply MLRun's interface for tf.keras:
    model.export_model('.')
   
    # Saved as ZIP file
    model_key=f'{insightpak_name}_{instance_id}'
    model_state_key=f'{insightpak_name}_state_{instance_id}'
    shutil.make_archive('./model/ad_model_zip', 'zip', './model/ad_model')
    context.log_model(key=model_key,model_file='./model/ad_model_zip.zip')
    
    # log joblib model
    context.log_model(key=model_state_key,model_dir='./model',model_file='ad_state.joblib')
    
    # delete model folder
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)