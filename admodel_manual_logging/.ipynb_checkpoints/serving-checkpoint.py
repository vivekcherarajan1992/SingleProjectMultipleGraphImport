import os
import gc
import mlrun
from mlrun.serving import V2ModelServer
import json
from zipfile import ZipFile
import tempfile
from mlrun.artifacts import get_model
import dummy_ad
from tensorflow import keras

# dummy_ad=mlrun.function_to_module('/User/vivek/SingleProjectMultipleGraphImport/admodel_manual_logging/dummy_ad.py')

def preprocess(event: dict):    
    return event
    
class ModelOnDemandServer(V2ModelServer):
    def __init__(
        self,
        context: mlrun.MLClientCtx=None,
        name: str = None,
        input_path: str = None,
        result_path: str = None,
        **kwargs,
    ):
        # V2ModelServer initialization with no models:
        super().__init__(
            context=context,
            name=name,
            model_path=None,
            model=None,
            protocol=None,
            input_path=input_path,
            result_path=result_path,
            **kwargs
        )
        
        # Mark the server as ready for '_post_init' to not call 'load':
        self.ready = True    
    
    def predict(self, event:dict):
        # Unpacking event:
        inputsList = event['inputs']
        model_inputs=json.dumps(inputsList[0]['inputs'])
        models=inputsList[0]['models']
        instanceId=inputsList[0]['instanceId']
        
        # Loading the model:
        print("Loading model...")
        models_path=models[f'ad_model_{instanceId}']
        tmp = tempfile.TemporaryDirectory()
        model_file,model_obj, _ = get_model(models_path)
        model_file = ZipFile(model_file, 'r')
        model_file.extractall(tmp.name)        
        model=keras.models.load_model(tmp.name)
                
        print(type(model))

        # Inferring thourgh the model:
        dummy_ad_obj = dummy_ad.DummyAD()
        outputs=dummy_ad_obj.predict(model,model_inputs)        
        
        # model response
        print("prediction : ",outputs)
        
        # Deleting model:
        print("Releasing model from memory...")
        del model
        gc.collect()
        
        return outputs


def postprocess(event: dict):
    print("Post processing ......")
    return event
    