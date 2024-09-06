#Importação das bibliotecas
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from funcoes import *
from tensorflow.keras.models import load_model
import joblib
import const

from utilidades import *
import sys

#Implantação da API

try:
    app = Flask(__name__)
    
    modelo = load_model('meu_modelo.keras')
    
    @app.route('/predict', methods = ['POST'])
    
    def predictions():
        input_data = request.get_json()
    
        df = pd.DataFrame(input_data)
        df = load_scalers(df,['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal'] )
        df = load_encoders(df, ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto'])
    
        predictions = modelo.predict(df)
    
        return jsonify(predictions.tolist())
    
    
    if __name__ == '__main__':
        app.run('0.0.0.0', port = 5000, debug = False)
    
    
    
except Exception as e:
    print(f'An error has ocurred{e}')
        
    
    sys.exit(1)