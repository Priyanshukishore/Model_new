# from flask import Flask
# from azure.storage.blob import BlobServiceClient
# import pandas as pd
# import pickle
# import json
# import os

# app = Flask(__name__)

# # ---- Load ML Model (.pkl) ----
# with open('force_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # ---- Azure Blob Storage Details ----
# source_connect_str = os.getenv('SOURCE_CONNECTION_STRING', "DefaultEndpointsProtocol=https;AccountName=priystorage;AccountKey=MPVPHi8uvHh1hx1Z+3dabHedORLlVaKIyR35euAisSQVMFyzqzuFxv3Ty5sCxB8YuMl51EhV9hjC+ASt3gBCGQ==;EndpointSuffix=core.windows.net")
# source_container_name = "storagefortest"
# source_blob_name = "0_b1556fa28096485bb60325539f0f49c9_1.json"

# target_connect_str = os.getenv('TARGET_CONNECTION_STRING', "DefaultEndpointsProtocol=https;AccountName=priystorage;AccountKey=MPVPHi8uvHh1hx1Z+3dabHedORLlVaKIyR35euAisSQVMFyzqzuFxv3Ty5sCxB8YuMl51EhV9hjC+ASt3gBCGQ==;EndpointSuffix=core.windows.net")
# target_container_name = "storagefortest"
# target_blob_name = "0_e5aab7b9883a4f308b0cc6ed21705012_1.json"

# @app.route('/')
# def home():
#     return "✅ Azure Web App Running - Ready for Prediction!"

# @app.route('/predict')
# def predict():
#     try:
#         # ---- Download JSON from Source Blob ----
#         source_blob_service = BlobServiceClient.from_connection_string(source_connect_str)
#         source_blob_client = source_blob_service.get_blob_client(container=source_container_name, blob=source_blob_name)
#         download_stream = source_blob_client.download_blob()
#         json_data = json.loads(download_stream.readall())

#         # ---- Convert JSON to DataFrame ----
#         df = pd.DataFrame(json_data)
        
#         # ---- Prediction ----
#         df['prediction'] = model.predict(df)
        
#         # ---- Convert to JSON ----
#         prediction_json = df.to_json(orient='records')
        
#         # ---- Upload Prediction JSON to Target Blob ----
#         target_blob_service = BlobServiceClient.from_connection_string(target_connect_str)
#         target_blob_client = target_blob_service.get_blob_client(container=target_container_name, blob=target_blob_name)
#         target_blob_client.upload_blob(prediction_json, overwrite=True)
        
#         return "✅ Prediction done & uploaded to Target Blob!"

#     except Exception as e:
#         return f" Error: {str(e)}"

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8000))  # Important for Azure Web Apps
#     app.run(host='0.0.0.0', port=port)


from flask import Flask
from azure.storage.blob import BlobServiceClient
import pandas as pd
import pickle
import json
import os

app = Flask(__name__)

# ---- Load ML Model (.pkl) ----
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---- Azure Blob Storage Details ----
source_connect_str = os.getenv('SOURCE_CONNECTION_STRING', "DefaultEndpointsProtocol=https;AccountName=priystorage;AccountKey=MPVPHi8uvHh1hx1Z+3dabHedORLlVaKIyR35euAisSQVMFyzqzuFxv3Ty5sCxB8YuMl51EhV9hjC+ASt3gBCGQ==;EndpointSuffix=core.windows.net")
source_container_name = "storagefortest"
source_blob_name = "0_b1556fa28096485bb60325539f0f49c9_1.json"

target_connect_str = os.getenv('TARGET_CONNECTION_STRING', "DefaultEndpointsProtocol=https;AccountName=priystorage;AccountKey=MPVPHi8uvHh1hx1Z+3dabHedORLlVaKIyR35euAisSQVMFyzqzuFxv3Ty5sCxB8YuMl51EhV9hjC+ASt3gBCGQ==;EndpointSuffix=core.windows.net")
target_container_name = "storagefortest"
target_blob_name = "0_e5aab7b9883a4f308b0cc6ed21705012_1.json"

@app.route('/')
def home():
    return "✅ Azure Web App Running - Ready for Prediction!"

@app.route('/predict')
def predict():
    try:
        # ---- Download JSON from Source Blob ----
        source_blob_service = BlobServiceClient.from_connection_string(source_connect_str)
        source_blob_client = source_blob_service.get_blob_client(container=source_container_name, blob=source_blob_name)
        download_stream = source_blob_client.download_blob()
        json_data = json.loads(download_stream.readall())

        # ---- Convert JSON to DataFrame ----
        df = pd.DataFrame(json_data)
        
        # ---- Prediction ----
        df['prediction'] = model.predict(df)
        
        # ---- Convert to JSON ----
        prediction_json = df.to_json(orient='records')
        
        # ---- Upload Prediction JSON to Target Blob ----
        target_blob_service = BlobServiceClient.from_connection_string(target_connect_str)
        target_blob_client = target_blob_service.get_blob_client(container=target_container_name, blob=target_blob_name)
        target_blob_client.upload_blob(prediction_json, overwrite=True)
        
        # ---- Convert DataFrame to HTML Table ----
        table_html = df.to_html(classes='table table-bordered table-striped', index=False)

        # ---- Return Data & Prediction on Webpage ----
        return f"""
        <h2>📥 Input Data & Prediction Result 📊</h2>
        {table_html}
        <p>✅ Prediction uploaded to Target Blob!</p>
        """

    except Exception as e:
        return f"<h3>❌ Error: {str(e)}</h3>"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Important for Azure Web Apps
    app.run(host='0.0.0.0', port=port)

