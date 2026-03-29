import google.generativeai as genai

genai.configure(api_key="AIzaSyD8WWMESMQ73wWJpB_RmakPvo02zm7gXfE")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)