import google.generativeai as genai

genai.configure(api_key="AIzaSyCCMIphUCd0UVZc-UHlm1h3kEQiSEjtCMc")

models = genai.list_models()

for m in models:
    print("NAME:", m.name)
    print("SUPPORTED:", m.supported_generation_methods)
    print("-" * 50)