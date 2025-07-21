# import google.generativeai as genai
# import os 
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# llm = genai.GenerativeModel(model_name='gemini-2.0-flash')



# def get_genai_response(input,img,prompt):
#     response  = llm.generate_content([input,img,prompt])
#     return response.text