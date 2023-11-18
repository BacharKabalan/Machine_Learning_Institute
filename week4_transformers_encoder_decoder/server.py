#!/usr/bin/env python
# coding: utf-8

# In[5]:


import fastapi

app = fastapi.FastAPI()

@app.on_event("startup")
async def startup_event():
    print("Starting up")
    
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down")


@app.get("/")
def on_root():
    return { "message": "Hello App" }

import bash_gpt_inference
@app.post("/from_txt_to_sql")
async def on_learn_to_search(request: fastapi.Request):
  text = (await request.json())["text"]
#   ctx = (await request.json())["ctx"]
  print("Input txt:", text)
#   print("Input ctx:", ctx)
  return { "story": bash_gpt_inference.inference(text,'multi_head_with_pos_encod_weights_0_100000.pt') }




# In[ ]:




