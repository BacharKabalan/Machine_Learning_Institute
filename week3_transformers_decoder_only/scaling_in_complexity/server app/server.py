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

from inference import inference
@app.post("/tell_me_stories")

async def on_tell_me_stories(request: fastapi.Request):
  text = (await request.json())["text"]
  print("Input text:", text)
  return { "story": inference(text, 'weights_0_15000.pt') }


# In[ ]:




