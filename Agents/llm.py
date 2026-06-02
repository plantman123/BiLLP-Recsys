from typing import Union, Literal
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.schema import (
    HumanMessage
)
import openai
import os

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'
    
    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content
            
class OpenAILLM:
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get('model_name', 'gpt-3.5-turbo-16k') 
        self.temperature = kwargs.get('temperature', 0)
        self.max_tokens= kwargs.get('max_tokens', 8000)
        
        self.model_kwargs = kwargs.get('model_kwargs', {"stop": "\n"})
        self.openai_api_key = kwargs.get('openai_api_key', os.environ['OPENAI_API_KEY'])
        self.openai_api_base = kwargs.get('openai_api_base', os.environ.get('OPENAI_API_BASE', 'https://api.deepseek.com'))
    
    def __call__(self, prompt: str):
        client = openai.OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.model_kwargs.get('stop'),
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
