# -*- coding:utf-8 -*-
# @FileName  : config.py
# @Time      : 2023/7/16
# @Author    : LaiJiahao
# @Desc      : 函数配置文件

import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import BasePromptTemplate
from langchain.chains import LLMChain
from common.ernie_bot import ErnieBot
from common.chatglm_bot import ChatGlmBot


def get_openai_proxy():
    return os.getenv("OPENAI_API_PROXY")


def get_access_info():
    return os.getenv("ACCESS_INFO")


class Config:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        os.environ["OPENAI_API_KEY"] = "sk"

    def get_llm(self):
        llm = self._get_model()
        return llm

    @staticmethod
    def get_SESSDATA():
        return os.getenv("SESSDATA")

    @staticmethod
    def create_llm_chain(llm, prompt: BasePromptTemplate) -> LLMChain:
        return LLMChain(llm=llm, prompt=prompt)

    @staticmethod
    def _get_model():
        return ChatGlmBot.get_instance()
