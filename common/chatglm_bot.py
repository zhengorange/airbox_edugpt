# -*- coding:utf-8 -*-
# @FileName  : ernie_bot.py
# @Time      : 2023/8/7
# @Author    : LaiJiahao
# @Desc      : 文心一言

import json
import requests
import os
import socket
from common.chat import TPUChatglm
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class ChatGlmBot(LLM):
    _instance = None
    _llm = None
    print(socket.gethostname())
    if socket.gethostname() != "MacBook-Pro.local":
        _llm = TPUChatglm()
        pass

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        return self._get_completion(content=prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"access_token": os.getenv("ACCESS_INFO")}

    def _get_completion(self, content: str) -> str:
        if self._llm is None:
            url = "http://8.130.113.198:25000/chatbot"
            payload = json.dumps({
                "question": content,
                "history": []
            })
            headers = {
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            result = response.json().get("result")
            return result
        res = ''
        for result_answer, _ in self._llm.stream_predict(content, []):
            res = result_answer
        return res

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ChatGlmBot()
        return cls._instance
