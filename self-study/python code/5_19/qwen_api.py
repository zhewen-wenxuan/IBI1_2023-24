import os
import time
from http import HTTPStatus #见ppt 关于http request的状态，如果程序正常执行，是没有什么作用的，但是如何程序报错或者被killer，那么就可以通过http状态来进行debug
import dashscope
from dashscope import Generation #dashscope 是一个模块或包，专门针对Qwen API 调用的包。而 Generation 是该模块中的一个类（Class）


class DashScopeAssistant:
    def __init__(self, api_key):
        # 设置 API 密钥
        dashscope.api_key = api_key #在 dashscope 模块中存在一个名为 api_key 的变量
        # openai.api_key = api_key
    def get_answer(self, question): #自己定义一个方法，该方法是为了调用API来实现某些功能，本质就是对prompt的处理
        start_time = time.time()
        messages = [{'role': 'user', 'content': question}] #见ppt
        responses = Generation.call(  #定义responses这个变量名，用来保存API返回的结果  使用的是dashscope库中的generation类的call方法，然后可以得到我们API的处理结果
            model='qwen-plus',
            max_tokens=1500,
            messages=messages,  #传入prompt
            result_format='message',  #格式同源
            stream=True,   #结果将以流的形式逐步输出，而不是一次性地将所有结果返回
            incremental_output=True #采用增量输出的方式，结果将逐步生成，而不是等待所有结果都生成完毕后再一次性返回。
        )

        full_content = ''  #这是返回的内容，我们定义full_content这个字符串来存储
        for response in responses:  #上面我们会发现，responses是一个列表，里面包括了很多的request处理后的信息 然后我们进行for循环遍历
            if response.status_code == HTTPStatus.OK:  #这个是根据HTTPStatus 状态 ==200，说明一切ok，否则就会出现报错。并且我们会发现，responses中的一条response包括status_code这个键值对。
                full_content += response.output.choices[0]['message']['content'] #我们将response中output对应的choice第一天进行输出，然后将其加到full_content中去。
            else:  #否则，状态出错，我这边会输出那种http状态，方便我们解决问题
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
        end_time = time.time()
        final_time = end_time - start_time
        print("模型回复用时：" + str(final_time))
        return full_content

# 创建 DashScopeAssistant 实例





