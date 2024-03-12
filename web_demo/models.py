#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
from transformers import AutoTokenizer
import numpy as np

#convert sail_dtype to numpy dtype
def type_convert(sail_dtype):
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    if sail_dtype == sail.Dtype.BM_BFLOAT16: # 后续需要修改bf16的接口,现在先用fp16的代替
        return np.float16
    
    raise TypeError("only support float32 and int32 right now")

def fp16_cast(arr:np.ndarray): #这个接口的作用在于把np.float16假冒成np.uint16传进Tensor，sail update_data如果能接收传输二进制，那就不需要这个了。(后续需要改成bf16的接口)
    """
    reinterpret an array with int16 instead of float16, because pybind11 do not support float16.
    """
    if arr.dtype == np.float16:
        return arr.view(np.uint16)
    else:
        return arr

class BaseModel:
    def __init__(self, model_path, token_path, dev_id, Llama_like=True):
        self.input_str = ""
        self.system_prompt = ""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.token_path = token_path
        self.model_path = model_path
        self.dev_id = dev_id
        self.Llama_like = Llama_like
        self.kv_update_first = False
        self.kv_update_next = False

        # load tokenizer
        print("Load " + token_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_path, trust_remote_code=True)

        # warm up
        self.tokenizer.decode([0]) 
        self.EOS = self.tokenizer.eos_token_id
        print("Done!")

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = sail.Engine(self.model_path, self.dev_id, sail.IOMode.DEVIO)
        self.handle = sail.Handle(dev_id)
        self.graph_names = self.net.get_graph_names()

        # initialize qwen parameters
        self.NUM_LAYERS = (len(self.graph_names) - 2) // 2
        self.first_hidden_input_shape = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        self.SEQLEN, tmp, self.HIDDEN_SIZE = self.first_hidden_input_shape if not self.Llama_like else (self.first_hidden_input_shape[1], 1, self.first_hidden_input_shape[2])

        # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]

        # initialize tensors (inputs & outputs)
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN])
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN, self.HIDDEN_SIZE], False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, [1,  self.HIDDEN_SIZE], False)

        # forward_first: hidden_state
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_first: position_id_tensor and attention_mask_tensor
        self.first_pid = self.init_sail_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_sail_tensor(self.name_blocks[0], 2)
       
        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_sail_tensor(self.name_blocks_cache[0], 2)

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False)

        # forward_first: key_tensor and value_tensor
        self.past_key_output = []
        self.past_value_output = []

        # forward_next: kv cache block 
        self.cache_key_input = []
        self.cache_key_output = []
        self.cache_value_input = []
        self.cache_value_output = []

        for _ in range(self.NUM_LAYERS): 
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0], 2, None, False))

            self.cache_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 3))
            self.cache_key_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False))

            self.cache_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 4))
            self.cache_value_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        self.token_length = 0

        self.unit_size = self.first_embed_output["shape"][-1] if self.Llama_like else np.prod(self.present_key["shape"][1:])
        print("Initial Complete")
    
    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        if is_input:
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True) 
        return tensor

    # Add prompt template and output tokens
    def generate_tokens(self, messages):
        pass

    def forward_first(self, token):
        pass

    # The following tokens prediction
    def forward_next(self):
        pass
    
    def stream_predict(self, query, messages):
        if not messages:
            messages = [{"role": "system", "content": self.system_prompt}]

        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
            
        messages.append({"role":"user","content":query})
        tokens = self.generate_tokens(messages)
        messages.append({"role":"assistant","content":""})
        res = ''
        first_token = self.forward_first(tokens)
        output_tokens = [first_token]
        while True:
            next_token = self.forward_next()
            if next_token == self.EOS:
                break
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            output_tokens += [next_token]
            res = self.tokenizer.decode(output_tokens)
            messages[-1] = {"role":"assistant","content":res}
            yield res, messages
    
    
    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.token_length = 0

class GLM(BaseModel):
    def __init__(self, model_path, token_path, dev_id, Llama_like=False):
        super().__init__(model_path, token_path, dev_id, Llama_like)
        self.system_prompt = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
        self.kv_update_first = True

    def generate_tokens(self, messages): 
        input_str = messages[-1]["content"]
        tokens = self.tokenizer.build_chat_input(input_str, history=messages[:-1], role="user")
        return tokens
    
    def forward_first(self, token):
       # Keep
        # print("history length: ",len(token))
        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input["dtype"]))
        input_ids[:min(self.SEQLEN, len(token))] = token
        self.token_length = len(token)
        input_ids = input_ids.reshape(1, -1)

        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid["dtype"])) 
        for i in range(self.token_length):
            position_id[i] = i
            
        attention_mask = np.zeros(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention["dtype"])) #这里的type要从模型获取。
        for i in range(self.SEQLEN):
            for j in range(self.SEQLEN):
                if not (j <= i and i < self.token_length):
                    attention_mask[i*self.SEQLEN + j] = -10000.0
        # embedding
        self.first_embed_input["data"].update_data(fp16_cast(input_ids))
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(fp16_cast(position_id.reshape(self.first_pid["shape"])))
        self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"])))

        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor, 
                                self.first_pid["name"]: self.first_pid["data"], 
                                self.first_attention["name"]: self.first_attention["data"]}

        for i in range(self.NUM_LAYERS):        
            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.present_key["data"],
                                    self.past_value_output[i]["name"]: self.present_value["data"],}
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
            
            unit_size = np.prod(self.present_key["shape"][1:])
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.SEQLEN - self.token_length)*unit_size, self.token_length * unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.SEQLEN - self.token_length)*unit_size, self.token_length * unit_size)

        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}

        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())

    # The following tokens prediction
    def forward_next(self):
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention["dtype"]))
        for i in range(self.SEQLEN - self.token_length + 1):
            attention_mask[i] = -10000.0
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid["dtype"]))

        # embedding
        self.next_embed_input["data"] = self.lm_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(fp16_cast(position_id.reshape(self.next_pid["shape"])))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.past_key_output[i]["data"],
                                        self.cache_value_output[i]["name"]: self.past_value_output[i]["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy()) #int32

class Qwen1_5(BaseModel):
    def __init__(self, model_path, token_path, dev_id, Llama_like=True):
        super().__init__(model_path, token_path, dev_id, Llama_like)
        self.system_prompt = "You are a helpful assistant."
        self.kv_update_next = True

    def generate_tokens(self, messages):   
        text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        tokens = self.tokenizer(text).input_ids
        return tokens

    def forward_first(self, token):
        # Keep
        # print("history length: ",len(token))
        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input["dtype"]))
        input_ids[:min(self.SEQLEN, len(token))] = token
        self.token_length = len(token)
        input_ids = input_ids.reshape(1, -1)

        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid["dtype"])) 
        for i in range(self.token_length):
            position_id[i] = i
            
        attention_mask = np.zeros(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention["dtype"])) #这里的type要从模型获取。
        for i in range(self.SEQLEN):
            for j in range(self.SEQLEN):
                if not (j <= i and i < self.token_length):
                    attention_mask[i*self.SEQLEN + j] = -10000.0
        # embedding
        self.first_embed_input["data"].update_data(fp16_cast(input_ids))
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(fp16_cast(position_id.reshape(self.first_pid["shape"])))
        self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"])))

        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor, 
                                self.first_pid["name"]: self.first_pid["data"], 
                                self.first_attention["name"]: self.first_attention["data"]}

        for i in range(self.NUM_LAYERS):        
            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.present_key["data"],
                                    self.past_value_output[i]["name"]: self.present_value["data"],}
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
            
            unit_size = np.prod(self.present_key["shape"][1:])
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.SEQLEN - self.token_length)*unit_size, self.token_length * unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.SEQLEN - self.token_length)*unit_size, self.token_length * unit_size)

        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}

        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())

    def forward_next(self, ):
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention["dtype"]))
        for i in range(self.SEQLEN - self.token_length + 1):
            attention_mask[i] = -10000.0
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid["dtype"]))

        # embedding
        self.next_embed_input["data"] = self.lm_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(fp16_cast(position_id.reshape(self.next_pid["shape"])))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.past_key_output[i]["data"],
                                        self.cache_value_output[i]["name"]: self.past_value_output[i]["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy()) #int32
    