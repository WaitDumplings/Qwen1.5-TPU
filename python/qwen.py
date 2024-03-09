#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import argparse
import time
# from tokenization_util import make_context
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

class Qwen:
    def __init__(self, args):
        self.input_str = ""
        self.system_prompt = "You are a helpful assistant."
        self.history = []
        self.messages = [{"role": "system", "content": self.system_prompt}]

        # load tokenizer
        print("Load " + args.token + " ...")
        self.sp = AutoTokenizer.from_pretrained(args.token, trust_remote_code=True)
        
        # warm up
        self.sp.decode([0]) 
        self.EOS = self.sp.eos_token_id
        print("Done!")

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.DEVIO)
        self.handle = sail.Handle(args.dev_id)
        self.graph_names = self.net.get_graph_names()

        # initialize qwen parameters
        self.NUM_LAYERS = (len(self.graph_names) - 2) // 2
        self.first_hidden_input_shape = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        _, self.SEQLEN, self.HIDDEN_SIZE = self.first_hidden_input_shape

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
    def generate_tokens(self, input_str):
        tokens = make_context(self.sp,
                                input_str,
                                history=self.history,
                                system=self.system_prompt,
                                max_window_size=self.SEQLEN,
                                chat_format="chatml")
        return tokens

    # inference for the first token
    def forward_first(self, token):
        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input["dtype"]))
        input_ids[:min(self.SEQLEN, len(token))] = token
        input_ids = input_ids.reshape(1, -1)
        self.token_length = len(token)
        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid["dtype"])) 
        for i in range(self.token_length):
            position_id[i] = i

        attention_mask = np.ones(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention["dtype"])) * (-10000.0)
        for i in range(self.token_length):
            for j in range(self.SEQLEN):
                if (j <= i):
                    attention_mask[i*self.SEQLEN + j] = 0
        
        # embedding
        self.first_embed_input["data"].update_data(input_ids)
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}

        # Embedding Layer Inference
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(position_id.reshape(self.first_pid["shape"]))
        self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"]))) # set bf16 in the future.

        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor, 
                                self.first_pid["name"]: self.first_pid["data"], 
                                self.first_attention["name"]: self.first_attention["data"]}

        # Transformer Block Inference
        for i in range(self.NUM_LAYERS):        
            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.past_key_output[i]["data"],
                                    self.past_value_output[i]["name"]: self.past_value_output[i]["data"]}
            
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
        
        # get the last token info as Lm head input
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        
        # Lm_head Inference
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())

    # The following tokens prediction
    def forward_next(self, ):
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention["dtype"]))
        for i in range(self.token_length-1, self.SEQLEN):
            attention_mask[i] = -10000.0
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid["dtype"]))

        # embedding
        self.next_embed_input["data"] = self.lm_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        # Embedding Layer Inference
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(position_id.reshape(self.next_pid["shape"]))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        # Transformer Block Inference
        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.present_key["data"],
                                        self.cache_value_output[i]["name"]: self.present_value["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

            # update kv_cache()
            unit_size = self.present_key["shape"][-1]*self.present_key["shape"][-2]
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.token_length-1)*unit_size, unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.token_length-1)*unit_size, unit_size)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        
        # Lm_head Inference
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())
        
    def chat(self):

        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            if self.input_str == "exit":
                break

            # tokens_with_template = self.generate_tokens(self.input_str)
            self.messages.append({"role":"user","content":self.input_str})
            text = self.sp.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.sp(text).input_ids

            print("\nAnswer: ")
            self.answer(model_inputs)

    def answer(self, tokens):
        tok_num = 0
        self.answer_cur = ""

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.SEQLEN:
            print("The maximum question length should be shorter than {} but we get {} instead.".format(self.SEQLEN, self.token_length))
            return
        
        # First token
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()

        # Following tokens
        while token != self.EOS and self.token_length < self.SEQLEN:
            diff = self.sp.decode([token])
            self.answer_cur += diff
            print(diff, flush=True, end='')
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            tok_num += 1
            token = self.forward_next()
        
        # counting time
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration

        if self.token_length >= self.SEQLEN - 128:
            print("... (reach the maximal length)", flush=True, end='')
            self.history.clear()
            self.messages = self.messages[0]
            self.messages.append({"role": "user", "content": self.input_str})
            self.messages.append({"role": "assistant", "content": self.answer_cur})
        else:
            self.messages.append({"role": "assistant", "content": self.answer_cur})

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def main(args):
    qwen = Qwen(args)
    qwen.chat()
    pass

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='./qwen1.5-1.8b_int8_1dev.bmodel', help='path of bmodel')
    parser.add_argument('--token', type=str, default='./token_config/', help='path of tokenizer')
    parser.add_argument('--dev_id', type=int, default=1, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')