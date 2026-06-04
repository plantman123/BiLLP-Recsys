import json
import argparse
import concurrent.futures
import random
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
from functools import partial
from Agents.agent_reflexion import ReactReflectAgent
from Agents.agent_retrival import ReactReflectRetrivalAgent
from Agents.agent_revise import ReactReflectReviseAgent
from Agents.agent_a2c import ReactA2CAgent
from models.openai import chatgpts, gpts
from models.llama import LlamaInterface

from tasks import get_task
from env import get_envs, get_groundingmodel
from tools import call_tools
from tools.search import search_save
from datetime import datetime
import re
import os
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def get_fewshot_prompt(promptpath, task=None, chatgpt_format=False):
    if len(promptpath) == 0:
        return [] if chatgpt_format else ""
    elif promptpath == "default" and task is not None:
        return task.get_prompt()
    if not chatgpt_format:
        with open(f"./prompts/{promptpath}.txt", "r") as fin:
            prompt = fin.read() 
        return prompt
    else:
        with open(f"./prompts/{promptpath}.json", "r") as fin:
            prompt = json.load(fin)
        return prompt

def prepare_prompt(question):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

def prune_thought(prompt):
    if prompt.startswith("Thought:"):
        return prompt[len("Thought:"):].strip()
    return prompt

def save_info(infos, outfilename):
    if 'trajs' in infos:
        traj_file_name = f"output/trajs_agent/{outfilename}.json"
        with open(traj_file_name, "w") as fout:
            json.dump(infos['trajs'], fout, indent=2)
    
    if 'reflections' in infos:
        reflection_file_name = f'output/reflections/{outfilename}.txt'
        with open(reflection_file_name, 'w') as file:
            for item in infos['reflections']:
                file.write(str(item) + '\n')
    
    if 'Q_table' in infos:
        memory_file_name = f'output/memory/{outfilename}.json'
        with open(memory_file_name, "w") as fout:
            json.dump(infos['Q_table'], fout, indent=2, cls=NpEncoder)
            
    if 'actor_memory' in infos:
        memory_file_name = f'output/memory/{outfilename}.json'
        with open(memory_file_name, "w") as fout:
            json.dump(infos['actor_memory'], fout, indent=2, cls=NpEncoder)
    
    if 'critic_memory' in infos:
        critic_memory_file_name = f'output/critic_memory/{outfilename}.json'
        with open(critic_memory_file_name, "w") as fout:
            json.dump(infos['critic_memory'], fout, indent=2, cls=NpEncoder)

      
def load_info(input_file_name):
    if input_file_name == None:
        return None, None, None
    
    reflection_file_name = f'reflections/{input_file_name}.txt'
    if os.path.exists(reflection_file_name):
        reflections = []
        with open(reflection_file_name, "r") as file:
            for line in file:
                line = line.strip()  # 去除行尾的换行符和空白字符
                reflections.append(line)
    else:
        reflections = None
        
    memory_file_name = f'memory/{input_file_name}.json'
    if os.path.exists(memory_file_name):
        with open(memory_file_name, 'r') as file:
            Q_Memory = json.load(file)
    else:
        Q_Memory = None
    
    critic_memory_file_name = f'critic_memory/{input_file_name}.json'
    if os.path.exists(critic_memory_file_name):
        with open(critic_memory_file_name, 'r') as file:
            Critic_Memory = json.load(file)
    else:
        Critic_Memory = None
    
    return reflections, Q_Memory, Critic_Memory
    
'''
inital prompt -> agents
while true
    action, action_type = agents.run(observation, reward)
    observation, reward = env.step()
'''



def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='gpt-4')
    args.add_argument('--agent_name', type=str, default='reflexion', required=True)
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True)
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--evaluate', action='store_true')
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')
    args.add_argument('--alpaca_format', action='store_true')
    args.add_argument('--chatgpt_format', action='store_true')
    args.add_argument('--question_prefix', type=str, default='')

    args.add_argument('--modelpath', type=str, default='./model/shakechen/Llama-2-7b-hf')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--promptpath', type=str, default='')
    args.add_argument('--env_path', type=str, default='./env')
    args.add_argument('--grounding_model_path', type=str, default='./model/shakechen/Llama-2-7b-hf')
    
    args.add_argument('--env', type=str, required=True)
    args.add_argument('--env_window_length', type=int, default=5)
    args.add_argument('--env_threshold', type=float, default=-1)
    
    args.add_argument('--Max_Iteration', type=int, default=11)
    args.add_argument('--Max_Reflections', type=int, default=2)
    args.add_argument('--batch_size', type=int, default=5)
    args.add_argument('--traj', action='store_true')
    args.add_argument('--change_examples', action='store_true')
    args.add_argument('--input_file_name', default=None)
    args.add_argument('--max_tokens', type=int, default=6000,
                      help="Model output token limit. Default 6000 matches the original generation script.")
    args.add_argument('--rerank_after_grounding', action='store_true',
                      help="After grounding an actor action to Top-K real items, ask the actor LLM to choose from those candidates.")
    args.add_argument('--grounding_topk', type=int, default=5,
                      help="Number of grounded candidates shown to the actor LLM when --rerank_after_grounding is enabled.")
    args.add_argument('--reflection_retrieval_mode', type=str, default='original',
                      choices=['original', 'episode', 'dynamic', 'hybrid'],
                      help="Planner reflection retrieval timing. original preserves the paper code path; the other modes enable the retrieval experiment.")
    args.add_argument('--static_reflection_k', type=int, default=0,
                      help="Number of episode-start reflections for hybrid/episode mode. 0 means use --Max_Reflections.")
    args.add_argument('--dynamic_reflection_k', type=int, default=0,
                      help="Number of current-state reflections for dynamic/hybrid mode. 0 means use --Max_Reflections.")
    args.add_argument('--reflection_query_window', type=int, default=15,
                      help="Number of latest items used to build current-state reflection queries; 15 matches the original task question format.")
    args.add_argument('--reflection_memory_policy', type=str, default='full',
                      choices=['full', 'fifo', 'lru'],
                      help="Planner reflection memory retention policy. full keeps all reflections, fifo keeps newest N, lru evicts least recently retrieved reflections.")
    args.add_argument('--reflection_memory_size', type=int, default=0,
                      help="Maximum number of planner reflections to keep for fifo/lru. 0 disables pruning.")
    args.add_argument('--run_name', type=str, default='',
                      help="Optional label included in the output filename.")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = get_task(args.task, args.task_split)

    random.seed(0)

    MAX_TOKENS = args.max_tokens

    modelname = args.backend
    if args.backend == 'llama':
        pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
        modelname += f"_{pathname}"

    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    static_k = args.static_reflection_k if args.static_reflection_k > 0 else args.Max_Reflections
    dynamic_k = args.dynamic_reflection_k if args.dynamic_reflection_k > 0 else args.Max_Reflections
    query_window = max(1, args.reflection_query_window)
    if args.run_name:
        experiment_tag = re.sub(r'[^A-Za-z0-9_.-]+', '-', args.run_name.strip())
    elif args.reflection_retrieval_mode == 'hybrid':
        experiment_tag = f"retrieval-hybrid-s{static_k}-d{dynamic_k}-q{query_window}"
    elif args.reflection_retrieval_mode == 'original':
        experiment_tag = ''
    else:
        retrieval_k = static_k if args.reflection_retrieval_mode == 'episode' else dynamic_k
        experiment_tag = f"retrieval-{args.reflection_retrieval_mode}-k{retrieval_k}-q{query_window}"
    filename_parts = [
        args.task,
        args.task_split,
        str(args.task_start_index),
        str(args.task_end_index),
        modelname,
        str(args.temperature),
    ]
    if experiment_tag:
        filename_parts.append(experiment_tag)
    filename_parts.append(time_str)
    outfilename = '_'.join(filename_parts)
    print(outfilename)

    idxs_all = list(range(len(task)))
    if args.random:
        random.Random(233).shuffle(idxs_all)
    idxs = idxs_all[args.task_start_index:args.task_end_index]

    if args.backend == "llama":
        print(args.modelpath, args.peftpath, args.add_lora)
        llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
        model = partial(llama.generate_responses_from_llama, temperature=args.temperature, stop=['\n', 'Action', 'Observation', 'Thought'])
    elif args.chatgpt_format:
        model = partial(chatgpts, model=args.backend, temperature=args.temperature, max_tokens=MAX_TOKENS, stop='\n')
    else:
        model = partial(gpts, model=args.backend, temperature=args.temperature, max_tokens=MAX_TOKENS, stop='\n')
    
    envs = get_envs(args.env, args, args.task_split)
    grounding_model = get_groundingmodel(args.env, args.grounding_model_path, args, args.task_split) 
    
    reflections, Q_Memory, Critic_Memory = load_info(args.input_file_name)
    
    if args.agent_name == "agent_reflection":
        agent = ReactReflectAgent(task, idxs, args, envs, grounding_model, max_steps=args.Max_Iteration, react_llm=model, reflect_llm=model, reflections_memory=reflections)
    elif args.agent_name == "agent_retrival":
        agent = ReactReflectRetrivalAgent(task, idxs, args, envs, grounding_model, max_steps=args.Max_Iteration, react_llm=model, reflect_llm=model, reflections_memory=reflections, Q_memory=Q_Memory)
    elif args.agent_name == "agent_a2c":
        agent = ReactA2CAgent(task, idxs, args, envs, grounding_model, max_steps=args.Max_Iteration, react_llm=model, reflect_llm=model, critic_llm=model, reflections_memory=reflections, actor_memory=Q_Memory, critic_memory=Critic_Memory)
    elif args.agent_name == "agent_revise":
        agent = ReactReflectReviseAgent(task, idxs, args, envs, grounding_model, max_steps=args.Max_Iteration, react_llm=model, reflect_llm=model)

    infos = agent.run(outfilename=outfilename)

    save_info(infos, outfilename)
