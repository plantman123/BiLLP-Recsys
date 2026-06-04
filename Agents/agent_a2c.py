import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from Agents.llm import AnyOpenAILLM, OpenAILLM
from Agents.prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, react_reflect_retrival_agent_prompt, critic_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from Agents.prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from Agents.fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from Agents.agent_base import ReactAgent, parse_action, format_step, truncate_scratchpad
from Agents.agent_reflexion import ReactReflectAgent, ReflexionStrategy
import random
from collections import defaultdict
import json
import numpy as np
import openai
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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

class ReactA2CAgent(ReactReflectAgent):
    def __init__(self,
                 task,
                 idxs: list, 
                 args, 
                 rec_env,
                 grounding_model,
                 max_steps: int = 30,
                 agent_prompt: PromptTemplate = react_reflect_retrival_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                             temperature=0,
                                             max_tokens=8000,
                                             model_name="gpt-3.5-turbo-16k",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY'],
                                             openai_api_base = os.environ['OPENAI_API_BASE']),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                                               temperature=0,
                                               max_tokens=8000,
                                               model_name="gpt-3.5-turbo-16k",
                                               openai_api_key=os.environ['OPENAI_API_KEY'],
                                               openai_api_base = os.environ['OPENAI_API_BASE']),
                 critic_llm: AnyOpenAILLM = AnyOpenAILLM(
                                               temperature=0,
                                               max_tokens=8000,
                                               model_name="gpt-3.5-turbo-16k",
                                               openai_api_key=os.environ['OPENAI_API_KEY'],
                                               openai_api_base = os.environ['OPENAI_API_BASE']),
                 reflections_memory = None,
                 actor_memory = None,
                 critic_memory = None,
                 ) -> None:
        
        super().__init__(task, idxs, args, rec_env, grounding_model, max_steps, agent_prompt, react_llm, reflect_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.critic_prompt = critic_prompt
        self.reflect_examples = REFLECTIONS
        self.critic_llm = critic_llm
        self.reflections_str: dict = {}
        self.episode_reflections_str: dict = {}
        self.reflection_retrieval_records = defaultdict(list)
        self.grounding_rerank_records = defaultdict(list)
        self._last_formatted_reflections = {}
        
        
        if reflections_memory == None:
            self.reflections: list = []
            self.faiss_reflections = None
        else:
            self.reflections = reflections_memory
            self.faiss_reflections = None
        self._reflection_clock = 0
        self._reflection_usage = {}
        self._initialize_reflection_usage()
        self._apply_reflection_memory_policy()
        if self.reflections:
            self._update_reflections_lib()
        
        if actor_memory == None:
            self.actor_memory: dict =defaultdict(dict) 
            self.faiss_actor_memory = None
        else:
            self.actor_memory = actor_memory
            embeddings = HuggingFaceEmbeddings(model_name="./model/sentence-transformers/all-MiniLM-L6-v2")
            self.faiss_actor_memory = FAISS.from_texts(self.actor_memory.keys(), embeddings)

        if critic_memory == None:
            self.critic_memory: dict =defaultdict(dict) 
            self.faiss_critic_memory = None
        else:
            self.critic_memory = critic_memory
            embeddings = HuggingFaceEmbeddings(model_name="./model/sentence-transformers/all-MiniLM-L6-v2")
            self.faiss_critic_memory = FAISS.from_texts(self.critic_memory.keys(), embeddings)
        
        self.infos = {}
        self.final_infos = {}
        
        self.batch_size = args.batch_size
        self.enc = tiktoken.encoding_for_model("text-davinci-003")

    def get_reflect_str(self,
                strategy: ReflexionStrategy, idxs) -> None:
        mode = getattr(self.args, 'reflection_retrieval_mode', 'original')
        for id in idxs:
            self.reflection_retrieval_records[id] = []

        if mode == 'dynamic':
            for id in idxs:
                self.reflections_str[id] = ''
                self.episode_reflections_str[id] = ''
            return

        static_k = self._static_reflection_k()
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION:
            for id in idxs:
                self.reflections_str[id] = self.format_reflections(self.reflections, id, MAX=static_k)
                self.episode_reflections_str[id] = self.reflections_str[id]
                if mode != 'original':
                    self._mark_reflections_used(self._last_formatted_reflections.get(id, []))
                    self._record_episode_retrieval(id, static_k)
                print(self.reflections_str[id])
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
    
    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION, outfilename='') -> None:
        
        for i in range(0, len(self.idxs), self.batch_size):
            temp_idxs = self.idxs[i: i+self.batch_size]
            print(f'temp_idxs:{temp_idxs}')
            
            self.get_reflect_str(reflect_strategy, temp_idxs)
            
            self.single_run(temp_idxs, reset)

            self.reflect(reflect_strategy, temp_idxs)
            self._apply_reflection_memory_policy()
            self._update_reflections_lib()
            self._update_memory(temp_idxs)
        
            self._build_info(temp_idxs)
        
            self.final_infos['trajs'] = self.infos
            self.final_infos['reflections'] = self.reflections
            self.final_infos['actor_memory'] = self.actor_memory
            self.final_infos['critic_memory'] = self.critic_memory
            save_info(self.final_infos, outfilename)
            
        return self.final_infos

    def reflect(self,
                strategy: ReflexionStrategy, idxs) -> None:
        print('Reflecting...')

        if strategy == ReflexionStrategy.REFLEXION:
            new_reflections = self.prompt_reflection(idxs)
            if getattr(self.args, 'reflection_memory_policy', 'full') == 'full':
                self.reflections += new_reflections
            else:
                self._add_reflections(new_reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
    
    def step(self, idxs) -> None:
        self._refresh_step_reflections(idxs)

        # Think
        for id in idxs:
            self.scratchpad[id] += f'\nThought {self.step_n}:'
        prompts = self.prompt_agent(idxs)
        for i, id in enumerate(idxs):
            self.scratchpad[id] += ' ' + prompts[i]

        random_type = []
        q_prompt = {}
        if self.tool_use ==True:
        # print(self.scratchpad.split('\n')[-1])
            for i, id in enumerate(idxs):
                hist = self.env.get_hist_list(self.argument_lists[id])
                random_type.append(random.sample([x for x in self.GENRE if x not in hist], 2))
                self.scratchpad[id] += f'(Please recommend {random_type[i][0]} and {random_type[i][1]} items to help users explore their interests)'
            
                if self.faiss_actor_memory!=None:
                    q_prompt[id] = self._get_actor_memory(self.task.get_history_actions(id), self.argument_lists[id])
                    self.scratchpad[id] += q_prompt[id]
                    print(q_prompt[id])
                
        # Act
        for id in idxs:
            self.scratchpad[id] += f'\nAction {self.step_n}:'
        for _ in range(5):
            try:
                action = self.prompt_agent(idxs)
                action_types, arguments = parse_action(action)
                print(f'a:{action}')
                break
            except:
                print('b')
                continue
            
        for i, id in enumerate(idxs):
            if self.tool_use and i < len(random_type):
                self.scratchpad[id] = self.scratchpad[id].replace(f'(Please recommend {random_type[i][0]} and {random_type[i][1]} items to help users explore their interests)', '')
            if self.faiss_actor_memory!=None:
                self.scratchpad[id] = self.scratchpad[id].replace(q_prompt.get(id, ''), '')
        
             
        
        for i, id in enumerate(idxs):
            
            self.scratchpad[id] += ' ' + action[i]
            
            if action_types[i] == 'recommend':
                old_film = arguments[i]
                rerank_after_grounding = getattr(self.args, 'rerank_after_grounding', False)
                grounding_topk = getattr(self.args, 'grounding_topk', 5) if rerank_after_grounding else self.args.Max_Iteration
                argument_candidate = self.grounding_model.get_topk_near_item([old_film], grounding_topk)[0]
                arguments[i] = self._select_grounded_item(id, old_film, argument_candidate, q_prompt.get(id, ''))
                self.argument_lists[id].append(arguments[i])
                if old_film != arguments[i]:
                    self.scratchpad[id] += f'\nObservation {self.step_n}: [{old_film}] can not be recommened, instead, recommend[{arguments[i]}]'
                

        # Observe
        for i, id in enumerate(idxs):
            self.scratchpad[id] +=  f'\nObservation {self.step_n}: '
        
            if action_types[i] == 'recommend':
                reward = self.env.get_reward(self.userids[id], arguments[i])
                if self.env.whether_to_leave(self.userids[id], arguments[i], self.argument_lists[id]):
                    self.scratchpad[id] += f"Episode finished, User Stop, reward=-1000.000"
                    self.reward_lists[id].append(-1000)
                    self.finished[id] = True
                else:
                    self.scratchpad[id] += f"Episode continue, reward={reward}"
                    self.reward_lists[id].append(reward)

            else:
                self.scratchpad[id] += 'Invalid Action. Valid Actions are recommend[item].'
                self.finished[id] = True
        
        # update actor memory
        value = self.prompt_critic_llm(idxs)
        self._update_actor_memory(self.reward_lists, value, arguments, idxs)
        self._update_critic_memory(self.reward_lists, value, idxs)
            
        
        # print(self.scratchpad.split('\n')[-1])
        self.step_n += 1
        print(self.step_n)
    
    
    
    
    def _build_agent_prompt(self, idxs) -> str:
        prompts = [self.agent_prompt.format(
                            examples = self.react_examples,
                            trajs = '', 
                            reflections = self.reflections_str[id],
                            question = self.task[id],
                            scratchpad = truncate_scratchpad(self.scratchpad[id],tokenizer=self.enc)) for id in idxs]

        return prompts 
    
    def _build_critic_prompt(self, idxs) -> str:
        history_list = []
        instruction_list = []
        for i, id in enumerate(idxs):
            temp_list = (self.task.get_history_actions(id)+self.argument_lists[id])[-10:]
            history_list.append(temp_list)
            instruction_list.append(self._get_critic_memory(temp_list)) 
            
        prompts = [self.critic_prompt.format(
                            history_list = history_list[i],
                            instruction = instruction_list[i]) for i, id in enumerate(idxs)]
        return prompts

    def _refresh_step_reflections(self, idxs):
        mode = getattr(self.args, 'reflection_retrieval_mode', 'original')
        if mode in ('original', 'episode'):
            return

        dynamic_k = self._dynamic_reflection_k()
        for id in idxs:
            current_query = self._current_reflection_query(id)
            static_selected = (
                self._latest_retrieval_selection(id, 'episode')
                if mode == 'hybrid'
                else []
            )
            current_reflections = self._format_reflections_by_query(
                current_query,
                dynamic_k,
                record_id=id,
                scope='dynamic',
                exclude_reflections=static_selected,
            )

            if mode == 'dynamic':
                self.reflections_str[id] = current_reflections
            elif mode == 'hybrid':
                dynamic_selected = self._latest_retrieval_selection(id, 'dynamic')
                self.reflections_str[id] = self._format_selected_reflections(
                    static_selected + dynamic_selected
                )
            else:
                raise NotImplementedError(f'Unknown reflection retrieval mode: {mode}')

    def _initial_reflection_query(self, id):
        return self.task[id]

    def _current_reflection_query(self, id):
        if not self.argument_lists[id]:
            return self._initial_reflection_query(id)
        current_items = self.task.get_history_actions(id) + self.argument_lists[id]
        return format_reflection_query(current_items, Max=self._reflection_query_window())

    def _reflection_query_window(self):
        return max(1, getattr(self.args, 'reflection_query_window', 15))

    def _static_reflection_k(self):
        k = getattr(self.args, 'static_reflection_k', 0)
        return k if k > 0 else self.args.Max_Reflections

    def _dynamic_reflection_k(self):
        k = getattr(self.args, 'dynamic_reflection_k', 0)
        return k if k > 0 else self.args.Max_Reflections

    def _record_episode_retrieval(self, id, requested_k):
        self.reflection_retrieval_records[id].append({
            'scope': 'episode',
            'step': 0,
            'query': self._initial_reflection_query(id),
            'requested_k': requested_k,
            'memory_pool_size': len(self.reflections),
            'excluded_reflections': [],
            'selected_reflections': list(self._last_formatted_reflections.get(id, [])),
        })

    def _format_reflections_by_query(
        self,
        query,
        MAX,
        header=REFLECTION_HEADER,
        label='Reflections',
        record_id=None,
        scope=None,
        exclude_reflections=None,
    ):
        excluded_reflections = list(exclude_reflections or [])
        excluded = set(excluded_reflections)
        candidate_k = min(len(self.reflections), MAX + len(excluded))
        candidates = []
        if self.reflections and MAX > 0 and len(self.reflections) <= candidate_k:
            candidates = [r.strip() for r in self.reflections]
        elif self.reflections and MAX > 0 and self.faiss_reflections is None:
            candidates = [r.strip() for r in self.reflections[-candidate_k:]]
        elif self.reflections and MAX > 0:
            docs = try_with_delay(self.faiss_reflections, query, candidate_k)
            candidates = [r.page_content.strip() for r, score in docs]

        selected_reflections = []
        for reflection in candidates:
            if reflection in excluded or reflection in selected_reflections:
                continue
            selected_reflections.append(reflection)
            if len(selected_reflections) == MAX:
                break

        self._mark_reflections_used(selected_reflections)
        if record_id is not None:
            self.reflection_retrieval_records[record_id].append({
                'scope': scope,
                'step': 0 if scope == 'episode' else getattr(self, 'step_n', 0),
                'query': query,
                'requested_k': MAX,
                'memory_pool_size': len(self.reflections),
                'excluded_reflections': excluded_reflections,
                'selected_reflections': selected_reflections,
            })

        return self._format_selected_reflections(selected_reflections, header, label)

    def _latest_retrieval_selection(self, id, scope):
        for record in reversed(self.reflection_retrieval_records.get(id, [])):
            if record['scope'] == scope:
                return record['selected_reflections']
        return []

    def _format_selected_reflections(
        self,
        selected_reflections,
        header=REFLECTION_HEADER,
        label='Reflections',
    ):
        if not selected_reflections:
            return ''
        return header + f'{label}:\n- ' + '\n- '.join(selected_reflections)

    def _experiment_config(self):
        return {
            'run_name': getattr(self.args, 'run_name', ''),
            'reflection_retrieval_mode': getattr(self.args, 'reflection_retrieval_mode', 'original'),
            'reflection_query_style': 'original_task_question',
            'reflection_query_window': self._reflection_query_window(),
            'static_reflection_k': self._static_reflection_k(),
            'dynamic_reflection_k': self._dynamic_reflection_k(),
            'Max_Reflections': self.args.Max_Reflections,
            'Max_Iteration': self.args.Max_Iteration,
            'reflection_memory_policy': getattr(self.args, 'reflection_memory_policy', 'full'),
            'reflection_memory_size': getattr(self.args, 'reflection_memory_size', 0),
            'rerank_after_grounding': getattr(self.args, 'rerank_after_grounding', False),
            'grounding_topk': getattr(self.args, 'grounding_topk', 5),
        }

    def _experiment_enabled(self):
        return (
            bool(getattr(self.args, 'run_name', ''))
            or getattr(self.args, 'reflection_retrieval_mode', 'original') != 'original'
            or getattr(self.args, 'reflection_memory_policy', 'full') != 'full'
            or getattr(self.args, 'reflection_memory_size', 0) > 0
            or getattr(self.args, 'rerank_after_grounding', False)
        )

    def _initialize_reflection_usage(self):
        self._reflection_usage = {}
        self._reflection_clock = 0
        for reflection in self.reflections:
            self._reflection_clock += 1
            self._reflection_usage[reflection] = self._reflection_clock

    def _add_reflections(self, new_reflections):
        for reflection in new_reflections:
            if reflection is None:
                continue
            reflection = str(reflection).strip()
            if not reflection:
                continue
            self.reflections.append(reflection)
            self._mark_reflections_used([reflection])

    def _mark_reflections_used(self, reflections):
        for reflection in reflections:
            self._reflection_clock += 1
            self._reflection_usage[reflection] = self._reflection_clock

    def _apply_reflection_memory_policy(self):
        policy = getattr(self.args, 'reflection_memory_policy', 'full')
        memory_size = getattr(self.args, 'reflection_memory_size', 0)

        if policy == 'full' or memory_size <= 0 or len(self.reflections) <= memory_size:
            self._sync_reflection_usage()
            return

        if policy == 'fifo':
            self.reflections = self.reflections[-memory_size:]
            
        elif policy == 'lru':
            ranked = sorted(
                enumerate(self.reflections),
                key=lambda item: (self._reflection_usage.get(item[1], -1), item[0]),
                reverse=True,
            )
            keep_indices = set(index for index, reflection in ranked[:memory_size])
            self.reflections = [reflection for index, reflection in enumerate(self.reflections) if index in keep_indices]
        else:
            raise NotImplementedError(f'Unknown reflection memory policy: {policy}')

        self._sync_reflection_usage()

    def _sync_reflection_usage(self):
        current_reflections = set(self.reflections)
        self._reflection_usage = {
            reflection: used_at
            for reflection, used_at in self._reflection_usage.items()
            if reflection in current_reflections
        }
        for reflection in self.reflections:
            if reflection not in self._reflection_usage:
                self._reflection_clock += 1
                self._reflection_usage[reflection] = self._reflection_clock

    def _select_grounded_item(self, id, proposed_item, candidates, actor_memory_prompt=''):
        if not candidates:
            return proposed_item
        if not getattr(self.args, 'rerank_after_grounding', False) or len(candidates) == 1:
            return candidates[0]

        prompt = self._build_grounding_rerank_prompt(id, proposed_item, candidates, actor_memory_prompt)
        for _ in range(3):
            try:
                response = format_step(self.llm([prompt]))[0]
                action_type, argument = parse_action([response])
                if action_type[0] == 'recommend':
                    selected = self._match_grounded_candidate(argument[0], candidates)
                    if selected is not None:
                        self._record_grounding_rerank(id, proposed_item, candidates, selected, False)
                        return selected
            except Exception as e:
                print(f'grounding rerank failed: {e}')

        self._record_grounding_rerank(id, proposed_item, candidates, candidates[0], True)
        return candidates[0]

    def _record_grounding_rerank(self, id, proposed_item, candidates, selected_item, fallback):
        self.grounding_rerank_records[id].append({
            'step': self.step_n,
            'proposed_item': proposed_item,
            'candidates': list(candidates),
            'selected_item': selected_item,
            'fallback_to_first': fallback,
        })

    def _build_grounding_rerank_prompt(self, id, proposed_item, candidates, actor_memory_prompt=''):
        candidate_lines = '\n'.join([f'{i + 1}. {candidate}' for i, candidate in enumerate(candidates)])
        history = self.task.get_history_actions(id)
        current_traj = truncate_scratchpad(self.scratchpad[id], n_tokens=2500, tokenizer=self.enc)
        reflections = self.reflections_str.get(id, '')
        memory = actor_memory_prompt if actor_memory_prompt else 'None'

        return f"""You are choosing the final recommendation after grounding an actor action to real catalog items.
The original actor proposed: [{proposed_item}]

User viewing history:
{history}

Current reasoning trajectory:
{current_traj}

Relevant reflections:
{reflections if reflections else 'None'}

Relevant actor memory:
{memory}

Grounded candidate items:
{candidate_lines}

Choose exactly one item from Grounded candidate items. Your response must be exactly one line in this format:
recommend[exact candidate item]
"""

    def _match_grounded_candidate(self, selected, candidates):
        if selected in candidates:
            return selected

        normalized_selected = normalize_answer(selected)
        for candidate in candidates:
            if normalize_answer(candidate) == normalized_selected:
                return candidate
        return None
    
    def _build_info(self, idxs) -> str:
        for id in idxs:
            userid = self.userids[id]
            self.infos[id] = {}
            prompt = self.agent_prompt.format(
                                examples = self.react_examples,
                                reflections = '',
                                trajs = '',
                                question = '',
                                scratchpad = '')
            traj = self.task[id] + self.scratchpad[id]
            # reflection = format_reflections(self.reflections[id], MAX=1000)
            self.infos[id].update({'userid': userid, 'prompt': prompt, 'traj': traj, 'traj_by_line': traj.split('\n')})
            if self._experiment_enabled():
                self.infos[id].update({
                    'experiment_config': self._experiment_config(),
                    'reflection_retrievals': list(self.reflection_retrieval_records.get(id, [])),
                    'grounding_reranks': list(self.grounding_rerank_records.get(id, [])),
                })
        
    
    def _update_memory(self, idxs, alpha = 0.5):
        embeddings = HuggingFaceEmbeddings(model_name="./model/sentence-transformers/all-MiniLM-L6-v2")
        if self.actor_memory:
            self.faiss_actor_memory = FAISS.from_texts(list(self.actor_memory.keys()), embeddings)
        if self.critic_memory:
            self.faiss_critic_memory = FAISS.from_texts(list(self.critic_memory.keys()), embeddings)
    
    def _update_actor_memory(self, reward_lists, value, arguments, idxs, gamma=0.5):
        for i, id in enumerate(idxs):
            try:
                v_i = float(value[i])
            except:
                v_i = 0
            temp_list = self.task.get_history_actions(id)+self.argument_lists[id][:-1]
            query = format_query(temp_list)
            if query not in self.actor_memory:
                self.actor_memory[query] = {}
            if not reward_lists.get(id) or not self.value_lists.get(id):
                continue
            if reward_lists[id][-1] + gamma*v_i - self.value_lists[id][-1] >= 0:
                self.actor_memory[query][arguments[i]] = 1
            else:
                self.actor_memory[query][arguments[i]] = -1
            self.value_lists[id].append(float(value[i]))
    
    def _update_critic_memory(self, reward_lists, value, idxs, gamma=0.5):
        for i, id in enumerate(idxs):
            try:
                v_i = float(value[i])
            except (IndexError, ValueError):
                v_i = 0
            if not reward_lists.get(id):
                continue
            temp_list = self.task.get_history_actions(id)+self.argument_lists[id][:-1]
            query = format_query(temp_list)
            self.critic_memory[query] = reward_lists[id][-1] + gamma * v_i
    
    def _update_reflections_lib(self):
        if not self.reflections:
            self.faiss_reflections = None
            return
        embeddings = HuggingFaceEmbeddings(model_name="./model/sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_reflections = FAISS.from_texts(self.reflections, embeddings)
    
    def _get_actor_memory(self, history_list, argument_list, k=1):
        temp_list = history_list + argument_list
        query = format_query(temp_list)
       
        # keys = self.faiss_actor_memory.similarity_search_with_score(query, k=k)
        keys = try_with_delay(self.faiss_actor_memory, query, k)
        
        Q_values = [_ for key, score in keys for _ in self.actor_memory[key.page_content].values() if (score<0.5 and score>0.3) or score <0.01]
        actions = [_ for key, score in keys for _ in self.actor_memory[key.page_content].keys() if (score<0.5 and score>0.3) or score <0.01]
        
        pos_actions = [actions[i] for i in range(len(Q_values)) if Q_values[i]>0]
        neg_actions = [actions[i] for i in range(len(Q_values)) if Q_values[i]<0]
        if len(actions) == 0:
            return ''
        else:
            return f"According to historical experience, When {query}, we encourage to recommend {','.join(map(str, pos_actions))} items and not to recommend {','.join(map(str, neg_actions))} items."
    
    def _get_critic_memory(self, history_list, k=1):
        query = format_query(history_list)
        if self.faiss_critic_memory == None:
            return ''
        # keys = self.faiss_critic_memory.similarity_search_with_score(query, k=k)
        keys = try_with_delay(self.faiss_critic_memory, query, k)
        values = [self.critic_memory[key.page_content] for key, score in keys if score < 0.01]
        
        if len(values) == 0:
            return ''
        else:
            return f"According to historical experience, When {query}, the Value is {values[0]}"
        
        
    def format_reflections(self, reflections: List[str], id,
                        header: str = REFLECTION_HEADER, MAX=2) -> str:
        if reflections == [] or MAX == 0:
            self._last_formatted_reflections[id] = []
            return ''
        elif len(reflections) <= MAX:
            selected_reflections = [r.strip() for r in reflections]
        else:
            docs = try_with_delay(self.faiss_reflections, self.task[id], MAX)
            selected_reflections = [r.page_content.strip() for r, score in docs]

        self._last_formatted_reflections[id] = selected_reflections
        return header + 'Reflections:\n- ' + '\n- '.join(selected_reflections)
    
    def prompt_critic_llm(self, idxs):
        return format_step(self.critic_llm(self._build_critic_prompt(idxs)))
    
    
    

        
   
### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def try_with_delay(memory, query, k):
    while True:
        try:
            result = memory.similarity_search_with_score(query, k=k)
            break
        except openai.error.AuthenticationError as e:
            print(f'c:{e}')
            time.sleep(10)
    return result


def format_step(steps: list) -> list:
    return [step.strip('\n').strip().replace('\n', '') for step in steps]

def format_query(argument_list, Max=10):
    last_elements = argument_list[-Max:]  # 获取最后十个元素
    result = ','.join(map(str, last_elements)) 
    query = 'The user viewing history is [' + result + ']'
    return query

def format_reflection_query(argument_list, Max=15):
    return (
        f"The user's viewing history is {argument_list[-Max:]}, "
        "please recommend item for this user"
    )

def calculate_q_value(reward_list, gamma=0.5):
    q_value_list = [0] * len(reward_list)
    for i in range(len(reward_list)-1, -1, -1):
        if i == len(reward_list)-1:
            q_value_list[i] = reward_list[i]
        else:
            q_value_list[i] = reward_list[i] + gamma * q_value_list[i+1]
    return q_value_list
    
def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)
