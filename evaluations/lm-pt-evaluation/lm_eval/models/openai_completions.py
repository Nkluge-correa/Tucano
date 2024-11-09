import copy
import os
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple
import time

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import retry_on_specific_exceptions, eval_logger


def get_result(response, ctxlen: int) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response.logprobs.token_logprobs
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response.logprobs.token_logprobs)):
        token = response.logprobs.token_logprobs[i]
        top_tokens = response.logprobs.top_logprobs[i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


def oa_completion(client, chat: bool = False, **kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    if not find_spec("openai") or not find_spec("tiktoken"):
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
            "Please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`"
        )
    else:
        import openai

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        import traceback

        traceback.print_exc()

    @retry_on_specific_exceptions(
        on_exceptions=[openai.OpenAIError],
        max_retries=5,  
        backoff_time = 3.0,
        backoff_multiplier = 2, 
        on_exception_callback=_exception_callback,
    )
    def completion():
        if chat:
            return client.chat.completions.create(**kwargs)
        else:
            return client.completions.create(**kwargs)

    try:
        resp = completion()
        response = [c.message.content for c in resp.choices]
        eval_logger.info(f"Response: {response}")
        
    except Exception as e:
        eval_logger.info(f"Exception raised: {str(e)}")
        eval_logger.info("Given an empty response")
        response = [""]

    return response


@register_model("openai-completions", "local-completions")
class OpenaiCompletionsLM(LM):
    REQ_CHUNK_SIZE = 20
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model: str,
        base_url: str = None,
        tokenizer: Optional[str] = None,
        tokenizer_backend: Literal["tiktoken", "huggingface"] = "tiktoken",
        truncate: bool = False,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        seed: int = 1234,
        max_length: Optional[int] = None,
    ) -> None:
        """

        :param engine: str
            OpenAI API engine (e.g. gpt-3.5-turbo-instruct)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        self.seed = seed
        try:
            import openai  # noqa: E401
            import tiktoken
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .\"[openai]\"`",
            )
        self.model = model
        self.base_url = base_url
        self.tokenizer_backend = tokenizer_backend
        self.truncate = truncate
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length

        # if we have a local model, use HF tokenizer over tiktoken
        if self.tokenizer_backend == "huggingface":
            import transformers  # noqa: E401

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else self.model
            )
            self.vocab_size = self.tokenizer.vocab
            self.end_of_text_token_id = self.tokenizer.eos_token
        elif self.tokenizer_backend == "tiktoken":
            if self.base_url:
                eval_logger.warning(
                    f"Passed `base_url={self.base_url}` but using Tiktoken tokenizer backend. "
                    "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                )

            self.tokenizer = tiktoken.encoding_for_model(self.model)
            self.vocab_size = self.tokenizer.n_vocab
            self.end_of_text_token_id = self.tokenizer.eot_token
        else:
            raise ValueError(
                f"Expected tokenizer_backend to be one of ['tiktoken', 'huggingface'] but got {self.tokenizer_backend}"
            )

        # Read from environment variable OPENAI_API_KEY
        # Set to EMPTY for local
        openai.api_key = os.environ["OPENAI_API_KEY"]
        if self.base_url:
            self.client = openai.OpenAI(base_url=self.base_url)
        else:
            self.client = openai.OpenAI()

    @property
    def eot_token_id(self):
        return self.end_of_text_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        else:
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about, and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), self.batch_size)),
            disable=disable_tqdm,
        ):
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
                )

                inps.append(inp)
                ctxlens.append(ctxlen)

            response = oa_completion(
                client=self.client,
                model=self.model,
                prompt=inps,
                echo=True,
                max_tokens=0,
                temperature=0.0,
                logprobs=10,
                seed=self.seed,
            )

            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                response.choices, ctxlens, chunk
            ):
                answer = get_result(resp, ctxlen)

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return re_ord.get_original(res)

    def generate_until(self, requests) -> List[str]:
        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, request_args in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.batch_size))
        ):
            inps = []
            self._max_gen_toks = request_args.get("max_gen_toks", self.max_gen_toks)
            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)

            until = request_args.get("until", ["<|endoftext|>"])
            request_args["temperature"] = request_args.get("temperature", 0)

            response = oa_completion(
                client=self.client,
                model=self.model,
                prompt=inps,
                max_tokens=self.max_gen_toks,
                stop=until,
                seed=self.seed,
                **{
                    k: v
                    for k, v in request_args.items()
                    if k not in ["do_sample", "max_gen_toks"]
                },
            )
            for resp, (context, args_) in zip(response.choices, chunk):
                s = getattr(resp, "text")

                until_ = until

                for term in until_:
                    if len(term) > 0:
                        s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial(
                    "generate_until", (context, {"until": until_}), s
                )

                res.append(s)
        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests]):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

#import ftfy
import re

def remove_surrogates(text):
    cleaned_text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    return cleaned_text.strip()

@register_model("openai-chat-completions", "local-chat-completions")
class OpenaiChatCompletionsLM(LM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",  # GPT model or Local model using HuggingFace model paths
        base_url: str = None,
        truncate: bool = False,
        sleep_after_request: float = None,
        **kwargs,
    ) -> None:
        """

        :param model: str
            Implements an OpenAI-style chat completion API for
            accessing both OpenAI OR locally-hosted models using
            HuggingFace Tokenizer
            OpenAI API model (e.g. gpt-3.5-turbo)
            using the **gen_kwargs passed on init
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = base_url
        self.truncate = truncate
        self.sleep_after_request = sleep_after_request

        # Read from environment variable OPENAI_API_KEY
        # Set to EMPTY for local
        if self.base_url:
            self.client = openai.OpenAI(base_url=self.base_url, max_retries=0)
        else:
            self.client = openai.OpenAI()  # openai.AsyncOpenAI()

        self.fix_text = lambda x: x.strip()
        if "gemini" in self.model:
            self.fix_text = remove_surrogates

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        res = defaultdict(list)
        re_ords = {}

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer(
                [(req, req.args[1]) for req in reqs], lambda x: (-len(x[0].args[0]), x[0].args[0])
            )

        resp_exist = {}
        
        """
        if "gemini-1.5" in self.model:
            import json
            model_path = "/mnt/e/ceia/llm-eval/portuguese-llm-eval/outputs/gemini_1_5"
            for task in ["bluex", "enem_challenge", "oab_exams"]:
                resp_exist[task] = {}
                task_dir = os.path.join(model_path, task + '_limit200')
                filename = [f for f in os.listdir(task_dir) if f.endswith(".jsonl")][0]
                responses_path = os.path.join(task_dir, filename)
                with open(responses_path, 'r') as f:
                    resps = json.load(f)
                for resp in resps:
                    resp_exist[task][resp['doc']['id']] = resp['resps'][0][0]
        if "sabia" in self.model:
            import json
            log_path = "/mnt/e/ceia/llm-eval/portuguese-llm-eval/outputs/sabia-2_medium/bak/output.log"
            id_list = []
            task_dir = "/mnt/e/ceia/llm-eval/portuguese-llm-eval/outputs/gemini_1_5/enem_challenge/"
            filename = [f for f in os.listdir(task_dir) if f.endswith(".jsonl")][0]
            with open(os.path.join(task_dir, filename), 'r') as f:
                enem_challenge = json.load(f)
                for item in enem_challenge:
                    id_list.append(item['doc']['id'])
            resp_exist["enem_challenge"] = {}
            start_responses = False
            resp_index = 0
            with open(log_path, 'r') as f:
                for line in f:
                    if "[__main__.py:236] Selected Tasks:" in line:
                        resp_index = 0
                        start_responses = False
                        if "enem_challenge" in line:
                            start_responses = True
                    if start_responses:
                        if "[openai_completions.py:78] Response: " in line:
                            #The answer is in the format:
                            #2024-04-13:08:57:24,706 INFO     [openai_completions.py:78] Response: ['ANSWER']
                            resp_exist["enem_challenge"][id_list[resp_index]] = line.split("Response: ")[1].replace("['", "").replace("']", "")
                            resp_index += 1
                        elif "Given an empty response" in line:
                            resp_index += 1
        """
        
        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        for key, re_ord in re_ords.items():
            # n needs to be 1 because messages in
            # chat completion are not batch but
            # is regarded as a single conversation.
            chunks = utils.chunks(re_ord.get_reordered(), n=1)
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                inps = []
                metas = []    
                for context in contexts:
                    data = context.ctx_data
                    inps.append({"role": "system", "content": self.fix_text(data['description'])})
                    for shot, ans in data['fewshots']:
                        inps.append({"role": "user", "content": self.fix_text(shot)})
                        inps.append({"role": "assistant", "content": self.fix_text(ans)})
                    inps.append({"role": "user", "content": self.fix_text(data['example'])})
                    
                    if context.task_name in resp_exist and context.doc['id'] in resp_exist[context.task_name].keys():
                        metas.append(resp_exist[context.task_name][context.doc['id']])
                    else:
                        metas.append(None)
                    

                #inps = [{"role": "user", "content": context} for context in contexts]

                gen_kwargs = all_gen_kwargs[0]
                until = None
                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    if "do_sample" in kwargs.keys():
                        kwargs.pop("do_sample")
                    if "temperature" in kwargs.keys():
                        kwargs.pop("temperature")
                    if "top_k" in kwargs.keys():
                        kwargs.pop("top_k")
                    if "top_p" in kwargs.keys():
                        kwargs.pop("top_p")
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [kwargs]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                            )
                        kwargs["stop"] = until
                    if "claude" in self.model:
                        if "stop" in kwargs.keys():
                            kwargs.pop("stop")
                    kwargs["max_tokens"] = kwargs.pop("max_gen_toks", self.max_gen_toks)
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )

                skip_sleep = False
                if metas[0] is None:
                    #eval_logger.info(inps)
                    response = oa_completion(
                        client=self.client,
                        chat=True,
                        messages=inps,
                        model=self.model,
                        **kwargs,
                    )
                else:
                    print('Found answer in cache', metas)
                    response = metas
                    skip_sleep = True

                for resp, (context, args_) in zip(response, chunk):
                    s = resp

                    if until is not None:
                        for term in until:
                            if len(term) > 0:
                                s = s.split(term)[0]

                    res[key].append(s)

                    self.cache_hook.add_partial(
                        "generate_until", (context, {"until": until}), s
                    )
                    pbar.update(1)
                
                if self.sleep_after_request is not None and not skip_sleep:
                    time.sleep(self.sleep_after_request)

            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")