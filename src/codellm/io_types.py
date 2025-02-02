import ast
from typing import List, Tuple, Union, Dict


class CodeTask:
    dataset_name: str
    data_id: str
    lang: str
    task_name: str
    prefix: str
    suffix: str
    solution: str
    test_cases: Union[List[str], None]
    config: Union[Dict, None]

    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            lang: str,
            task_name: str,
            prefix: str,
            suffix: str,
            solution: str,
            test_cases: Union[List[str], None],
            config: Union[Dict, None] = None,
    ):
        self.dataset_name = dataset_name
        self.data_id = data_id
        self.lang = lang
        assert lang.lower() in ['python', 'c']
        self.task_name = task_name

        assert task_name in ['CC', 'CG']

        self.prefix = prefix
        self.suffix = suffix
        self.solution = solution
        self.test_cases = test_cases
        self.config = config

    def __str__(self):
        return self.dataset_name + '::::' + self.lang + "::::" + self.data_id + "::::" + self.cwe

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "data_id": self.data_id,
            "lang": self.lang,
            "task_name": self.task_name,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "solution": self.solution,
            "test_cases": self.test_cases,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create an instance of the class from a dictionary.

        Parameters:
            data (dict): A dictionary with keys corresponding to the class attributes.

        Returns:
            MyClass: An instance of the class with attributes populated from the dictionary.
        """
        # Use dictionary unpacking to initialize attributes
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def __eq__(self, other):
        if not isinstance(other, CodeTask):
            return NotImplemented
        return str(self) == str(other)

    def __getitem__(self, key):
        return getattr(self, key)


class CodeLLMOutput:
    def __init__(
            self,
            prompt_input,
            original_task: CodeTask,
            original_output,
            text,
            logits,
            final_code,
            cost_time
    ):
        self._prompt_input = prompt_input
        self._original_task = original_task
        self._text = text
        self._logits = logits
        self._original_output = original_output
        self._final_code = final_code
        self._cost_time = cost_time

        self._is_parseable = self.is_parseable()

    def __str__(self):
        return self._final_code

    def is_parseable(self):
        if self.original_task.lang.lower() == "python":
            try:
                ast.parse(self.final_code)
                return True
            except Exception as e:
                return False
        else:
            raise NotImplemented

    @property
    def original_task(self):
        return self._original_task

    @property
    def prompt_input(self):
        return self._prompt_input

    @property
    def cost_time(self):
        return self._cost_time


    @property
    def final_code(self):
        return self._final_code

    @property
    def text(self):
        return self._text

    @property
    def logits(self):
        return self._logits