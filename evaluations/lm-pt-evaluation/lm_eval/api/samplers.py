class ContextSampler:
    def __init__(
            self,
            docs_dataset,
            task,
            fewshot_indices=None,
            rnd=None,
            exclude_from_task=False,
            id_column = None
        ) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs_dataset = docs_dataset  # HF dataset split, provided by task._fewshot_docs()
        self.fewshot_indices = fewshot_indices
        if fewshot_indices:  # subset few-shot docs from
            self.docs_dataset = self.docs_dataset.select(fewshot_indices)

        self.id_column = id_column
        if self.id_column not in self.docs_dataset.column_names:
            if isinstance(self.id_column, int):
                self.id_column = self.docs_dataset.column_names[self.id_column]

        self.exclude_from_task = exclude_from_task
        self.docs = list(self.docs_dataset)
    
    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        #n_samples = (
        #    num_fewshot + 1
        #    if self.config.fewshot_split == self.config.test_split
        #    else num_fewshot
        #)
        n_samples = num_fewshot

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        shots = [
            ((
                self.doc_to_text(doc)
                if (
                    self.config.doc_to_choice is None
                    or isinstance(self.doc_to_text(doc), str)
                )
                else self.doc_to_choice(doc)[self.doc_to_text(doc)]
            ),(
                str(self.doc_to_target(doc)[0])
                if isinstance(self.doc_to_target(doc), list)
                else self.doc_to_target(doc)
                if (
                    self.config.doc_to_choice is None
                    or isinstance(self.doc_to_target(doc), str)
                )
                else str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
            ))
            for doc in selected_docs
        ]

        labeled_examples = (
            self.fewshot_delimiter.join(
                # TODO: is separating doc_to_text and doc_to_target by one space always desired?
                [shot[0] + self.target_delimiter + str(shot[1]) for shot in shots]
            )
            + self.fewshot_delimiter
        )

        return labeled_examples, shots

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.rnd.sample(self.docs, n)
    
    def remove_fewshot_from_task(self, dataset):
        if self.id_column in dataset.column_names:
            fewshot_ids = self.docs_dataset[self.id_column]
            return dataset.filter(lambda x: x[self.id_column] not in fewshot_ids)
        elif self.fewshot_indices is not None:
            to_select = [i for i in range(len(dataset)) if i not in self.fewshot_indices]
            return dataset.select(to_select)
        else:
            raise ValueError("Cannot remove fewshot from task without fewshot_indices or id_column")

class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert (
            n <= len(self.docs)
        ), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass

class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass

class DocIDSampler(FirstNSampler):
    def __init__(
            self,
            id_list,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.id_list = id_list
        assert self.id_column, "Must specify id_column for DocIDSampler"
        self.docs_dataset = self.docs_dataset.filter(lambda x: x[self.id_column] in self.id_list)
        
        # reorder the dataset to match the order of the id_list
        id_column_list = list(self.docs_dataset[self.id_column])
        self.docs_dataset = self.docs_dataset.select([id_column_list.index(i) for i in self.id_list])
        
        self.docs = list(self.docs_dataset)

    def remove_fewshot_from_task(self, dataset):
        return dataset.filter(lambda x: x[self.id_column] not in self.id_list)
        
SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
    "id_sampler": DocIDSampler
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        )
