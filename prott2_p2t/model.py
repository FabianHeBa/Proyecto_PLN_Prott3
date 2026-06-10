import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import Config
from .qformer import ProteinQFormer


class ProtT3GemmaQA(nn.Module):
    def __init__(self, protein_dim, config: Config):
        super().__init__()
        self.config = config


        self.tokenizer = AutoTokenizer.from_pretrained(config.gemma_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            compute_dtype = torch.float32

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )


        self.llm = AutoModelForCausalLM.from_pretrained(
            config.gemma_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

        self.llm.config.use_cache = False

        self.llm = prepare_model_for_kbit_training(self.llm)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )

        self.llm = get_peft_model(self.llm, lora_config)

        self.llm.config.use_cache = False

        if hasattr(self.llm, "generation_config"):
            self.llm.generation_config.use_cache = False

            if hasattr(self.llm.generation_config, "cache_implementation"):
                self.llm.generation_config.cache_implementation = None

        llm_hidden = self.llm.config.hidden_size

        self.protein_qformer = ProteinQFormer(
            qformer_id=config.qformer_id,
            protein_dim=protein_dim,
            llm_hidden_size=llm_hidden,
            num_query_tokens=config.n_query_tokens,
        )

        self.protein_qformer.to(self.llm_device())

    def llm_device(self):
        return self.llm.get_input_embeddings().weight.device

    def build_prompt(self, question):
        return (
            "<bos><start_of_turn>user\n"
            "You are given learned protein tokens extracted from an amino-acid sequence.\n"
            f"Question: {question}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def encode_text_batch(self, questions, answers):
        pad_id = self.tokenizer.pad_token_id

        input_ids_list = []
        labels_list = []

        for question, answer_text in zip(questions, answers):
            prompt = self.build_prompt(question)
            answer = str(answer_text) + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
            ).input_ids

            answer_ids = self.tokenizer(
                answer,
                add_special_tokens=False,
            ).input_ids

            if len(prompt_ids) + len(answer_ids) > self.config.max_text_len:
                remaining = max(1, self.config.max_text_len - len(prompt_ids))
                answer_ids = answer_ids[:remaining]

            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            input_ids_list.append(ids)
            labels_list.append(labels)

        max_len = max(len(x) for x in input_ids_list)

        input_ids = torch.full(
            (len(input_ids_list), max_len),
            pad_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (len(input_ids_list), max_len),
            dtype=torch.long,
        )

        labels = torch.full(
            (len(input_ids_list), max_len),
            -100,
            dtype=torch.long,
        )

        for i, ids in enumerate(input_ids_list):
            length = len(ids)

            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            labels[i, :length] = torch.tensor(labels_list[i], dtype=torch.long)

        return input_ids, attention_mask, labels

    def forward(self, protein_emb, protein_mask, questions, answers):
        device = self.llm_device()

        protein_emb = protein_emb.to(device)
        protein_mask = protein_mask.to(device)

        protein_tokens = self.protein_qformer(
            protein_emb.float(),
            protein_mask,
        )

        input_ids, text_mask, labels = self.encode_text_batch(
            questions,
            answers,
        )

        input_ids = input_ids.to(device)
        text_mask = text_mask.to(device)
        labels = labels.to(device)

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        protein_tokens = protein_tokens.to(
            device=device,
            dtype=text_embeds.dtype,
        )


        inputs_embeds = torch.cat(
            [protein_tokens, text_embeds],
            dim=1,
        )

        protein_token_mask = torch.ones(
            protein_tokens.shape[:2],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.cat(
            [protein_token_mask, text_mask],
            dim=1,
        )

        protein_labels = torch.full(
            protein_tokens.shape[:2],
            -100,
            dtype=torch.long,
            device=device,
        )

        labels = torch.cat(
            [protein_labels, labels],
            dim=1,
        )

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        return out

    @torch.no_grad()
    def generate(self, protein_emb, protein_mask, question, max_new_tokens=None):

        self.eval()

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        device = self.llm_device()

        protein_emb = protein_emb.to(device)
        protein_mask = protein_mask.to(device)

        # ---------------------------------------------------------
        # 1. Protein tokens
        # ---------------------------------------------------------
        protein_tokens = self.protein_qformer(
            protein_emb.float(),
            protein_mask,
        )

        prompt = self.build_prompt(question)

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        protein_tokens = protein_tokens.to(
            device=device,
            dtype=text_embeds.dtype,
        )

        inputs_embeds = torch.cat(
            [protein_tokens, text_embeds],
            dim=1,
        )

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=device,
        )

        generated_ids = []

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        if pad_id is None:
            pad_id = eos_id

        for _ in range(max_new_tokens):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            token_id_int = next_token_id.item()

            if token_id_int == eos_id:
                break

            generated_ids.append(token_id_int)

            next_token_embeds = self.llm.get_input_embeddings()(next_token_id)

            inputs_embeds = torch.cat(
                [inputs_embeds, next_token_embeds],
                dim=1,
            )

            next_attention = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=torch.long,
                device=device,
            )

            attention_mask = torch.cat(
                [attention_mask, next_attention],
                dim=1,
            )

        if len(generated_ids) == 0:
            return ""

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
