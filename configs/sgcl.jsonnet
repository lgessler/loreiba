// --------------------------------------------------------------------------------
// Language settings
// --------------------------------------------------------------------------------
local language = "wolof";
local language_code_index = {
    "coptic": "cop",
    "greek": "grc",
    "indonesian": "id",
    "maltese": "mt",
    "tamil": "ta",
    "uyghur": "ug",
    "wolof": "wo",
};

// a helper
local stringifyPair(k,v) = std.toString(k) + "-" + std.toString(v);
local stringifyObject(o) = std.join('_', std.objectValues(std.mapWithKey(stringifyPair, o)));

// --------------------------------------------------------------------------------
// Model settings
// --------------------------------------------------------------------------------
local max_length = 512;

// For pretrained
// local FROM_PRETRAINED = true;
// local model_path = "distilbert-base-cased";
// local tokenizer = { pretrained_model_name_or_path: model_path };
// local model = {
//     type: "loreiba.sgcl.model::sgcl_model",
//     pretrained_model_name_or_path: model_path,
// };

// For non-pretrained
local FROM_PRETRAINED = false;
local roberta_config = {
    hidden_size: 128,
    num_layers: 3,
    num_attention_heads: 8,
    intermediate_size: 512,
};
local model_path = "./workspace/models/roberta_" + stringifyObject(roberta_config);
local tokenizer = { pretrained_model_name_or_path: model_path };
local model = {
    type: "loreiba.sgcl.model::sgcl_model",
    roberta_config: roberta_config,
};


// --------------------------------------------------------------------------------
// Trainer settings
// --------------------------------------------------------------------------------
local training_steps = 4000;   # total number of optimization steps to train for
local validate_every = 400;    # how often to validate and save checkpoints
local batch_size = 64;
local amp = false;  # use PyTorch's native automatic mixed precision

// --------------------------------------------------------------------------------
// Optimizer settings
// --------------------------------------------------------------------------------
local warmup_steps = 400;
local learning_rate = 3e-5;  # you can probably use a higher LR for a small model like "gpt2"

local training_engine = {
    type: "torch",
    optimizer: {
        type: "torch::AdamW",
        lr: learning_rate,
        betas: [0.9, 0.95],
        eps: 1e-5,
    },
    lr_scheduler: {
        type: "transformers::linear",
        num_warmup_steps: warmup_steps,
        num_training_steps: training_steps,
    },
    amp: amp
};

local collate_fn = {
    type: "loreiba.sgcl.collator::collator",
    tokenizer: tokenizer
};
local train_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: collate_fn,
};
local val_dataloader = {
    shuffle: false,
    batch_size: batch_size,
    collate_fn: collate_fn
};

{
    steps: {
        raw_treebank_data: {
            type: "loreiba.data::read_ud_treebank",
            shortcut: language,
            tag: "r2.11"  // Use UD treebanks from release 2.11
        },
        bare_text_data: {
            type: "loreiba.data::read_text_only_conllu",
            shortcut: language,
            stanza_retokenize: true,
            stanza_language_code: language_code_index[language],
        },
        [if FROM_PRETRAINED then null else "tokenizer"]: {
            type: "loreiba.data::train_tokenizer",
            dataset: { "type": "ref", "ref": "bare_text_data" },
            model_path: model_path
        },
        tokenized_text_data: {
            type: "loreiba.data::tokenize_plus",
            dataset: { "type": "ref", "ref": "bare_text_data" },
            max_length: max_length,
            tokenizer: tokenizer,
            step_extra_dependencies: if FROM_PRETRAINED then [] else [ {type: "ref", "ref": "tokenizer" } ]
        },
        parsed_text_data: {
            type: "loreiba.data::stanza_parse_dataset",
            dataset: { "type": "ref", "ref": "tokenized_text_data" },
            language_code: language_code_index[language],
            allow_retokenization: false,  // we tokenized earlier
        },
        model_inputs: {
            type: "loreiba.data::finalize",
            dataset: { "type": "ref", "ref": "parsed_text_data" },
        },
        trained_model: {
            type: "torch::train",
            model: model,
            dataset_dict: { type: "ref", ref: "model_inputs" },
            training_engine: training_engine,
            log_every: 1,
            train_dataloader: train_dataloader,
            train_steps: training_steps,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            validation_split: "dev",
            validation_dataloader: val_dataloader,
            val_metric_name: "loss",
            minimize_val_metric: true,
        },
        //final_metrics: {
        //    type: "torch::eval",
        //    model: { type: "ref", ref: "trained_model" },
        //    dataset_dict: { type: "ref", ref: "stype_instances" },
        //    dataloader: val_dataloader,
        //    metric_names: ["loss", "accuracy"],
        //    test_split: "test",
        //},
    }
}
