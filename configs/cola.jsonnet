// --------------------------------------------------------------------------------
// Language settings
// --------------------------------------------------------------------------------
local model_path = std.extVar("MODEL");

local tokenizer = { pretrained_model_name_or_path: model_path };
local model = {
    type: "loreiba.eval.cola.model::cola_model",
    tokenizer: tokenizer,
    model_path: model_path,
};

// our settings
local batch_size = 32;
local num_epochs = 10;

// --------------------------------------------------------------------------------
// Optimizer settings
// --------------------------------------------------------------------------------
local training_engine = {
    type: "torch",
    optimizer: {
        type: "torch::AdamW",
        lr: 1e-5,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: 0.01
    },
    //lr_scheduler: {
    //    type: "transformers::cosine",
    //    num_warmup_steps: validate_every,
    //    num_training_steps: num_steps,
    //},
    amp: false
};
local collate_fn = {
    type: "loreiba.eval.cola.data::collator",
    tokenizer: tokenizer,
};
local train_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: collate_fn,
};
local val_dataloader = {
    shuffle: false,
    batch_size: batch_size,
    collate_fn: collate_fn,
};

{
    steps: {
        inputs: {
            type: "loreiba.eval.cola.data::read",
            tokenizer: tokenizer,
        },
        trained_model: {
            type: "torch::train",
            model: model,
            dataset_dict: { type: "ref", ref: "inputs" },
            training_engine: training_engine,
            log_every: 1,
            train_dataloader: train_dataloader,
            train_epochs: num_epochs,
            validate_every: 2679,
            checkpoint_every: 2679,
            validation_split: "dev",
            validation_dataloader: val_dataloader,
            // val_metric_name: "perplexity",
            // minimize_val_metric: true,
        },
        final_metrics: {
            type: "torch::eval",
            model: { type: "ref", ref: "trained_model" },
            dataset_dict: { type: "ref", ref: "inputs" },
            dataloader: val_dataloader,
            metric_names: ["loss", "accuracy", "mcc"],
            test_split: "test",
        },
    }
}
