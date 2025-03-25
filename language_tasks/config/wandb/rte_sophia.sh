project: squeezebert-rte  # here
program: finetune.py  # here
method: grid
metric:
  name: accuracy.accuracy
  goal: maximize
parameters:
  optimizer:
    values: ['sophiah']  # here
  task_name:
    values: ['rte']  # here
  learning_rate:
    values: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
  weight_decay:
    values: [0, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
  update_each:
    values: [1, 2, 3, 4, 5, 10]
  clip_threshold:
    values: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
  eps:
    values: [1e-4, 1e-6, 1e-8]
  model_name_or_path:
    values: ['squeezebert/squeezebert-uncased']
  max_length:
    values: [512]
  num_train_epochs:
    values: [10]
  lr_scheduler_type:
    values: ['polynomial']
  per_device_train_batch_size:
    values: [16]
  