config = {}

training_opt = {}
training_opt['dataset'] = 'cifar10'
training_opt['num_classes'] = 10
training_opt['batch_size'] = 128
training_opt['num_workers'] = 4
training_opt['num_epochs'] = 200
training_opt['start_epoch'] = 0
training_opt['print_freq'] = 10
training_opt['accumulation_steps'] = 1
training_opt['log_root'] = 'results'
training_opt['data_root'] = '../data-local'
training_opt['imb_type'] = 'exp'
training_opt['gpu'] = 0
config['training_opt'] = training_opt

networks = {}
networks['arch'] = 'resnet32'
networks['param'] = {'num_classes': training_opt['num_classes'],
                     'use_norm': False,
                     'use_block': True}
config['networks'] = networks

optimizer = {}
optimizer['type'] = 'SGD'
optimizer['optim_params'] = {'lr': 0.1,
                             'momentum': 0.9,
                             'weight_decay': 2e-4,
                             'nesterov': True}
optimizer['scheduler'] = 'step'
optimizer['scheduler_params'] = {'step': [160, 180],
                          'gamma': 0.1,
                          'warmup_epoch': 5}
config['optimizer'] = optimizer