#!/usr/bin/python3
import os
import itertools
import json
import pandas as pd
import numpy as np
from keras.backend import clear_session
from keras.layers import Input, Dense, Flatten, Normalization, Rescaling, Conv2D, MaxPooling2D, RandomRotation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import image_dataset_from_directory
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import time
import plotly.express as px


def flatten(x, sep='.'):
    """
    Flatten a json object into a dictionary with keys that indicate the nesting.

    :param x: the json object
    :param sep: the separator to use in the key names
    :return: the flattened dictionary
    """

    obj = {}
    def recurse(t, parent_key=''):
        if isinstance(t, list):
            for i, ti in enumerate(t):
                recurse(ti, f'{parent_key}{sep}{i}' if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, f'{parent_key}{sep}{k}' if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(x)
    return obj


def build_fn(input_shape, optimizer, loss, output_activation, hidden_activation, metrics):
    """
    Custom model build function can take any parameters you want to build a network
    for your model. You should customize this to parameterize the models you want
    to explore. (For example, this function builds linear models with different
    input_shape, optimizer, loss, output_activation, and metrics.)

    :param input_shape: the shape of each sample (image)
    :param optimizer: the optimizer function
    :param loss: the loss function
    :param output_activation: the output activation function
    :param metrics: other metrics to track
    :return: a compiled model ready to 'fit'
    """
    # make sure to clear any previous nodes in the computation graph to save memory
    clear_session()

    # Fully connected linear model
    model = Sequential([
        Input(shape=input_shape),
        Rescaling(1/255),
        #RandomRotation((0.5, 0.5)),
        Conv2D(16, 3, activation=hidden_activation),
        Conv2D(8, 3, activation=hidden_activation),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(4, 3, activation=hidden_activation),
        Conv2D(2, 3, activation=hidden_activation),
        Flatten(),
        Dense(1, activation=output_activation)
    ])
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def analyze_models(output_dir):
    data = []
    histories = []
    for model_name in os.listdir(output_dir):
        model_dir = os.path.join(output_dir, model_name)
        #print("Current Model: {}".format(model_dir))
        # make CSV with results
        params_file = os.path.join(model_dir, 'params.json')
        if not os.path.isfile(params_file):
            print(f'Warning: {params_file} does not exist.')
            continue
        with open(params_file, 'r') as fp:
            #print("Current params file: {}".format(params_file))
            raw_str = fp.read()
            params = json.loads(raw_str)
            #params = json.load(fp)
            
            #print(type(params))
            #print(params)
            #print(params.keys())
            results = params['results']
            history = results['history']
            del params['results']
            params = flatten(params)
            i = results['best_epoch']
            for k in history:
                params[k] = history[k][i]
            params['runtime'] = results['runtime']
            params['runtime'] = results['timestamp']
            params['name'] = results['name']
            history['name'] = results['name']
            histories.append(history)
            data.append(params)

    df = pd.DataFrame(data=data)
    columns = df.columns
    first_columns = ['name', 'val_loss', 'val_accuracy', 'loss', 'accuracy', 'runtime']
    last_columns = sorted([c for c in columns if c not in first_columns])
    columns = first_columns + last_columns
    df = df[columns]
    df = df.sort_values(by=['val_loss'], ascending=True).copy()
    csv_file = os.path.join(output_dir, 'results.csv')
    df.to_csv(csv_file, index=False)

    # create parallel coordinates
    quantiles = 10
    color = 'val_loss'
    color_cat = f'{color}_cat'
    range_color = tuple(np.percentile(df[color], [25, 100] if 'accuracy' in color else [0, 75]))
    df = df.sort_values(by=[color], ascending=True).copy()
    df[color_cat] = pd.cut(df[color], np.percentile(df[color], np.linspace(0, 100, quantiles+1)), include_lowest=True)
    use_columns = [c for c in last_columns if len(df[c].value_counts()) > 1] + [color_cat]
    df[use_columns] = df[use_columns].astype(str)
    fig = px.parallel_categories(df, dimensions=use_columns, color=color, range_color=range_color,
                                 color_continuous_scale=px.colors.sequential.Inferno)
    parcat_file = os.path.join(output_dir, 'parallel_categories.html')
    fig.write_html(parcat_file)

    # create history plots
    histories = {k: list(v.values()) for k, v in pd.DataFrame(data=histories).to_dict().items()}
    for subset in ['val_', '']:
        for metric in ['loss', 'accuracy']:
            k = subset + metric
            x = dict(zip(histories['name'], histories[k]))
            n = max(len(x[k]) for k in x)
            x = {k: x[k] + [float('nan')] * (n - len(x[k])) for k in x}
            df = pd.DataFrame(data=x).stack().to_frame(name='y').reset_index(names=['epoch', 'name'])
            fig = px.line(df, x='epoch', y='y', color='name')
            html_file = os.path.join(output_dir, f'{k}.html')
            fig.write_html(html_file)
            html_path = os.path.join(os.getcwd(), html_file)
            print(f'Follow the link to see the {k} plot:\nfile://{html_path}')

    html_path = os.path.join(os.getcwd(), parcat_file)
    print(f'Follow the link to see the figure:\nfile://{html_path}')
    csv_path = os.path.join(os.getcwd(), csv_file)
    print(f'Open the CSV file to see the table:\nfile://{csv_path}')


def grid_search(data_path, output_dir):
    # put the path to your endpoint here
    # select an output dir for all of your models here
    # every time you run this program, new models will be created as subfolders in the output_dir
    os.makedirs(output_dir, exist_ok=True)

    # load in some words to use to create model names
    words = pd.read_csv('eff_large_wordlist.txt', sep='\t', names=['roll', 'word'])['word']

    # define parameter grid here
    if ('anthyllis' in output_dir):
        # Group 
        parameter_grids = {
            'dataset_params': ParameterGrid({
                'label_mode': ['binary'],
                'image_size': [(64, 64)],
                'batch_size': [128],
                'color_mode': ['grayscale'],
                'interpolation': ['bilinear'],
                'crop_to_aspect_ratio': [False],
            }),
            'build_fn_params': ParameterGrid({
                'optimizer': ['adam'],
                'loss': ['binary_crossentropy'],
                'output_activation': ['sigmoid'],
                'hidden_activation': ['relu'],
            }),
            'fit_params': ParameterGrid({
                'epochs': [10**10],  # must be integers
            }),
            'early_stopping_params': ParameterGrid({
                'patience': [50],
            })
        }
        
    elif ('dryopteris' in output_dir):
        # Personal
        parameter_grids = {
            'dataset_params': ParameterGrid({
                'label_mode': ['binary'],
                'image_size': [(128, 128)], 
                'batch_size': [64],
                'color_mode': ['grayscale'],
                'interpolation': ['bilinear'],
                'crop_to_aspect_ratio': [False],
            }),
            'build_fn_params': ParameterGrid({
                'optimizer': ['adam'],
                'loss': ['binary_crossentropy'],
                'output_activation': ['sigmoid'],
                'hidden_activation': ['relu'],
            }),
            'fit_params': ParameterGrid({
                'epochs': [10**10],  # must be integers
            }),
            'early_stopping_params': ParameterGrid({
                'patience': [50],
            })
        }

    else:
        parameter_grids = {
            'dataset_params': ParameterGrid({
                'label_mode': ['binary'],
                'image_size': [(2**p, 2**p) for p in range(5, 9)],
                'batch_size': [2**p for p in range(5, 8)],
                'color_mode': ['grayscale', 'rgb'],
                'interpolation': ['bilinear'],
                'crop_to_aspect_ratio': [False],
            }),
            'build_fn_params': ParameterGrid({
                'optimizer': ['adam'],
                'loss': ['binary_crossentropy'],
                'output_activation': ['sigmoid'],
                'hidden_activation': ['relu'],
            }),
            'fit_params': ParameterGrid({
                'epochs': [10**10],  # must be integers
            }),
            'early_stopping_params': ParameterGrid({
                'patience': [50],
            })
        }

    for params in itertools.product(*list(parameter_grids.values())):
        # store parameters in dictionary, one item per process
        params = dict(zip(parameter_grids.keys(), params))

        # input_shape can be determined from the dataset_kwargs
        color_mode = params['dataset_params']['color_mode']
        image_size = params['dataset_params']['image_size']
        num_channels = {'rgb': 3, 'rgba': 4, 'grayscale': 1}[color_mode]
        input_shape = (image_size[0], image_size[1], num_channels)

        # create train and validation dataset using dataset_params
        train_ds = image_dataset_from_directory(os.path.join(data_path, 'train'), **params['dataset_params'])
        valid_ds = image_dataset_from_directory(os.path.join(data_path, 'valid'), **params['dataset_params'])
        #print("\n\t {}\n\t".format(str(params['dataset_params'])))

        # create early stopping with early stopping parameters
        early_stopping = EarlyStopping(monitor='val_loss', verbose=1, **params['early_stopping_params'])

        # Reduces learning rate if plateua is detected.
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=100, factor=0.2, min_lr=0.001)

        # to make it easier to remember the best model we'll use a random name
        random_name = '_'.join(words.sample(n=2))
        output_path = os.path.join(output_dir, random_name)
        os.makedirs(output_path, exist_ok=False)

        # create model checkpoint with output path
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(output_path, 'model.h5'),  # always overwrite the existing model
            save_weights_only=False, save_freq='epoch',
            save_best_only=True, monitor='val_loss', verbose=1)  # only save models that improve the 'monitored' value
        callbacks = [early_stopping, model_checkpoint]#, reduce_lr]

        # build model using build_fn_kwargs
        model = build_fn(input_shape=input_shape, metrics=['accuracy'], **params['build_fn_params'])

        # train model using fit_kwargs
        t0 = time.time()
        history = model.fit(train_ds, validation_data=valid_ds, callbacks=callbacks, **params['fit_params'])
        t1 = time.time()

        # save some extra stuff with the parameters for this model
        params['results'] = {
            'runtime': t1 - t0,
            'best_val_loss': early_stopping.best,
            'best_epoch': early_stopping.best_epoch,
            'monitor': early_stopping.monitor,
            'data_path': data_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'name': random_name,
            'history': {k: history.history[k] for k in history.history},
        }
        print('params:', params)

        print('name:', params['results']['name'])
        # save params in model directory
        params_file = os.path.join(output_path, 'params.json')
        with open(params_file, 'w') as fp:
            #json.dump(params.items(), fp)
            #json.dump(str(params), fp)
            json.dump(params, fp)

def make_specific(input_shape, hidden_activation, output_activation, loss, optimizer, metrics):
    clear_session()
    
    model = Sequential([
        Input(shape = input_shape),
        Rescaling(1/255),
        Conv2D(16, 3, activation = hidden_activation),
        Conv2D(8, 3, activation = hidden_activation),
        MaxPooling2D(pool_size = (2,2)),
        Conv2D(4, 3, activation = hidden_activation),
        Conv2D(2, 3, activation = hidden_activation),
        Flatten(),
        Dense(1, activation = output_activation)
    ])
    model.summary()
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    return model





def fit_specific(data_path, output_dir, image_size, batch_size, color_mode, interpolation, ctar, model):
    train_ds = image_dataset_from_directory(os.path.join(data_path, 'train'), label_mode = 'binary', image_size = image_size, batch_size = batch_size, color_mode = color_mode, interpolation = interpolation, crop_to_aspect_ratio = ctar)
    valid_ds = image_dataset_from_directory(os.path.join(data_path, 'valid'), label_mode = 'binary', image_size = image_size, batch_size = batch_size, color_mode = color_mode, interpolation = interpolation, crop_to_aspect_ratio = ctar)

    early_stopping = EarlyStopping(monitor = 'val_loss', verbose = 1, patience = 50)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', verbose = 1, patience = 40, factor = 0.2, min_lr = 0.001)
    
    output_name = 'specific'
    output_path = os.path.join(output_dir, output_name)
    output_path = output_dir
    os.makedirs(output_path, exist_ok = True)

    model_checkpoint = ModelCheckpoint(
        filepath = os.path.join(output_path, 'model.h5'),
        save_weights_only = False,
        save_freq = 'epoch',
        save_best_only = True,
        monitor = 'val_loss',
        verbose = 1
    )
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    t0 = time.time()
    history = model.fit(train_ds, validation_data = valid_ds, callbacks = callbacks, epochs = 10**10) 
    t1 = time.time()





if __name__ == '__main__':
    # output_dir, data_path = 'lotus_leaves_1', 'groups/lotus corniculatus l vs viola tricolor l'
    group_problem = 'anthyllis vulneraria l vs mentha aquatica l' # Group
    personal_problem = 'dryopteris filix-mas (l.) schott vs solanum nigrum l' # Personal
    
    data_path_group = group_problem
    data_path_personal = personal_problem
    output_dir_group = "./output/" + group_problem
    output_dir_personal = "./output/" + personal_problem
    
    submission_dir_group = "./submissions/" + group_problem
    submission_dir_personal = "./submissions/" + personal_problem

    paths_dict = {
        "group": [data_path_group, output_dir_group, submission_dir_group],
        "personal": [data_path_personal, output_dir_personal, submission_dir_personal]
    }

    # Group Problem
    grid_search(paths_dict['group'][0], paths_dict['group'][2])
    #analyze_models(paths_dict['group'][1])
    # best: relu, 128 batch, grayscale, 64x64
    #group_model = make_specific((64, 64, 1), 'relu', 'sigmoid', 'binary_crossentropy', 'adam', ['accuracy'])
    #fit_specific(paths_dict['group'][0], paths_dict['group'][2], (64, 64), 128, 'grayscale', 'bilinear', False, group_model)

    # Personal Problem
    grid_search(paths_dict['personal'][0], paths_dict['personal'][2])
    #analyze_models(paths_dict['personal'][1])
    # best: relu, 64 batch, grayscale, 128x128
    #personal_model = make_specific((128, 128, 1), 'relu', 'sigmoid', 'binary_crossentropy', 'adam', ['accuracy'])
    #fit_specific(paths_dict['personal'][0], paths_dict['personal'][2], (128, 128), 64, 'grayscale', 'bilinear', False, personal_model)






