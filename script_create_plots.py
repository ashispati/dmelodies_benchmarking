import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

from src.utils.evaluation import EVAL_METRIC_DICT
from src.utils.plotting import create_box_plot, create_scatter_plot, create_bar_plot, create_violin_plot


def get_results_path(folder_name):
    return os.path.join(
        'src',
        'saved_models',
        folder_name,
        'results_dict.json'
    )


def get_beta_vae_results_file(model_type, params):
    if model_type == 'dmelCNN':
        folder_name = f'DMelodiesVAE_CNN_beta-VAE_b_{params[0]}_c_{params[1]}_r_{params[2]}_'
    elif model_type == 'dmelRNN':
        folder_name = f'DMelodiesVAE_RNN_beta-VAE_b_{params[0]}_c_{params[1]}_r_{params[2]}_'
    elif model_type == 'dsprCNN':
        folder_name = f'DspritesVAE_beta-VAE_b_{params[0]}_c_{params[1]}_r_{params[2]}_'
    else:
        raise ValueError("Invalid src type")
    return get_results_path(folder_name)


def get_annealed_vae_results_file(model_type, params):
    if model_type == 'dmelCNN':
        folder_name = f'DMelodiesVAE_CNN_annealed-VAE_b_{params[0]}_c_{params[1]}_r_{params[2]}_'
    elif model_type == 'dmelRNN':
        folder_name = f'DMelodiesVAE_RNN_annealed-VAE_b_{params[0]}_c_{params[1]}_r_{params[2]}_'
    elif model_type == 'dsprCNN':
        folder_name = f'DspritesVAE_annealed-VAE_b_{params[0]}_c_{params[1]}_r_{params[2]}_'
    else:
        raise ValueError("Invalid src type")
    return get_results_path(folder_name)


def get_factor_vae_results_file(model_type, params):
    if model_type == 'dmelCNN':
        folder_name = f'FactorVAE_CNN_b_{params[0]}_c_{params[1]}_g_{params[2]}_r_{params[3]}_nowarm_'
    elif model_type == 'dmelRNN':
        folder_name = f'FactorVAE_RNN_b_{params[0]}_c_{params[1]}_g_{params[2]}_r_{params[3]}_nowarm_'
    elif model_type == 'dsprCNN':
        folder_name = f'DspritesFactorVAE_b_{params[0]}_c_0_g_{params[2]}_r_{params[3]}_'
    else:
        raise ValueError("Invalid src type")
    return get_results_path(folder_name)

d1 = '#0f5e89'
d2 = '#c45277'
d3 = '#7bb876'
dark_colors = [d1, d2, d3]

vae_type_dict = {
    r'$\beta$-VAE': get_beta_vae_results_file,
    'Annealed-VAE': get_annealed_vae_results_file,
    'Factor-VAE': get_factor_vae_results_file,
}

seed_list = [0, 1, 2]
vae_param_dict = {
    r'$\beta$-VAE': list(product([0.2, 1.0, 4.0], [50.0], seed_list)),
    'Annealed-VAE': list(product([1.0], [25.0, 50.0, 75.0], seed_list)),
    'Factor-VAE': list(product([1], [50], [1, 10, 50], seed_list)),
}
vae_param__values_dict = {
    r'$\beta$-VAE': (r'$\beta$', 0),
    'Annealed-VAE': (r'$C$', 1),
    'Factor-VAE': (r'$\gamma$', 2),
}

model_type_dict = {
    'dmelCNN': 'dMelodies-CNN',
    'dmelRNN': 'dMelodies-RNN',
    'dsprCNN': 'dSprites-CNN'
}


def main():
    # create plots folder if it doesn't exist
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(cur_dir, "plots")):
        os.mkdir(os.path.join(cur_dir, "plots"))

    # PLOT HYPERPARAMETER SENSITIVITY SCATTER PLOT
    for v in vae_type_dict.keys():
        data = []
        for m in model_type_dict.keys():
            fp_function = vae_type_dict[v]
            temp_list = []
            m_list = []
            acc_list = []
            param_list = []
            num_exps = 0
            a = vae_param_dict[v]
            for p in a:
                results_fp = fp_function(m, p)
                # if results_fp is None:
                #     continue
                # if not os.path.exists(results_fp):
                #     continue
                with open(results_fp, 'r') as infile:
                    results_dict = json.load(infile)
                m_list.append(results_dict['mig'])
                acc_list.append(results_dict['test_acc'] * 100)
                param_list.append(str(p[vae_param__values_dict[v][1]]))
                num_exps += 1
            if len(m_list) != 0:
                temp_list.append(m_list)
                temp_list.append(acc_list)
                temp_list.append(num_exps * [model_type_dict[m]])
                temp_list.append(param_list)
                data.append(temp_list)
        data = np.concatenate(data, axis=1).T
        column_1 = 'MIG'
        column_2 = 'Reconstruction Accuracy (in %)'
        column_3 = 'Model'
        column_4 = vae_param__values_dict[v][0]
        df = pd.DataFrame(columns=[column_1, column_2, column_3, column_4], data=data)
        df[column_1] = df[column_1].astype(float)
        df[column_2] = df[column_2].astype(float)
        if v == r'$\beta$-VAE':
            v = 'beta-VAE'
        save_path = os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'plots', f'hyperparam_results_{v}.pdf'
        )
        fig, ax = create_scatter_plot(
            data_frame=df,
            x_axis=column_1,
            y_axis=column_2,
            grouping=column_3,
            style=column_4,
            d_list=dark_colors
        )
        plt.savefig(save_path)

    # PLOT DISENTANGLEMENT BOX PLOT
    for e in EVAL_METRIC_DICT.keys():
        data = []
        for v in vae_type_dict.keys():
            for m in model_type_dict.keys():
                fp_function = vae_type_dict[v]
                temp_list = []
                m_list = []
                p_list = []
                num_exps = 0
                a = vae_param_dict[v]
                for p in a:
                    results_fp = fp_function(m, p)
                    # if results_fp is None:
                    #     continue
                    # if not os.path.exists(results_fp):
                    #     continue
                    with open(results_fp, 'r') as infile:
                        results_dict = json.load(infile)
                    m_list.append(results_dict[e])
                    p_list.append(p)
                    num_exps += 1
                if len(m_list) != 0:
                    temp_list.append(m_list)
                    temp_list.append(num_exps * [model_type_dict[m]])
                    temp_list.append(num_exps * [v])
                    temp_list.append(p_list)
                    data.append(temp_list)
        data = np.concatenate(data, axis=1).T
        df = pd.DataFrame(columns=[EVAL_METRIC_DICT[e], 'Model', 'Method', 'Param'], data=data)
        save_path = f'/Users/som/Desktop/aggregated_results_{e}.csv'
        df.to_csv(save_path)
        df[EVAL_METRIC_DICT[e]] = df[EVAL_METRIC_DICT[e]].astype(float)
        model_list = [m for m in vae_type_dict.keys()]
        save_path = os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'plots', f'disent_results_{EVAL_METRIC_DICT[e]}.pdf'
        )
        y_axis_range = None
        location='upper left'
        if e == 'modularity_score':
            y_axis_range = (0.7, 1.0)
            location = 'lower left'
        fig, ax = create_box_plot(
            data_frame=df,
            model_list=model_list,
            d_list=dark_colors,
            x_axis='Model',
            y_axis=EVAL_METRIC_DICT[e],
            grouping='Method',
            width=0.5,
            legend_on=True,
            location=location,
            y_axis_range=y_axis_range
        )
        plt.savefig(save_path)

    # PLOT RECONSTRUCTION BOX PLOT
    data = []
    for v in vae_type_dict.keys():
        for m in model_type_dict.keys():
            fp_function = vae_type_dict[v]
            temp_list = []
            m_list = []
            p_list = []
            num_exps = 0
            a = vae_param_dict[v]
            for p in a:
                results_fp = fp_function(m, p)
                # if results_fp is None:
                #     continue
                # if not os.path.exists(results_fp):
                #     continue
                with open(results_fp, 'r') as infile:
                    results_dict = json.load(infile)
                m_list.append(results_dict['test_acc'] * 100)
                p_list.append(p)
                num_exps += 1
            if len(m_list) != 0:
                temp_list.append(m_list)
                temp_list.append(num_exps * [model_type_dict[m]])
                temp_list.append(num_exps * [v])
                temp_list.append(p_list)
                data.append(temp_list)
    data = np.concatenate(data, axis=1).T
    column_label = 'Reconstruction Accuracy (in %)'
    df = pd.DataFrame(columns=[column_label, 'Model', 'Method', 'Param'], data=data)
    df[column_label] = df[column_label].astype(float)
    save_path = f'/Users/som/Desktop/aggregated_results_recons.csv'
    df.to_csv(save_path)
    model_list = [m for m in vae_type_dict.keys()]
    save_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), 'plots', f'recons_results.pdf'
    )
    fig, ax = create_box_plot(
        data_frame=df,
        model_list=model_list,
        d_list=dark_colors,
        x_axis='Model',
        y_axis=column_label,
        grouping='Method',
        width=0.5,
        legend_on=True,
        location='lower right',
    )
    plt.savefig(save_path)


if __name__ == '__main__':
    main()