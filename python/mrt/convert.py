import os
import argparse
import logging
import random
import numpy as np
from os import path

import mxnet as mx

from mrt import utils, conf
from mrt.gluon_zoo import save_model

class QUANT_TYPE:
    PTQ = 'PTQ'
    QAT = 'QAT'

class DEVICE:
    CPU = 'CPU'
    GPU = 'GPU'

class PTQ_SECTION:
    PREPARE = 'PREPARE'

# TODO: move some of the helper funcs into other files.
class convert_helper_funcs:
    def _seed_all(seed=1013):
        """Set ALL the seed of random env.
        TODO: fill the comment.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        mx.random.seed(seed)

    def _load_fname(prefix, suffix=None, with_ext=False):
        """Get the model files at a given stage.

        Parameters
        ----------
        prefix : string
            The file path without and extension.
        suffix : string
            The file suffix with respect to a given stage of MRT.
        with_ext: bool
            Whether to include ext file.

        Returns
        -------
        files : tuple of string
            The loaded file names.
        """
        suffix = "."+suffix if suffix is not None else ""
        return utils.extend_fname(prefix+suffix, with_ext)

    def _get_ctx(device, id_list, dctx=mx.cpu()):
        """Get the context specified in configuration file.
            WARNING: Because it's not recommanded to set the GPU ID in the application, \'Device_ids\' is abandoned.
                     Please use Env Variable CUDA_VISIBLE_DEVICES to assign the GPUs to MRT..
        TODO: fill the comments.
        Returns
        -------
        path : mxnet.context
            The context specified in the option.
        """
        contex = dctx
        device_type = args.device
        
        if device_type == 'GPU':
            contex = mx.gpu(device_ids[0]) if len(device_ids) == 1 \
                  else [mx.gpu(i) for i in num]
        else:
            device_ids = _get_val(config, section, 'Device_ids', dval='')
            # _check(device_ids == '', section, 'Device_ids',
                   # message='`Device_ids` should be null given `cpu` device type')
        return contex

def post_training_quant(args, logger):
    logger.info('Start PTQ.')
    model_name = args.model_name
    model_prefix = path.join(args.model_path, model_name)
    model_ctx = args.ctx

    sec = PTQ_SECTION.PREPARE
    logger.debug(f'Start {sec}.')




if __name__ == "__main__":
    # Define Args
    parser = argparse.ArgumentParser(description='''Welcome to MRT (Model Representing Tool). ''' 
    '''It's designed to support both PTQ(Post-Training Quantization) and QAT(Quant-Aware Training). '''
    '''MRT holds a packet of almost all the sart methods for PTQ, and it's being updated. '''
    '''For QAT, MRT provides only kinds of typical implementation for some common demands. '''
    '''It's highly recommended to follow the DOC (TODO: add link to doc) and implement specifically the
    numerous methods to bring up your model, since it's a tricky thing all the time.''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbosity', default=logging.NOTSET, choices=[logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR], type=int, help='Indicate the level of logging.')
    # General parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='Random seed for stable results reproduction.')
    parser.add_argument('--model_name', '-n', required=True, type=str, help='The name of the target model.')
    parser.add_argument('--model_path', '-p', default=conf.MRT_MODEL_ROOT, type=str, help='The path to the target model.')
    
    parser.add_argument('--batch_size', default=64, type=int, help='The size of mini-batch for data loader. It has an effect on the quantized model\'s acc when utilized batch-related method.')
    parser.add_argument('--workers', default=4, type=int, help='The number of workers(processes) for data loader.')
    parser.add_argument('--dataset_name', '-d', required=True, type=str, help='The path to dataset.')
    parser.add_argument('--dataset_path', default=conf.MRT_DATASET_ROOT, type=str, help='The path to dataset.')

    # Quantization parameters
    parser.add_argument('--quant_type', '-q', required=True, default=QUANT_TYPE.PTQ, choices=[QUANT_TYPE.PTQ, QUANT_TYPE.QAT], type=str, help='PTQ or QAT.')
    parser.add_argument('--n_bits_w', default=32, type=int, help='Bitwidth for weight quantization.')
    parser.add_argument('--channel_wise', action='store_true', help='Apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=32, type=int, help='Bitwidth for activation quantization')
    parser.add_argument('--act_quant', default=True, action='store_true', help='Apply activation quantization.')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # Weight calibration parameters
    # TODO: define types of weight quantization. For now, it supports only AdaRound.
    parser.add_argument('--num_samples', default=1024, type=int, help='The size of the calibration dataset')
    parser.add_argument('--w_iters', default=20000, type=int, help='The number of iteration for opt-based methods in weights (e.g., AdaRound).')
    parser.add_argument('--sym', action='store_true', help='The symmetric reconstruction, not recommended.')
    parser.add_argument('--weight_rovre_adar', default=0.01, type=int, help='[AdaRound] The weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--b_start_adar', default=20, type=int, help='[AdaRound] The temperature at the beginning of calibration.')
    parser.add_argument('--b_end_adar', default=2, type=int, help='[AdaRound] The temperature at the end of calibration.')
    parser.add_argument('--warmup_adar', default=0.2, type=float, help='[AdaRound] During the warmup period, no regularization is applied.')
    parser.add_argument('--record_step_adar', default=20, type=int, help='[AdaRound] Record SNN output per step.')

    # Activation calibration parameters
    # TODO: add more methods for activation quantization. For now, it supports only LSQ.
    parser.add_argument('--a_iters', default=5000, type=int, help='The number of iteration for opt-based methods in weights (e.g., LSQ).')
    parser.add_argument('--a_lr_lsq', default=4e-4, type=float, help='[LSQ] Learning rate.')
    parser.add_argument('--a_p_lsq', default=2.4, type=float, help='[LSQ] Norm Minimization.')

    args = parser.parse_args()

    # Init logging
    utils.log_init(level=args.verbosity)
    logger = logging.getLogger("log.main")

    # Init Rand Seed
    _seed_all(args.seed)

    if args.quant_type == QUANT_TYPE.PTQ:
        post_training_quant(args, logger)
    elif args.quant_type == QUANT_TYPE.QAT:
        err_info = f"Not yet the QAT implemented."
        logger.error(err_info)
        raise NotImplementedError(err_info)
    else:
        err_info = f"Unknown error for args.quant_type is {args.quant_type} that is out of choices."
        logger.error(err_info)
        raise NameError(err_info)