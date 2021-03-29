import os
import argparse
import logging
import random
import numpy as np
from os import path

import mxnet as mx

from mrt import utils, conf
from mrt.gluon_zoo import save_model
from mrt.transformer import Model

class QUANT_TYPE:
    PTQ = 'PTQ'
    QAT = 'QAT'


class DEVICE:
    CPU = 'CPU'
    GPU = 'GPU'


class PTQ_SECTION:
    PREPARE = 'PREPARE'
    SPLIT_MODEL = 'SPLIT_MODEL'


# TODO: move some of the helper funcs into other files.
class ConvertHelperFuncs:
    @staticmethod
    def _seed_all(seed=1013):
        """Set ALL the seed of random env.
        TODO: fill the comment.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        mx.random.seed(seed)

    @staticmethod
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

    @staticmethod
    def _check(expression, section, option, message='Not a valid value'):
        """check whether an operation of main2 if valid and report error message if invalid.

        Parameters
        ----------
        expression : bool
            The judgement conditions in main2.
        section : string
            The section of configuration file.
        option : string
            The option of the section.
        message : string
            The error message to be reported.
        """
        assert expression, message + '.\noption `%s` in section `%s`' \
            % (option, section)

    @staticmethod
    def _get_ctx(device_type, d_num, dctx=mx.cpu()):
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
        if device_type == 'GPU':
            contex = mx.gpu(0) if d_num == 1 \
                  else [mx.gpu(i) for i in range(d_num)]
        else:
            contex = mx.cpu()
            # TODO: check if mx supports the specific cpu (Core Binding).
        return contex
    
    @staticmethod
    def set_batch(input_shape, batch):
        """Get the input shape with respect to a specified batch value and an original input shape.

        Parameters
        ----------
        input_shape : tuple
            The input shape with batch axis unset.
        batch : int
            The batch value.

        Returns
        -------
        ishape : tuple
            The input shape with the value of batch axis equal to batch.
        """
        return [batch if s == -1 else s for s in input_shape]


def post_training_quant_prepare(start_here:bool, dump: bool,
    model_name:str, model_prefix:str, model_ctx:mx.Context, input_shape:tuple, batch_size: int):
    """The prepare section of PTQ. In this section, a model with specific name and dir is loaded and it has been reshaped with input_shape and batch_size.
        The reshape function transforms input_shape's element that is equivalent to -1 into batch_size. 
        TODO: fill the comments.
        Returns
        -------
        kwreses : dict
            A dict includes the prepared model (transformer.Model) with key 'model'.
        """

    logger.debug(f'Start {PTQ_SECTION.PREPARE}. model_name: {model_name}, model_prefix: {model_prefix}, '
                 f'model_ctx: {model_ctx}, input_shape: {input_shape}, batch_size:{batch_size}')
    if -1 not in input_shape:
        logger.warn(f'The input_shape does not includes -1, the batch_size: {batch_size} will not be used.')
    sym_file, prm_file = ConvertHelperFuncs._load_fname(model_prefix, suffix=PTQ_SECTION.PREPARE.lower())
    sym_path, prm_path = ConvertHelperFuncs._load_fname(model_prefix)
    if not path.exists(sym_path) or not path.exists(prm_path):
        logger.debug("Not path.exists(sym_path) or not path.exists(prm_path).")
        save_model(model_name, data_dir=args.model_dir_path)
    if not start_here:
        model = Model.load(sym_path, prm_path)
        model.prepare(ConvertHelperFuncs.set_batch(input_shape, batch_size))
        if dump:
            model.save(sym_file, prm_file)
        logger.info("Prepare stage finished.")
    else:
        ConvertHelperFuncs._check(path.exists(sym_file) and path.exists(prm_file), 'DEFAULT',
               'Start', message=f"Check point of {PTQ_SECTION.PREPARE} is not found in {prm_file}, " \
               f"please move the start point earlier.")
        model = Model.load(sym_file, prm_file)
        logger.info("{PTQ_SECTION.PREPARE} stage checked")
    return {
        'model': model,
    }

def post_training_quant(args, logger):
    """The progress of quant of a CV model.
    TODO: fill the comments; design a more general Quanter class for different models.
    """
    logger.info('Start PTQ.')
    model_name = args.model_name
    model_dir_path = args.model_dir_path
    model_prefix = path.join(model_dir_path, model_name)
    model_ctx = ConvertHelperFuncs._get_ctx(args.device_type, args.device_num)
    input_shape = args.input_shape
    batch_size = args.batch_size
    dump = args.dump
    
    sec = PTQ_SECTION.PREPARE
    pre_kwargs = {'model_name':model_name, 'model_prefix': model_prefix, 'model_ctx':model_ctx, 'input_shape':input_shape, 'batch_size': batch_size}
    kwreses = post_training_quant_prepare(start_here=False, dump=dump, **pre_kwargs)
    assert 'model' in kwreses
    model = kwreses['model']
    print(model.params)
    sec = PTQ_SECTION.SPLIT_MODEL
    #PASS

    sec = 'QUANTIZATION'


if __name__ == "__main__":
    # Define Args
    parser = argparse.ArgumentParser(description='''Welcome to MRT (Model Representing Tool). '''
    '''It's designed to support both PTQ(Post-Training Quantization) and QAT(Quant-Aware Training). '''
    '''MRT holds a packet of almost all the sart methods for PTQ, and it's being updated. '''
    '''For QAT, MRT provides only kinds of typical implementation for some common demands. '''
    '''It's highly recommended to follow the DOC (TODO: add link to doc) and implement specifically the'''
    '''numerous methods to bring up your model, since it's a tricky thing all the time.''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbosity', default=logging.NOTSET, choices=[logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR], type=int, help='Indicate the level of logging.')
    # General parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='Random seed for stable results reproduction.')
    parser.add_argument('--model_name', '-n', required=True, type=str, help='The name of the target model.')
    parser.add_argument('--model_dir_path', '-p', default=conf.MRT_MODEL_ROOT, type=str, help='The path to the target model\'s dir.')
    parser.add_argument('--input_shape', '-s', required=True, nargs='+', type=int, help="The shape of tuple. The batch dim should be set to -1.")
    parser.add_argument('--batch_size', default=64, type=int, help='The size of mini-batch for data loader. It has an effect on the quantized model\'s acc when utilized batch-related method.')
    parser.add_argument('--workers', default=4, type=int, help='The number of workers(processes) for data loader.')
    parser.add_argument('--dataset_name', '-d', required=True, type=str, help='The path to dataset.')
    parser.add_argument('--dataset_path', default=conf.MRT_DATASET_ROOT, type=str, help='The path to dataset.')
    parser.add_argument('--device_type', '-D', default=DEVICE.CPU, choices=[DEVICE.GPU, DEVICE.CPU], type=str, help='Assign the device MXNet running on.')
    parser.add_argument('--device_num', '-N', default=1, type=int, help='# of GPU devices.')
    parser.add_argument('--dump', action='store_true', default=True, help='Dump intermediate data.')
    # Quantization parameters
    parser.add_argument('--quant_type', '-q', default=QUANT_TYPE.PTQ, choices=[QUANT_TYPE.PTQ, QUANT_TYPE.QAT], type=str, help='PTQ or QAT.')
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
    ConvertHelperFuncs._seed_all(args.seed)

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
