""" Collection of MRT pass tions.
    Graph-level symbol helper function for MRT.
    Stage-level symbol pass designation for MRT.
"""

from mxnet import ndarray as nd
import math
import numpy as np
import time

from .tfm_utils import get_bit, scale, requant
from .sym_utils import is_var, is_params, is_inputs
from .tfm_base import *
from . import dataset as ds
from . import utils
from . import sim_quant_helper as sim

# === symbol pass == 

def calculate_ops(symbol, params, normalize=True):
    """ Customized graph-level topo pass definition.

        Calulate the numbe of operations with respect to the given input shape.
    """
    ops, infer_shapes = [0], infer_shape(symbol, params)
    def _impl(op, **kwargs):
        ops[0] += apply_pass("calculate_ops")(op, **kwargs)
    topo_visit_transformer(symbol, params, _impl,
            infer_shapes=infer_shapes)

    ops = ops[0]
    if normalize:
        LEVELS = ['', 'K', 'M', 'G', 'T', 'P']
        idx = 0
        while ops > 1000:
            ops /= 1000
            idx += 1
        ops = "{:5.2f}{}".format(ops, LEVELS[idx])
    return ops

@N.register_nm("fuse_transpose")
def fuse_transpose(symbol, params):
    """ Customized graph-level topo pass definition.

        Equivalent graph transformation. 
        Fuse or swap the current operator with the former Transpose.
    """
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("fuse_transpose", infer_shapes=infer_shapes))

@N.register_nm("rewrite")
def rewrite(symbol, params):
    """ Customized graph-level topo pass definition.

        Equivalent graph transformation. 
        Rewrite the current symbol into equivalent ones that is feasible for quantization.
    """
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("rewrite", infer_shapes=infer_shapes))

@N.register_nm("quantize")
def quantize(symbol, params, th_dict, precs, scales, op_input_precs,
             restore_names, shift_bits, softmax_lambd):
    """ Customized graph-level topo pass definition.

        MRT quantization process function.

        The original graph can be denoted as follows:

        **Case 1. Enable Restore**

        .. code-block::

            X1   X2  ...  Xn
            xs1  xs2 ...  xsn
            \    |       /
             \   |      /
              \  |     /
              current_op

        where 'xs1', 'xs2', etc are respectively stand for the scale of the inputs.

        The original graph will be restored into:

        .. code-block::

            X1   X2  ... Xn
            xs1  xs2 ... xsn
            |    |       |
            |    |       |
            RX1  RX2 ... RXn
            1    1       1
            \    |       /
             \   |      /
              \  |     /
              current_op
                 1
                 |
                 |
                ...

        where 'RX1', 'RX2' ... stand for restored symbol with unit scale.

        **Case 2. Other Cases**

        The original graph will be quantized into:

        .. code-block::

                    X1   X2  ... Xn
                    xs1  xs2 ... xsn
                    |    |       |
                    |    |       |
                    QX1  QX2 ... QXn
                    \    |       /
                     \   |      /
                      \  |     /
            [equivalent quantization subgraph]
                         |
                         |
                        ...

        For either cases, if the 'current_op' belongs to one of the output symbols, 
        it will quantized again with respect to the specified output precision.
    """
    infer_shapes = infer_shape(symbol, params)

    def restore(op, **kwargs):
        th_dict, precs, scales = kwargs['th_dict'], kwargs['precs'], kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()

        childs = [] if childs is None else childs

        new_childs = [c / scales[c.attr('name')] \
            if scales.get(c.attr('name'), 1) != 1 else c \
                     for c in childs]

        out = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        precs[name][OUT_KEY] = get_bit(th_dict[name])
        scales[name] = 1

        return out

    def _quant(op, **kwargs):
        op = apply_pass("quantize",
            infer_shapes=kwargs['infer_shapes'],
            th_dict=kwargs['th_dict'],
        )(op, **kwargs) if op.attr('name') not in restore_names \
            else restore(op, **kwargs)

        if is_var(op, kwargs['params']):
            return op

        name = op.attr('name')
        th_dict, scales = kwargs['th_dict'], kwargs['scales']
        precs = kwargs['precs']
        th = th_dict[name]
        scale = scales[name]
        tight_prec = get_bit(th_dict[name] * scales[name])
        if precs[name][OUT_KEY] > tight_prec:
            op = mx.sym.Custom(op, precision=tight_prec,
                    name=N.n('clip'), op_type='cvm_clip')
            clip_name = op.attr('name')
            infer_shapes[clip_name] = infer_shapes[name]
            th_dict[clip_name] = th_dict[name]
            precs[clip_name] = { OUT_KEY: tight_prec }
            scales[clip_name] = scales[name]
            if name in precs and name in precs[name]:
                oprec = precs[name][name]
                del precs[name][name]
                precs[clip_name][clip_name] = oprec

        return op

    sym, params = topo_visit_transformer(symbol, params,
            _quant,
            infer_shapes=infer_shapes, th_dict=th_dict,
            precs=precs, scales=scales,
            op_input_precs=op_input_precs,
            shift_bits=shift_bits,
            softmax_lambd=softmax_lambd)

    def quantize_output(op, **kwargs):
        name = op.attr('name')
        th_dict = kwargs['th_dict']
        precs, scales = kwargs['precs'], kwargs['scales']

        # Requantize output symbol
        if name in precs and name in precs[name]:
            oprec = precs[name][name]
            os = scale(th_dict[name], oprec)
            op, oprec, os = requant(op, oprec, os, oname=name, **kwargs)

            oname = op.attr('name')
            th_dict[oname] = th_dict[name]
            precs[oname] = oprec
            scales[oname] = os
        return op

    return topo_visit_transformer(sym, params,
            quantize_output, th_dict=th_dict,
            precs=precs, scales=scales,
            shift_bits=shift_bits,
            softmax_lambd=softmax_lambd)

@N.register_nm("prepare_for_compile")
def prepare_for_compile(symbol, params):
    """ Customized graph-level topo pass definition.

        Equivalent graph transformation. 
        Convert quantized symbols into feasible ones for compile.
    """
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("prepare_for_compile", infer_shapes=infer_shapes))


@N.register_nm("cvm")
def to_cvm(symbol, params):
    """ Customized graph-level topo pass definition.

        Attribute cast to cvm symbol attributes.
        Mxnet symbols cast to CVM symbols using cvm-runtime libs.
    """
    infer_shapes = infer_shape(symbol, params)
    graph = {}
    for op in topo_sort(symbol):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        childs = [] if childs is None else childs
        childs = [get_node(c, graph) for c in childs]
        op = apply_pass("compile", infer_shapes=infer_shapes)(
            op, childs=childs, attr=attr,
            params=params, graph=graph)
        graph[name] = op

    nodes = []
    for sym in symbol:
        node = get_node(sym, graph)
        nodes.append(node)
    if len(nodes) > 1:
        return cvm.sym.Group(*nodes), params
    return nodes[0], params

# === symbol helper ===

@N.register_nm("fmi")
def fuse_multiple_inputs(sym, params):
    """ Customized graph-level topo pass definition.

        The original graph has multiple inputs.

        .. code-block:: none

           Inp1 Inp2  ...  Inpn
            |   |           |
            |   |           |
           ... ...    ...  ...

        Where 'Inp1', 'Inp2' and so on stands for the input symbol.

        The original graph will be transformed into:

        .. code-block:: none

                           data_sum
                         /     |      \ 
                        /      |       \ 
                       /       |        \ 
                      /        |         \ 
                     /         |          \ 
                    /          |           \ 
            slice_axis     slice_axis ...   slice_axis
            axis: 0        axis: 0          axis: 0
            begin:0        begin:end1       begin: end(n-1)
            end:pd1        end:end1+pd2     end:end(n-1)+pdn
               /               |               \ 
               |               |                |
            reshape        reshape          reshape
            shape:shp1     shape:shp2       shape:shpn
               |               |                |
               |               |                |
              Inp1            Inp2    ...      Inpn
               |               |                |
               |               |                |
              ...             ...     ...      ...

        where 'data_sum' is the tiled and concatenated form of all the inputs, 
        and 'pd1', 'pd2', ..., 'pdn' is the infer shape product of 'inp's, 
        and 'shp1', 'shp2', ..., 'shpn' is the infer shape if 'Inps'.
    """
    infer_shapes = infer_shape(sym, params)
    dim_sum, dim_per, dims = 0, {}, {}
    def _sum_input(node, params, **kwargs):
        name = node.attr('name')
        nonlocal dim_sum, dim_per, dims
        if is_inputs(node, params):
            dims[name] = infer_shapes[name][0]
            dot = np.product(dims[name])
            dim_per[name] = dot
            dim_sum += dot
    topo_visit_transformer(sym, params, _sum_input)

    assert len(dim_per) > 0, "no input in graph"
    if len(dim_per) == 1:
        return sym, params

    data_sum = mx.sym.var('data', shape=(dim_sum,))
    first, last = 0, 0
    def _change_node(op, params, graph, **kwargs):
        name = op.attr('name')
        if is_inputs(op, params):
            nonlocal first, last
            last = first + dim_per[name]
            op = mx.sym.slice(data_sum, name=N.n('slice'),
                    begin=(first,), end=(last,))
            op = mx.sym.reshape(op, name=N.n('reshape'),
                    shape=dims[name])
            first = last
        return op
    sym, params = topo_visit_transformer(sym, params, _change_node)
    return sym, params

def model_inputs(symbol, params):
    """ Customized graph-level topo pass definition.

        Count the number of model inputs.
    """
    input_count = 0
    def _count(op, params, graph):
        nonlocal input_count
        input_count += is_inputs(op, params)
    topo_visit_transformer(symbol, params, _count)
    return input_count

def name_duplicate_check(symbol, params):
    """ Check whether duplicate names exist in the graph.
    """
    names = set()
    for sym in topo_sort(symbol):
        name = sym.attr('name')
        assert name not in names, "duplicated name in graph: %s" % name
        names.add(name)

def params_unique(symbol, params):
    """ Remove duplicate keys params dict.
    """
    new_params = {s.attr('name'):params[s.attr('name')] \
            for s in topo_sort(symbol) if is_params(s, params)}
    return symbol, new_params

def input_name_replace(symbol, params):
    """ Customized graph-level topo pass definition.

        Replace the single input name to 'data'.
    """
    def _name_replace(op, params, graph):
        name, attr = op.attr('name'), op.list_attr()
        if is_inputs(op, params):
            op = mx.sym.var("data", attr=attr)
        return op
    return topo_visit_transformer(symbol, params, _name_replace)

@N.register_nm("fc")
def fuse_constant(symbol, params):
    """ Customized graph-level topo pass definition.

        Fix the constant operator shape with respect to the infer shape.
    """
    nparams = convert_params_dtype(params, dest_dtype="float32")

    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if is_var(op, params):
            pass
        elif childs is None:
            params[name] = get_nd_op(op_name)(**attr)
            attr = { 'precision': str(get_bit(params[name])) }
            op = mx.sym.var(name, shape=params[name].shape, attr=attr)
        elif all([is_params(c, params) for c in childs]):
            in_params = [params[c.attr('name')] for c in childs]
            params[name] = get_nd_op(op_name)(*in_params, **attr)
            attr = { 'precision': str(get_bit(params[name])) }
            op = mx.sym.var(name, shape=params[name].shape, attr=attr)
        return op

    sym, params = topo_visit_transformer(symbol, nparams, _impl)
    params = convert_params_dtype(params, dest_dtype="float64")
    return sym, params

@N.register_nm("ais")
def attach_input_shape(symbol, params, input_shapes):
    """ Customized graph-level topo pass definition.

        Attach the infer shapes for the graph.

        Parameters
        __________
        symbol : mxnet.symbol
            The graph symbols.
        params : dict
            The graph parameters dict.
        input_shapes : dict
            The name-shape map.

        Returns
        _______
        ret : tuple
            The symbol-param tuple after input shapes attachment.
    """
    assert isinstance(input_shapes, dict)
    def _impl(op, params, graph):
        name, attr = op.attr('name'), op.list_attr()
        if is_inputs(op, params) and name in input_shapes:
            op = mx.sym.var(name, shape=input_shapes[name], attr=attr)
        return op
    return topo_visit_transformer(symbol, params, _impl)

# TODO: reduce graph for adjacent broadcast_mul
def reduce_graph(symbol, params):
    pass

def infer_shape(symbol, params, input_shape=None):
    """ Customized graph-level topo pass definition.

        Collect the infer shapes from graph.

        Parameters
        __________
        symbol : mxnet.symbol
            The graph symbols.
        params : dict
            The graph parameters dict.
        input_shape : tuple
            The input shape of the data.

        Returns
        _______
        ret : dict
            The name-shape map.
    """
    infer_shapes = {}
    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        if is_params(op, params):
            oshp = [params[name].shape]
            op = mx.sym.var(name, shape=oshp[0])
        else:
            _, oshp, _ = op.infer_shape()

        if is_inputs(op, params):
            if input_shape is None:
                assert oshp is not None, "It seems that graph doesn't set \
                        input_shape, please invoke attach_input_shape first."
            else:
                oshp = [input_shape]
                op = mx.sym.var(name, shape=oshp[0])
        infer_shapes[name] = oshp
        return op
    topo_visit_transformer(symbol, params, _impl)
    return infer_shapes

def _collect_attribute(op, **kwargs):
    """ Collect the attribute value from the specified operator.

        Parameters
        __________
        op : mxnet.symbol
            The input operator.

        Returns
        _______
        ret : mxnet.symbol
            The output operator.
    """
    attr_name, func = kwargs['attr_name'], kwargs['func']
    func(op.attr(attr_name))
    return op

def collect_op_names(symbol, params):
    """ Customized graph-level topo pass definition.

        Collect all kinds of operators that exist in the graph.

        Parameters
        __________
        symbol : mxnet.symbol
            The graph symbols.
        params : dict
            The graph parameters dict.

        Returns
        _______
        ret : set
            The opname set.
    """
    op_names = set()
    _ = topo_visit_transformer(symbol, params, _collect_attribute,
            attr_name='op_name', func=op_names.add)
    return op_names

@N.register_nm("fmo")
def fuse_multiple_outputs(symbol, params):
    """ Customized symbol-level topo pass definition.

        Symbol-level multiple-outputs-fusion pass.

        .. code-block:: none

                           X
                           |
                           |
                      SliceChannel
                      /    |     \ 
                     /     |      \ 
                    /      |       \ 
                   /       |        \ 
                 A1        A2   ...  An

        The original graph will be transformed into:

        .. code-block:: none

                               X
                         /     |      \ 
                        /      |       \ 
                       /       |        \ 
                      /        |         \ 
                     /         |          \ 
                    /          |           \ 
            slice_axis     slice_axis ...   slice_axis
            axis: ich      axis: ich        axis: ich
            begin:0        begin:stride     begin: (n-1)*stride
            end:stride     end:2*stride     end:n*stride
               /               |               \ 
              A1               A2      ...      An

        where 'ich' is the attribute 'axis' along which to split, and

        .. math:
            dim = xshape(axis)

        .. math::
            stride = dim / num\_outputs

        where 'xshape' is infer shape of 'X' and 'num_ouputs' is the number of splits.

    """
    infer_shapes = infer_shape(symbol, params)
    channel, graph = {}, {}
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        if childs is not None:
            childs = [get_node(c, graph) for c in childs]
            sym = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if op_name == 'SliceChannel':
            # Only designed for special usei, thus
            # check of "SliceChannel" has not been added to _sym_check
            assert childs is not None and len(childs) == 1, \
                "Invalid Layer: %s, the 'SliceChannel' \
                operator must have exactly one input" % name
            axis = get_attr(attr, 'axis', 1)
            num_outputs = get_attr(attr, 'num_outputs')
            chchild_shape = infer_shapes[childs[0].attr('name')]
            eid = get_entry_id(childs[0])
            dim = chchild_shape[eid][axis]
            assert num_outputs > 0 and dim % num_outputs == 0, \
                "Invalid Layer: %s, the 'SliceChannel' operator \
                has a wrong attribute, 'num_outputs': %d" \
                % (name, num_outputs)
            stride = int(dim / num_outputs)
            interval = [(i * stride, (i + 1) * stride) \
                       for i in range(num_outputs)]
            channel[name] = [childs, axis, interval]
        elif childs is not None:
            is_split = False
            for i in range(len(childs)):
                cname = childs[i].attr('name')
                if cname in channel:
                    is_split = True
                    eid = get_entry_id(childs[i])
                    chchilds, axis, interval = channel[cname]
                    begin, end = interval[eid]
                    chattr = {'axis': axis, 'begin': begin, 'end': end}
                    slp_name = N.n('slice_axis')
                    if slp_name not in graph:
                        graph[slp_name] = mx.sym.slice_axis(*chchilds,
                                **chattr, name=slp_name)
                    childs[i] = graph[slp_name]
            if is_split:
                sym = get_mxnet_op(op_name)(*childs, **attr, name=name)
        graph[name] = sym

    nodes = [get_node(sym, graph) for sym in symbol]
    ret = mx.sym.Group(nodes) if len(nodes) > 1 else nodes[0]
    return ret, params

def _get_opt(out, lambd):
    """ Calibrate the MRT model after setting mrt data.

        .. math::
            mean\_v = mean(out)

        .. math::
            n = product(shape(out))

        .. math::
            sqrt\_n = sqrt(n)

        .. math::
            std = norm(out-mean) / sqrt\_n

        .. math::
            alpha = |mean| + lambd * std

        .. math::
            absmax = ||out||_2

        **Case 1. alpha < 0.95 * absmax**

        opt = alpha

        **Case 2. Other Cases**

        out = absmax

        Parameters
        __________
        out : nd.NDArray
            The graph level output.
        lambd : float
            Hyperparameter that set the alpha of data.

        Returns
        _______
        opt : float
            The opt value.
    """
    absmax = out.abs().max().asscalar()
    if lambd is None:
        return absmax
    mean = nd.mean(out).asscalar()
    sqrt_n = math.sqrt(np.product(out.shape))
    std = nd.norm(out - mean).asscalar() / sqrt_n
    alpha = abs(mean) + lambd * std

    #  pos_out = nd.abs(out)
    #  pos_mean = nd.mean(pos_out).asscalar()
    #  pos_std = nd.norm(pos_out - pos_mean).asscalar() / sqrt_n
    #  pos_alpha = abs(pos_mean) + lambd * pos_std

    opt = absmax
    if alpha < 0.95 * absmax:
        print ("mean, std = [", mean, std, "]", "alpha=", alpha,
               "absmax=", absmax)
        opt = alpha
    #  if opt > 30:
        #  print ("mean, std = [", mean, std, "]", "alpha=", alpha,
               #  "absmax=", absmax)
        #  print ("ABS mean, std = [", pos_mean, pos_std, "]",
               #  "alpha=", pos_alpha, "absmax=", absmax)
    return opt

def sym_calibrate(symbol, params, data, **kwargs):
    """ Customized graph-level topo pass definition.

        Calibrate the MRT model after setting mrt data.

        Parameters
        __________
        symbol : mxnet.symbol
            The graph symbols.
        params : dict
            The graph parameters dict.
        data : nd.NDArray
            The input data.

        Returns
        _______
        ret : dict
            The threshold dict after calibration.
    """
    logger = logging.getLogger('log.mrt')
    _, deps = topo_sort(symbol, logger=logger, with_deps=True)
    th_dict, out_cache = {}, {}
    ctx = kwargs.get('ctx', mx.cpu())
    logger.info("calibrate model outputs")
    nparams = convert_params_dtype(params, src_dtypes="float64",
            dest_dtype="float32")
    data = data.astype('float32')

    def _impl(op, params, graph, **kwargs):
        deps, old_ths = kwargs['deps'], kwargs['old_ths']
        logger = logging.getLogger('log.mrt.calibrate')
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if op_name == 'null':
            out = data if is_inputs(op, params) else params[name]
        elif childs is None:
            out = get_nd_op(op_name)(**attr)
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            for n, _ in cinfos:
                assert n in deps
                if name not in deps[n]:
                    # for op like: op = broadcast_mul(X, X)
                    # `cinfos` will have duplicate entries
                    # avoid removing more than once
                    continue
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        out = [out] if len(op) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]
        opts = float(_get_opt(out[0], kwargs['lambd']))
        if old_ths and name in old_ths:
            th_dict[name] = max(old_ths[name], opts)
        else:
            th_dict[name] = opts
            p = logger.debug if opts < 30 else logger.warn
            p("collect symbol %-40s out_shape=%-20s th_dict: (%s)",
                    name, [o.shape for o in out], th_dict[name])

    topo_visit_transformer(symbol, nparams, _impl, logger=logger,
            deps=deps, data=data, **kwargs)
    out_cache.clear()

    return th_dict

def convert_params_dtype(params, src_dtypes=["float32", "float64"],
        dest_dtype="float64"):
    """ Convert the source data type into to target data type.

        Parameters
        __________
        params : nd.NDArray
            The input data to be converted.
        src_dtypes : str or list of str
            The source data type(s)

        Returns
        _______
        ret : int
            The precision of the input.
    """
    if not params:
        return {}
    if isinstance(src_dtypes, str):
        src_dtypes = [src_dtypes]
    nparams = {}
    for k, v in params.items():
        dtype = v.dtype.__name__
        if dtype != dest_dtype and dtype in src_dtypes:
            nparams[k] = v.astype(dest_dtype)
        else:
            nparams[k] = v
    return nparams
