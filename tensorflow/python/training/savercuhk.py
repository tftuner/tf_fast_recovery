from tensorflow.python.client import session
from tensorflow.python.ops import variables, array_ops, state_ops, control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.training.savercuhk_context import TFTunerContext


def get_alive_node():
    ret = []
    for task_index in range(0, TFTunerContext.get_num_task()):
        if task_index == TFTunerContext.get_task_index():
            continue

        ret.append("/job:worker/task:{}".format(task_index))
    return ret


def get_my_device():
    return "/job:worker/task:{}".format(TFTunerContext.get_task_index())


def get_var_device_mapping(variables, devices):
    ret = {}
    for idx, var in enumerate(variables):
        ret[var] = devices[idx % len(devices)]
    return ret


def replicate_variable(v, replica_name):
    return variables.Variable(array_ops.zeros(v.shape, dtype=v.dtype),
                              shape=v.shape,
                              dtype=v.dtype,
                              name=replica_name)


def get_restore_graph_1(target_variables, variable_device_mapping, dst_device):
    copy_ops = []
    with ops.Graph().as_default() as g:
        for variable in target_variables:
            assert(variable_device_mapping[variable])
            with ops.device(variable_device_mapping[variable]):
                a = replicate_variable(variable, variable.op.name)

            with ops.device(dst_device):
                tmp_a = replicate_variable(variable, "tmp_" + variable.op.name)

            copy_ops.append(state_ops.assign(tmp_a, a))
        return g, control_flow_ops.group(copy_ops)


def get_restore_graph_2(target_variables, dst_device):
    copy_ops = []
    with ops.Graph().as_default() as g:
        for variable in target_variables:
            with ops.device(dst_device):
                # tmp_a = ops.Variable(0, name="tmp_" + variable.op.name)
                # a = ops.Variable(0, name=variable.op.name)
                tmp_a = replicate_variable(variable, "tmp_" + variable.op.name)
                a = replicate_variable(variable, variable.op.name)

                copy_ops.append(state_ops.assign(a, tmp_a))
        return g, control_flow_ops.group(copy_ops)


def get_my_device_name(graph):
    return graph.get_collection("global_step")[0].device


def restore(sess, save_path, callback_org_restore):

    if not TFTunerContext.get_is_init():
        print("Context is not init")
        return callback_org_restore(sess, save_path)
    # Assume 0 wont die. if yes, recover from checkpoint
    if TFTunerContext.get_task_index() == 0:
        return callback_org_restore(sess, save_path)

    # get alive node's device
    _restore(sess)


def _restore(sess):
    # get alive node's device
    alive_node_device = get_alive_node()

    variables = sess.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    # create a var->node mapping
    dict_var_device = get_var_device_mapping(variables, alive_node_device)

    session_config = sess._config
    session_config.device_filters[:] = []

    restore_p1_graph, restore_p1_copy_op = get_restore_graph_1(variables, dict_var_device,
                                                                    get_my_device())
    with session.Session(target=sess._target, graph=restore_p1_graph, config=session_config) as sess:
        sess.run(restore_p1_copy_op)
    restore_p2_graph, restore_p2_copy_op = get_restore_graph_2(variables, get_my_device())
    with session.Session(target=sess._target, graph=restore_p2_graph) as sess:
        sess.run(restore_p2_copy_op)


class DummyCheckpoint:
    def __init__(self):
        new_restore_ops_callback = []

class CustomLoadStatus:

    def __init__(self):
        self._checkpoint = DummyCheckpoint()
    def assert_consumed(self):
        pass

    def assert_existing_objects_matched(self):
        pass

    def assert_nontrivial_match(self):
        pass

    def run_restore_ops(self, session=None):
        _restore(session)

    def initialize_or_restore(self, session=None):
        pass


def hack_tracable_checkpoint_restore(save_path, callback_org_restore):
    if not TFTunerContext.get_is_init():
        print("Context is not init")
        return callback_org_restore(save_path)
    # Assume 0 wont die. if yes, recover from checkpoint
    if TFTunerContext.get_task_index() == 0:
        return callback_org_restore(save_path)

    # get alive node's device
    return CustomLoadStatus()