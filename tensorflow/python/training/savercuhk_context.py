from tensorflow.python import tf_export


@tf_export("train.TFTunerContext")
class TFTunerContext:
    @classmethod
    def init_context(cls, num_task, task_index):
        cls.num_task = num_task
        cls.task_index = task_index
        cls.is_init = True

    @classmethod
    def get_num_task(cls):
        return cls.num_task

    @classmethod
    def get_task_index(cls):
        return cls.task_index

    @classmethod
    def get_is_init(cls):
        return cls.is_init