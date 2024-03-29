import builtins
import datetime
import os
import time
from collections import defaultdict, deque

import paddle
import paddle.amp as amp
import paddle.distributed as dist
from paddle.fluid.dygraph.parallel import ParallelEnv

try:
    import wandb
    has_wandb = True

    class WandbLogger(object):

        def __init__(self, args, **kwargs):
            wandb.init(config=args, **kwargs)

        def set_step(self, step=None):
            if step is not None:
                self.step = step
            else:
                self.step += 1

        def update(self, metrics):
            log_dict = dict()
            for k, v in metrics.items():
                if v is None:
                    continue
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                log_dict[k] = v
            wandb.log(log_dict, step=self.step)

        def flush(self):
            pass

except ImportError:
    has_wandb = False


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = paddle.to_tensor([self.count, self.total], dtype=paddle.float64)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype=paddle.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t", log_file=""):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_file = log_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, num_training_steps_per_epoch=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        steps = num_training_steps_per_epoch or len(iterable)
        space_fmt = ':' + str(len(str(steps))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == steps - 1:
                eta_seconds = iter_time.global_avg * (steps - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                msg = log_msg.format(
                    i, steps, eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time))
                print(msg)
                if self.log_file and is_main_process():
                    with open(self.log_file, mode="a", encoding="utf-8") as f:
                        f.write(msg + '\n')
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / steps))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


class NativeScalerWithGradNormCount:

    def __init__(self):
        self._scaler = amp.GradScaler(init_loss_scaling=2.**16)

    def __call__(self, loss, optimizer, parameters=None, create_graph=False, update_grad=True):
        scaled = self._scaler.scale(loss)  # scale the loss
        scaled.backward(retain_graph=create_graph)  # do backward
        if update_grad:
            assert parameters is not None
            self._scaler.unscale_(optimizer)
            norm = get_grad_norm_(parameters)
            self._scaler.minimize(optimizer, scaled)
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return paddle.to_tensor(0.)
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = paddle.norm(paddle.stack([paddle.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
    return total_norm


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def is_dist_avail_and_initialized():
    return ParallelEnv().world_size > 1


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return ParallelEnv().local_rank


def is_main_process():
    return get_rank() == 0


def init_distributed_mode():
    dist.init_parallel_env()
    setup_for_distributed(get_local_rank() == 0)


def save_on_master(*args, **kwargs):
    if is_main_process():
        paddle.save(*args, **kwargs)


def save_model(args, epoch, model_without_ddp, model_ema=None, optimizer=None, loss_scaler=None, tag=None):
    to_save = {
        'model': model_without_ddp.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if optimizer is not None:
        to_save['optimizer'] = optimizer.state_dict()
    if model_ema is not None:
        to_save['model_ema'] = model_ema.state_dict()
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()
    os.makedirs(args.output, exist_ok=True)
    save_on_master(to_save, os.path.join(args.output, f'checkpoint-{tag or epoch}.pd'))


def load_model(args, model_without_ddp, model_ema=None, optimizer=None, loss_scaler=None):
    start_epoch = 0
    if args.resume:
        checkpoint = paddle.load(args.resume)
        model_without_ddp.set_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if getattr(args, 'start_epoch', None) is None and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if model_ema is not None and 'model_ema' in checkpoint:
            model_ema.set_state_dict(checkpoint['model_ema'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer'])
        if loss_scaler is not None and 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
    return start_epoch
